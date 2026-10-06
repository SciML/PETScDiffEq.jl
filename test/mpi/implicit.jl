using MPI, PETScDiffEq, SciMLBase, SparseArrays, Test
using LinearAlgebra: Diagonal
using PETScDiffEq: PETSc, AutoFiniteDiff, AutoForwardDiff, DiffEqBase
using SciMLBase: ODEProblem, ODEFunction, DAEProblem, DAEFunction, SplitODEProblem, ReturnCode,
    solve

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
# PetscInitialize is collective, and the serial reference solves run on single ranks.
for S in (Float64, Float32)
    PETSc.initialize(PETSc.getlib(; PetscScalar = S))
end

const N = 23
const SPAN = (0.0, 0.1)
const TOL = (abstol = 1.0e-8, reltol = 1.0e-8)
const SERIAL_GAP = 5.0e-10
const INIT_GAP = 1.0e-14
const thrower = nranks - 1

function uneven(n)
    counts = floor.(Int, n .* (1:nranks) ./ sum(1:nranks))
    counts[end] += n - sum(counts)
    return counts
end
even(n) = [n ÷ nranks + (r < n % nranks) for r in 0:(nranks - 1)]
owned(counts) = (lo = sum(counts[1:rank]) + 1; lo:(lo + counts[rank + 1] - 1))

const counts = uneven(N)
const rows = owned(counts)

function gathered(u::AbstractVector{<:Number}, counts)
    out = rank == 0 ? zeros(eltype(u), sum(counts)) : nothing
    MPI.Gatherv!(u, rank == 0 ? MPI.VBuffer(out, counts) : nothing, comm)
    return out
end
gathered(sol::SciMLBase.AbstractODESolution, counts) = [gathered(u, counts) for u in sol.u]

same_everywhere(x) = MPI.bcast(x, 0, comm) == x
anywhere(b) = MPI.Allreduce(b, |, comm)
maxdiff(a, b) = maximum(maximum(abs, x - y) for (x, y) in zip(a, b))

function caught(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

remote(e) = e isa ErrorException && occursin("another rank", e.msg)
raised(e, what) = rank == thrower ? e isa ErrorException && occursin(what, e.msg) : remote(e)
refused(f, what) = (e = caught(f); e isa ArgumentError && occursin(what, e.msg))

const dx = 1 / (N + 1)
heat0(idx) = sinpi.(idx .* dx) .+ 0.5 .* sinpi.(3 .* idx .* dx)
eigval(k) = -4 / dx^2 * sinpi(k * dx / 2)^2
heat_exact(t) = exp(eigval(1) * t) .* sinpi.((1:N) .* dx) .+
    0.5 * exp(eigval(3) * t) .* sinpi.(3 .* (1:N) .* dx)

function laplacian!(du, u, left, right)
    n = length(u)
    for i in 1:n
        l = i == 1 ? left : u[i - 1]
        r = i == n ? right : u[i + 1]
        du[i] = (l - 2u[i] + r) / dx^2
    end
    return nothing
end

function halo(u)
    left = rank == 0 ? MPI.PROC_NULL : rank - 1
    right = rank == nranks - 1 ? MPI.PROC_NULL : rank + 1
    gl, gr = zeros(eltype(u), 1), zeros(eltype(u), 1)
    MPI.Sendrecv!(u[1:1], gr, comm; dest = left, source = right)
    MPI.Sendrecv!(u[end:end], gl, comm; dest = right, source = left)
    return gl[1], gr[1]
end

heat!(du, u, p, t) = laplacian!(du, u, halo(u)...)
heat_serial!(du, u, p, t) = laplacian!(du, u, 0.0, 0.0)

neighbours(i) = max(1, i - 1):min(N, i + 1)

function heat_proto(idx)
    I = [k for (k, i) in enumerate(idx) for _ in neighbours(i)]
    J = [j for i in idx for j in neighbours(i)]
    return sparse(I, J, ones(length(I)), length(idx), N)
end

heat_jac(idx, scale = ones(N)) = function (J, u, p, t)
    for (k, i) in enumerate(idx), j in neighbours(i)
        J[k, j] = scale[i] * (i == j ? -2 : 1) / dx^2
    end
    return nothing
end

function heat_function(f, idx; jac, scale = ones(N), kw...)
    jac || return ODEFunction(f; jac_prototype = heat_proto(idx), kw...)
    return ODEFunction(f; jac = heat_jac(idx, scale), jac_prototype = heat_proto(idx), kw...)
end

const SOURCES = ((true, AutoForwardDiff()), (false, AutoFiniteDiff()))

function throwing(f, what)
    return function (args...)
        f(args...)
        rank == thrower && args[end] > 0.02 && error("$what threw on rank $rank")
        return nothing
    end
end

function compare(prob, alg, ref_prob, ref_alg, exact, err; layout = counts, kw...)
    sol = solve(prob, alg; kw...)
    @test sol.retcode == ReturnCode.Success
    @test same_everywhere(sol.t)
    u = gathered(sol.u[end], layout)
    if rank == 0
        ref = solve(ref_prob, ref_alg; kw...)
        @test sol.t[end] == ref.t[end]
        @test maximum(abs, u - ref.u[end]) <= SERIAL_GAP
        @test maximum(abs, u - exact) <= err
    end
    return sol
end

const METHODS = (
    ((c; kw...) -> TSImplicit("bdf"; comm = c, kw...), 1.5e-6),
    ((c; kw...) -> TSRosW(; comm = c, kw...), 6.0e-9),
    ((c; kw...) -> TSARKIMEX(; comm = c, kw...), 4.0e-8),
)

@testset "MPI implicit, $nranks ranks" begin
    @testset "autodiff defaults to PETSc's colouring" begin
        @test TSRosW(; comm).autodiff isa AutoFiniteDiff
        @test TSGeneric("alpha"; comm).autodiff isa AutoFiniteDiff
        @test TSImplicit("bdf").autodiff isa AutoForwardDiff
        @test TSGeneric("alpha").autodiff isa AutoForwardDiff
    end

    @testset "1-D heat with a local-row jac and with colouring" begin
        heat(idx, f; jac) = ODEProblem(heat_function(f, idx; jac), heat0(idx), SPAN)
        for (make, err) in METHODS, (jac, ad) in SOURCES
            sol = compare(
                heat(rows, heat!; jac), make(comm), heat(1:N, heat_serial!; jac),
                make(MPI.COMM_SELF; autodiff = ad), heat_exact(SPAN[2]), err; TOL...,
            )
            @test (sol.stats.njacs > 0) == jac
        end
        sol = compare(
            heat(rows, heat!; jac = true), TSImplicit("bdf"; comm),
            heat(1:N, heat_serial!; jac = true), TSImplicit("bdf"), heat_exact(SPAN[2]), 1.5e-6;
            tstops = [0.05], TOL...,
        )
        @test 0.05 in sol.t
        sol = solve(heat(rows, heat!; jac = true), TSImplicit("bdf"; comm); saveat = 0.02, TOL...)
        us = gathered(sol, counts)
        if rank == 0
            ref = solve(heat(1:N, heat_serial!; jac = true), TSImplicit("bdf"); saveat = 0.02, TOL...)
            @test sol.t == ref.t
            @test maxdiff(us, ref.u) <= SERIAL_GAP
            @test maxdiff(us, heat_exact.(sol.t)) <= 4.0e-6
        end
    end

    @testset "an implicit TSGeneric" begin
        heat(idx, f; jac) = ODEProblem(heat_function(f, idx; jac), heat0(idx), SPAN)
        cases = (("alpha", 3.0e-8, (; dt = 1.0e-4)), ("dirk", 5.0e-7, (; dt = 1.0e-4, TOL...)))
        for (type, err, kw) in cases, (jac, ad) in SOURCES
            sol = compare(
                heat(rows, heat!; jac), TSGeneric(type; comm), heat(1:N, heat_serial!; jac),
                TSGeneric(type; autodiff = ad), heat_exact(SPAN[2]), err; kw...,
            )
            @test (sol.stats.njacs > 0) == jac
        end
        irk_counts = even(N)
        idx = owned(irk_counts)
        irk(idx, f) = ODEProblem(heat_function(f, idx; jac = true), heat0(idx), SPAN)
        sol = solve(irk(idx, heat!), TSGeneric("irk"; comm); dt = 1.0e-3)
        @test sol.retcode == ReturnCode.Success
        us = gathered(sol, irk_counts)
        if rank == 0
            ref = solve(irk(1:N, heat_serial!), TSGeneric("irk"); dt = 1.0e-3)
            @test sol.t == ref.t
            @test maxdiff(us, ref.u) <= SERIAL_GAP
        end
    end

    @testset "TSIRK on PETSc's own split of the state" begin
        irk_counts = even(N)
        idx = owned(irk_counts)
        heat(idx, f) = ODEProblem(heat_function(f, idx; jac = true), heat0(idx), SPAN)
        for s in (1, 2, 3)
            sol = solve(heat(idx, heat!), TSIRK(s; comm); dt = 1.0e-3)
            @test sol.retcode == ReturnCode.Success
            us = gathered(sol, irk_counts)
            if rank == 0
                ref = solve(heat(1:N, heat_serial!), TSIRK(s); dt = 1.0e-3)
                @test sol.t == ref.t
                @test maxdiff(us, ref.u) <= SERIAL_GAP
                @test maximum(abs, us[end] - heat_exact(SPAN[2])) <= (4.0e-6, 1.0e-10, 5.0e-12)[s]
            end
        end
    end

    @testset "TSIRK on any split of the state" begin
        hollow = copy(counts)
        nranks > 1 && (hollow[2] += hollow[1]; hollow[1] = 0)
        function spread(blocks)
            full = findall(>(0), blocks) .- 1
            k = findfirst(==(rank), full)
            k === nothing && return (du, u, p, t) -> nothing
            left = k == 1 ? MPI.PROC_NULL : full[k - 1]
            right = k == length(full) ? MPI.PROC_NULL : full[k + 1]
            return function (du, u, p, t)
                gl, gr = zeros(eltype(u), 1), zeros(eltype(u), 1)
                MPI.Sendrecv!(u[1:1], gr, comm; dest = left, source = right)
                MPI.Sendrecv!(u[end:end], gl, comm; dest = right, source = left)
                return laplacian!(du, u, gl[1], gr[1])
            end
        end
        split_heat(blocks; jac) = ODEProblem(
            heat_function(spread(blocks), owned(blocks); jac), heat0(owned(blocks)), SPAN,
        )
        serial_heat(; jac) = ODEProblem(heat_function(heat_serial!, 1:N; jac), heat0(1:N), SPAN)
        stepped(prob, alg) = solve(prob, alg; dt = 1.0e-3)
        saved(prob, alg) = solve(prob, alg; dt = 1.0e-3, saveat = 0.0125)
        halve = SciMLBase.DiscreteCallback((u, t, i) -> t == 0.05, i -> (i.u .*= 0.5))
        halved(prob, alg) = solve(prob, alg; dt = 1.0e-3, callback = halve, tstops = [0.05])
        function restarted(prob, alg)
            integ = SciMLBase.init(prob, alg; dt = 1.0e-3)
            foreach(_ -> SciMLBase.step!(integ), 1:10)
            SciMLBase.set_u!(integ, 0.5 .* integ.u)
            return SciMLBase.solve!(integ)
        end
        ad = AutoForwardDiff()
        cases = (
            (stepped, TSIRK(1; comm), TSIRK(1), true),
            (stepped, TSIRK(2; comm), TSIRK(2), true),
            (stepped, TSIRK(3; comm), TSIRK(3), true),
            (stepped, TSIRK(2; comm, autodiff = ad), TSIRK(2; autodiff = ad), false),
            (stepped, TSGeneric("irk"; comm), TSGeneric("irk"), true),
            (saved, TSIRK(3; comm), TSIRK(3), true),
            (halved, TSIRK(3; comm), TSIRK(3), true),
            (restarted, TSIRK(3; comm), TSIRK(3), true),
        )
        splits = nranks > 1 ? (counts, hollow) : (counts,)
        for blocks in splits, (solve_with, alg, serial, jac) in cases
            sol = solve_with(split_heat(blocks; jac), alg)
            @test sol.retcode == ReturnCode.Success
            @test all(u -> length(u) == blocks[rank + 1], sol.u)
            us = gathered(sol, blocks)
            if rank == 0
                ref = solve_with(serial_heat(; jac), serial)
                @test sol.t == ref.t
                @test maxdiff(us, ref.u) <= SERIAL_GAP
                if solve_with === stepped && alg isa TSIRK
                    err = (4.0e-6, 1.0e-10, 5.0e-12)[alg.nstages]
                    @test maximum(abs, us[end] - heat_exact(SPAN[2])) <= err
                end
            end
        end
        bad_f = heat_function(throwing(heat!, "f"), rows; jac = true)
        bad_jac = ODEFunction(
            heat!; jac = throwing(heat_jac(rows), "jac"), jac_prototype = heat_proto(rows),
        )
        for (what, fn) in (("f", bad_f), ("jac", bad_jac))
            prob = ODEProblem(fn, heat0(rows), SPAN)
            @test raised(caught(() -> solve(prob, TSIRK(2; comm); dt = 1.0e-3)), what)
        end
        nan_after(f, i) = function (du, u, p, t)
            f(du, u, p, t)
            t > 0.05 && i > 0 && (du[i] = NaN)
            return nothing
        end
        nan_fn = heat_function(nan_after(heat!, rank == thrower ? 1 : 0), rows; jac = true)
        failed = solve(ODEProblem(nan_fn, heat0(rows), SPAN), TSIRK(2; comm); dt = 1.0e-3)
        @test failed.retcode == ReturnCode.ConvergenceFailure
        failed_us = gathered(failed, counts)
        if rank == 0
            first_row = sum(counts[1:thrower]) + 1
            serial_fn = heat_function(nan_after(heat_serial!, first_row), 1:N; jac = true)
            failed_ref = solve(ODEProblem(serial_fn, heat0(1:N), SPAN), TSIRK(2); dt = 1.0e-3)
            @test failed_ref.retcode == ReturnCode.ConvergenceFailure
            @test failed.t == failed_ref.t
            @test maxdiff(failed_us, failed_ref.u) <= SERIAL_GAP
        end
        mass = Diagonal(fill(2.0, length(rows)))
        massive = ODEProblem(
            heat_function(heat!, rows; jac = true, mass_matrix = mass), heat0(rows), SPAN,
        )
        @test refused(() -> solve(massive, TSIRK(2; comm); dt = 1.0e-3), "a mass matrix with TSIRK")
    end

    @testset "a DAEProblem" begin
        residual(f) = (r, du, u, p, t) -> (f(r, u, p, t); r .= du .- r; nothing)
        dae_jac(idx) = function (J, du, u, p, gamma, t)
            heat_jac(idx)(J, u, p, t)
            for (k, i) in enumerate(idx), j in neighbours(i)
                J[k, j] = (i == j) * gamma - J[k, j]
            end
            return nothing
        end
        function dae(idx, f; jac)
            du0 = similar(heat0(idx))
            f(du0, heat0(idx), nothing, 0.0)
            proto = heat_proto(idx)
            fn = jac ? DAEFunction(residual(f); jac = dae_jac(idx), jac_prototype = proto) :
                DAEFunction(residual(f); jac_prototype = proto)
            return DAEProblem(fn, du0, heat0(idx), SPAN)
        end
        for (jac, ad) in SOURCES
            compare(
                dae(rows, heat!; jac), TSDAE("bdf"; comm), dae(1:N, heat_serial!; jac),
                TSDAE("bdf"; autodiff = ad), heat_exact(SPAN[2]), 1.5e-6; TOL...,
            )
        end
    end

    @testset "a Diagonal mass matrix" begin
        d = 1 .+ (1:N) ./ N
        scaled(f) = (du, u, p, t) -> (f(du, u, p, t); du .*= p; nothing)
        massive(idx, f; jac) = ODEProblem(
            heat_function(scaled(f), idx; jac, scale = d, mass_matrix = Diagonal(d[idx])),
            heat0(idx), SPAN, d[idx],
        )
        for (make, err) in METHODS[1:2], (jac, ad) in SOURCES
            compare(
                massive(rows, heat!; jac), make(comm), massive(1:N, heat_serial!; jac),
                make(MPI.COMM_SELF; autodiff = ad), heat_exact(SPAN[2]), err; TOL...,
            )
        end
    end

    @testset "a sparse mass matrix of this rank's rows" begin
        # The consistent mass matrix of linear elements shares the Laplacian's eigenvectors.
        mu(k) = 2 / 3 + cospi(k * dx) / 3
        exact = exp(eigval(1) / mu(1) * SPAN[2]) .* sinpi.((1:N) .* dx) .+
            0.5 * exp(eigval(3) / mu(3) * SPAN[2]) .* sinpi.(3 .* (1:N) .* dx)
        function fem_mass(idx)
            I = [k for (k, i) in enumerate(idx) for _ in neighbours(i)]
            J = [j for i in idx for j in neighbours(i)]
            V = [i == j ? 2 / 3 : 1 / 6 for i in idx for j in neighbours(i)]
            return sparse(I, J, V, length(idx), N)
        end
        fem(idx, f; jac) =
            ODEProblem(heat_function(f, idx; jac, mass_matrix = fem_mass(idx)), heat0(idx), SPAN)
        for (make, err) in METHODS[1:2], (jac, ad) in SOURCES
            sol = compare(
                fem(rows, heat!; jac), make(comm), fem(1:N, heat_serial!; jac),
                make(MPI.COMM_SELF; autodiff = ad), exact, err; TOL...,
            )
            @test (sol.stats.njacs > 0) == jac
        end
        # A rank may give its rows as a Diagonal, while another gives them sparse.
        d = 1 .+ (1:N) ./ N
        scaled(f) = (du, u, p, t) -> (f(du, u, p, t); du .*= p; nothing)
        mixed(idx, f, mass; jac) = ODEProblem(
            heat_function(scaled(f), idx; jac, scale = d, mass_matrix = mass), heat0(idx), SPAN,
            d[idx],
        )
        n = length(rows)
        mass = rank == thrower ? sparse(1:n, rows, d[rows], n, N) : Diagonal(d[rows])
        for (jac, ad) in SOURCES
            compare(
                mixed(rows, heat!, mass; jac), TSImplicit("bdf"; comm),
                mixed(1:N, heat_serial!, Diagonal(d); jac), TSImplicit("bdf"; autodiff = ad),
                heat_exact(SPAN[2]), 1.5e-6; TOL...,
            )
        end
        wrong = rank == thrower ? sparse(1:n, rows, ones(n), n, N + 1) : fem_mass(rows)
        prob = ODEProblem(
            heat_function(heat!, rows; jac = true, mass_matrix = wrong), heat0(rows), SPAN,
        )
        e = caught(() -> solve(prob, TSImplicit("bdf"; comm)))
        @test rank == thrower ? e isa ArgumentError && occursin("must be $n x $N", e.msg) :
            remote(e)
    end

    @testset "algebraic rows with a zero on the diagonal" begin
        cell_counts = uneven(8)
        cells = owned(cell_counts)
        function cell!(du, u, p, t)
            for c in 1:(length(u) ÷ 3)
                b1, b2, a = u[3c - 2], u[3c - 1], u[3c]
                du[3c - 2], du[3c - 1], du[3c] = b2 - a, b1 - 2a, b2 - b1
            end
            return nothing
        end
        entries = ((1, 2, 1.0), (1, 3, -1.0), (2, 1, 1.0), (2, 3, -2.0), (3, 1, -1.0), (3, 2, 1.0))
        function cell_problem(cells; jac, sparse_mass = false, kw...)
            n, offset = 3length(cells), 3(first(cells) - 1)
            I = [3(c - 1) + i for c in 1:length(cells) for (i, _, _) in entries]
            J = [offset + 3(c - 1) + j for c in 1:length(cells) for (_, j, _) in entries]
            V = [v for _ in cells for (_, _, v) in entries]
            proto = sparse(I, J, ones(length(I)), n, 3sum(cell_counts))
            cell_jac!(Jm, u, p, t) = (foreach((i, j, v) -> Jm[i, j] = v, I, J, V); nothing)
            m = repeat([0.0, 0.0, 1.0], length(cells))
            M = sparse_mass ? sparse(1:n, offset .+ (1:n), m, n, 3sum(cell_counts)) : Diagonal(m)
            fn = jac ?
                ODEFunction(cell!; jac = cell_jac!, jac_prototype = proto, mass_matrix = M, kw...) :
                ODEFunction(cell!; jac_prototype = proto, mass_matrix = M, kw...)
            return ODEProblem(fn, repeat([2.0, 1.0, 1.0], length(cells)), (0.0, 1.0))
        end
        exact = repeat(exp(-1) .* [2.0, 1.0, 1.0], sum(cell_counts))
        for sparse_mass in (false, true), (jac, ad) in SOURCES
            compare(
                cell_problem(cells; jac, sparse_mass), TSImplicit("bdf"; comm),
                cell_problem(1:sum(cell_counts); jac, sparse_mass),
                TSImplicit("bdf"; autodiff = ad), exact, 6.0e-6; layout = 3 .* cell_counts, TOL...,
            )
        end
        u0 = repeat([2.0, 1.0, 1.0], length(cells))
        rank == thrower && (u0[1] += 1)
        for sparse_mass in (false, true)
            off = SciMLBase.remake(cell_problem(cells; jac = true, sparse_mass); u0)
            @test caught(() -> solve(off, TSImplicit("bdf"; comm); TOL...)) isa
                SciMLBase.CheckInitFailureError
        end
        whole = repeat([2.0, 1.0, 1.0], sum(cell_counts))
        whole[3sum(cell_counts[1:thrower]) + 1] += 1
        for sparse_mass in (false, true), (jac, ad) in SOURCES,
                init in (DiffEqBase.BrownFullBasicInit(), DiffEqBase.ShampineCollocationInit())
            off = SciMLBase.remake(cell_problem(cells; jac, sparse_mass); u0)
            sol = solve(off, TSImplicit("bdf"; comm); initializealg = init, TOL...)
            @test sol.retcode == ReturnCode.Success
            start = gathered(sol.u[1], 3 .* cell_counts)
            if rank == 0
                serial = SciMLBase.remake(
                    cell_problem(1:sum(cell_counts); jac, sparse_mass); u0 = whole,
                )
                serial_alg = TSImplicit("bdf"; autodiff = ad)
                ref = solve(serial, serial_alg; initializealg = init, TOL...)
                @test start != whole
                @test maximum(abs, start - ref.u[1]) <= INIT_GAP
            end
        end
        data = SciMLBase.OverrideInitData(
            SciMLBase.NonlinearProblem((u, p) -> u .- 1, [0.0]), nothing, nothing, nothing,
        )
        own = cell_problem(cells; jac = true, initialization_data = data)
        for init in (DiffEqBase.DefaultInit(), SciMLBase.OverrideInit())
            @test refused(
                () -> solve(own, TSImplicit("bdf"; comm); initializealg = init, TOL...),
                "OverrideInit",
            )
        end
        @test solve(
            own, TSImplicit("bdf"; comm); initializealg = SciMLBase.CheckInit(), TOL...,
        ).retcode == ReturnCode.Success
    end

    @testset "a rank whose own mass block is the identity" begin
        function pairs!(du, u, ks)
            for (i, k) in enumerate(ks)
                a = 2i - 1
                du[a] = -u[a]
                du[a + 1] = k > 0 ? u[a + 1] - u[a] : -u[a + 1]
            end
            return nothing
        end
        function pairs_jac!(Jm, u, ks)
            for (i, k) in enumerate(ks)
                a, c = 2i - 1, 2k + 1
                Jm[a, c] = -1.0
                Jm[a + 1, c] = k > 0 ? -1.0 : 0.0
                Jm[a + 1, c + 1] = k > 0 ? 1.0 : -1.0
            end
            return nothing
        end
        function pairs(ks, y0; jac)
            I = [2i - 1 + d for i in 1:length(ks) for d in (0, 1, 1)]
            J = [2k + 1 + d for k in ks for d in (0, 0, 1)]
            proto = sparse(I, J, ones(length(I)), 2length(ks), 2nranks)
            M = Diagonal([k > 0 && d == 1 ? 0.0 : 1.0 for k in ks for d in (0, 1)])
            f = (du, u, p, t) -> pairs!(du, u, ks)
            jac_f = (Jm, u, p, t) -> pairs_jac!(Jm, u, ks)
            fn = jac ? ODEFunction(f; jac = jac_f, jac_prototype = proto, mass_matrix = M) :
                ODEFunction(f; jac_prototype = proto, mass_matrix = M)
            return ODEProblem(fn, repeat([1.0, y0], length(ks)), (0.0, 1.0))
        end
        bdf = TSImplicit("bdf"; comm)
        sol = solve(pairs([rank], 1.0; jac = false), bdf; dt = 1.0e-3, TOL...)
        @test sol.retcode == ReturnCode.Success
        @test maximum(abs, sol.u[end] .- exp(-1)) <= 3.0e-6
        off = caught(() -> solve(pairs([rank], 2.0; jac = false), bdf; dt = 1.0e-3, TOL...))
        @test nranks == 1 ? off === nothing : off isa SciMLBase.CheckInitFailureError
        for (jac, ad) in SOURCES
            compare(
                pairs([rank], 1.0; jac), bdf, pairs(0:(nranks - 1), 1.0; jac),
                TSImplicit("bdf"; autodiff = ad), fill(exp(-1), 2nranks), 3.0e-6;
                layout = fill(2, nranks), TOL...,
            )
        end
    end

    @testset "BrownFullBasicInit and ShampineCollocationInit solve on every rank" begin
        brown = DiffEqBase.BrownFullBasicInit()
        shampine = DiffEqBase.ShampineCollocationInit()
        algebraic(i) = i % 4 == 0
        function chain(idx, serial, form; jac, hopeless = false, throws = false, mass = nothing)
            function f!(du, u, p, t)
                left, right = serial ? (0.0, 0.0) : halo(u)
                for (k, i) in enumerate(idx)
                    l = k == 1 ? left : u[k - 1]
                    r = k == length(u) ? right : u[k + 1]
                    du[k] = !algebraic(i) ? l - 2u[k] + r :
                        hopeless && i == 20 ? u[k]^2 + 1 : u[k]^3 + u[k] - (l + r) / 2 - 0.1
                end
                throws && rank == thrower && u != heat0(idx) && error("f threw on rank $rank")
                return nothing
            end
            function jac!(J, u, p, t)
                for (k, i) in enumerate(idx), j in neighbours(i)
                    J[k, j] = algebraic(i) ? (i == j ? 3u[k]^2 + 1 : -0.5) : (i == j ? -2 : 1)
                end
                return nothing
            end
            n, m = length(idx), [algebraic(i) ? 0.0 : 1.0 for i in idx]
            proto = heat_proto(idx)
            if form == :dae
                residual!(r, du, u, p, t) = (f!(r, u, p, t); r .= m .* du .- r; nothing)
                function dae_jac!(J, du, u, p, gamma, t)
                    jac!(J, u, p, t)
                    for (k, i) in enumerate(idx), j in neighbours(i)
                        J[k, j] = (i == j) * gamma * m[k] - J[k, j]
                    end
                    return nothing
                end
                fn = jac ? DAEFunction(residual!; jac = dae_jac!, jac_prototype = proto) :
                    DAEFunction(residual!; jac_prototype = proto)
                return DAEProblem(fn, zeros(n), heat0(idx), SPAN; differential_vars = m .!= 0)
            end
            M = form == :diag ? Diagonal(m) : sparse(1:n, idx, m, n, size(proto, 2))
            M = something(mass, M)
            fn = jac ? ODEFunction(f!; jac = jac!, jac_prototype = proto, mass_matrix = M) :
                ODEFunction(f!; jac_prototype = proto, mass_matrix = M)
            return ODEProblem(fn, heat0(idx), SPAN)
        end
        method(form; kw...) = form == :dae ? TSDAE("bdf"; kw...) : TSImplicit("bdf"; kw...)
        rest = [(N - 3) ÷ (nranks - 1) + (r < (N - 3) % (nranks - 1)) for r in 0:(nranks - 2)]
        lean = nranks == 1 ? [N] : [3; rest]
        for layout in (counts, lean), form in (:diag, :sparse, :dae), (jac, ad) in SOURCES
            prob = chain(owned(layout), false, form; jac)
            @test caught(() -> solve(prob, method(form; comm); TOL...)) isa
                SciMLBase.CheckInitFailureError
            for init in (brown, shampine)
                sol = solve(prob, method(form; comm); initializealg = init, TOL...)
                @test sol.retcode == ReturnCode.Success
                start, u = gathered(sol.u[1], layout), gathered(sol.u[end], layout)
                if rank == 0
                    serial = chain(1:N, true, form; jac)
                    ref = solve(serial, method(form; autodiff = ad); initializealg = init, TOL...)
                    @test maximum(abs, start - heat0(1:N)) > 0.1
                    @test maximum(abs, start - ref.u[1]) <= INIT_GAP
                    @test sol.t[end] == ref.t[end]
                    @test maximum(abs, u - ref.u[end]) <= SERIAL_GAP
                end
            end
        end
        for form in (:diag, :dae), init in (brown, shampine)
            sol = solve(
                chain(rows, false, form; jac = false, hopeless = true), method(form; comm);
                initializealg = init, TOL...,
            )
            @test sol.retcode == ReturnCode.InitialFailure
            @test sol.t == [SPAN[1]]
            for jac in (true, false)
                prob = chain(rows, false, form; jac, throws = true)
                e = caught(() -> solve(prob, method(form; comm); initializealg = init, TOL...))
                @test raised(e, "f threw")
            end
        end
        bump = SciMLBase.DiscreteCallback(
            (u, t, integ) -> t == 0.05, integ -> (integ.u .+= 0.02; nothing),
        )
        function lifecycle(prob, alg, layout)
            g(u) = layout === nothing ? copy(u) : gathered(u, layout)
            integ = SciMLBase.init(prob, alg; initializealg = brown, TOL...)
            SciMLBase.set_u!(integ, integ.u .+ 0.05)
            SciMLBase.initialize_dae!(integ, shampine)
            out = [g(integ.u)]
            SciMLBase.set_u!(integ, integ.u .+ 0.05)
            SciMLBase.initialize_dae!(integ)
            push!(out, g(integ.u))
            SciMLBase.reinit!(integ, prob.u0 .+ 0.01)
            push!(out, g(integ.u))
            SciMLBase.terminate!(integ)
            kw = (; initializealg = brown, callback = bump, tstops = [0.05])
            sol = solve(prob, alg; kw..., TOL...)
            return out, g(sol.u[findlast(==(0.05), sol.t)])
        end
        for form in (:diag, :sparse, :dae), (jac, ad) in SOURCES
            got, after = lifecycle(chain(rows, false, form; jac), method(form; comm), counts)
            if rank == 0
                ref, ref_after = lifecycle(
                    chain(1:N, true, form; jac), method(form; autodiff = ad), nothing,
                )
                @test maxdiff(got, ref) <= 1.0e-11
                @test maximum(abs, after - ref_after) <= SERIAL_GAP
            end
        end
        n = length(rows)
        moved = sparse(
            1:n, [i > 1 && algebraic(i - 1) ? i - 1 : i for i in rows],
            [algebraic(i) ? 0.0 : 1.0 for i in rows], n, N,
        )
        odd = chain(rows, false, :diag; jac = true, mass = moved)
        @test refused(
            () -> solve(odd, TSImplicit("bdf"; comm); initializealg = brown, TOL...),
            "its zero columns have to be its zero rows",
        )
        dae = chain(rows, false, :dae; jac = true)
        unmarked = DAEProblem(dae.f, dae.du0, dae.u0, SPAN)
        @test refused(
            () -> solve(unmarked, TSDAE("bdf"; comm); initializealg = brown, TOL...),
            "differential_vars",
        )
    end

    @testset "SplitODEProblem with an explicit f2" begin
        decay!(du, u, p, t) = (du .= -u; nothing)
        split(idx, f; jac) = SplitODEProblem(heat_function(f, idx; jac), decay!, heat0(idx), SPAN)
        for (jac, ad) in SOURCES
            compare(
                split(rows, heat!; jac), TSARKIMEX(; comm), split(1:N, heat_serial!; jac),
                TSARKIMEX(; autodiff = ad), exp(-SPAN[2]) .* heat_exact(SPAN[2]), 4.0e-8; TOL...,
            )
        end
    end

    @testset "TSARKIMEX takes its first stage at the step's start" begin
        a = 2 .+ (1:N) ./ N
        exact(t) = 1 ./ (a[rows] .- sin(t))
        rhs!(du, u, p, t) = (du .= u .^ 2 .* cos(t); nothing)
        function jac!(J, u, p, t)
            for (k, i) in enumerate(rows)
                J[k, i] = 2 * u[k] * cos(t)
            end
            return nothing
        end
        tight = ["-snes_rtol", "1e-12", "-snes_atol", "1e-14", "-ksp_rtol", "1e-12"]
        for jac in (nothing, jac!), span in ((1.0, 2.0), (2.0, 1.0))
            fn = ODEFunction(rhs!; jac, jac_prototype = heat_proto(rows))
            sol = solve(
                ODEProblem(fn, exact(span[1]), span), TSARKIMEX("4", tight; comm);
                dt = 0.05, adaptive = false,
            )
            err = MPI.Allreduce(maximum(abs, sol.u[end] - exact(span[2])), max, comm)
            @test err < 1.0e-7
        end
    end

    @testset "f, jac or f2 throwing on one rank raises on every rank" begin
        proto = heat_proto(rows)
        bdf = TSImplicit("bdf"; comm)
        fixed = (; dt = 1.0e-3, adaptive = false)
        f = throwing(heat!, "f")
        jac = throwing(heat_jac(rows), "jac")
        for (what, fn, alg, kw) in (
                ("f", heat_function(f, rows; jac = true), bdf, (;)),
                ("f", heat_function(f, rows; jac = false), bdf, (;)),
                ("f", heat_function(f, rows; jac = true), TSRosW(; comm), fixed),
                ("jac", ODEFunction(heat!; jac, jac_prototype = proto), bdf, (;)),
            )
            @test raised(caught(() -> solve(ODEProblem(fn, heat0(rows), SPAN), alg; kw...)), what)
        end
        residual = throwing((r, du, u, p, t) -> (heat!(r, u, p, t); r .= du .- r), "residual")
        dae = DAEProblem(
            DAEFunction(residual; jac_prototype = proto), zero(heat0(rows)), heat0(rows), SPAN,
        )
        noinit = (; initializealg = SciMLBase.NoInit())
        @test raised(caught(() -> solve(dae, TSDAE("bdf"; comm); noinit...)), "residual")
        f1 = heat_function(heat!, rows; jac = true)
        f2(when) = function (du, u, p, t)
            du .= -u
            rank == thrower && when(t) && error("f2 threw")
            return nothing
        end
        for (when, kw) in (
                (t -> t == 0, (;)),
                (t -> t > 0.02, (;)),
                (t -> t == 0.0105, (; fixed..., saveat = [0.0105], dense = true)),
            )
            prob = SplitODEProblem(f1, f2(when), heat0(rows), SPAN)
            @test raised(caught(() -> solve(prob, TSARKIMEX(; comm); kw...)), "f2 threw")
        end
    end

    @testset "a failed Newton solve is taken again smaller on every rank" begin
        breaks(f, i) = (du, u, p, t) -> (f(du, u, p, t); t > 0.05 && i > 0 && (du[i] = NaN); nothing)
        for (make, _) in METHODS
            fn = heat_function(breaks(heat!, rank == thrower ? 1 : 0), rows; jac = true)
            sol = @test_logs (:warn, r"nonlinear solve failed") solve(
                ODEProblem(fn, heat0(rows), SPAN), make(comm); TOL...,
            )
            @test sol.retcode == ReturnCode.Unstable
            @test same_everywhere(sol.t)
            @test 0.05 - sol.t[end] < 1.0e-12
            failed = make(comm) isa TSRosW ? sol.stats.nreject : sol.stats.nnonlinconvfail
            @test failed > 10
            budget = sol.stats.naccept + 10
            capped = @test_logs solve(
                ODEProblem(fn, heat0(rows), SPAN), make(comm); TOL..., maxiters = budget,
            )
            @test capped.retcode == ReturnCode.MaxIters
            stats = capped.stats
            @test same_everywhere((capped.t, stats.naccept, stats.nreject, stats.nnonlinconvfail))
            @test stats.naccept + stats.nreject + stats.nnonlinconvfail == budget
        end
        irk_counts = even(N)
        idx = owned(irk_counts)
        irk_first = sum(irk_counts[1:thrower]) + 1
        fn = heat_function(breaks(heat!, rank == thrower ? 1 : 0), idx; jac = true)
        sol = solve(ODEProblem(fn, heat0(idx), SPAN), TSIRK(2; comm); dt = 1.0e-3)
        @test sol.retcode == ReturnCode.ConvergenceFailure
        us = gathered(sol, irk_counts)
        if rank == 0
            serial = heat_function(breaks(heat_serial!, irk_first), 1:N; jac = true)
            ref = solve(ODEProblem(serial, heat0(1:N), SPAN), TSIRK(2); dt = 1.0e-3)
            @test ref.retcode == ReturnCode.ConvergenceFailure
            @test sol.t == ref.t
            @test maxdiff(us, ref.u) <= SERIAL_GAP
        end
    end

    @testset "the single-precision underflow warning is decided on the whole state" begin
        decay!(du, u, p, t) = (du .= -u; nothing)
        decay_jac!(J, u, p, t) = (foreach(k -> J[k, rows[k]] = -1, eachindex(rows)); nothing)
        proto = sparse(1:length(rows), rows, ones(Float32, length(rows)), length(rows), N)
        fn = ODEFunction(decay!; jac = decay_jac!, jac_prototype = proto)
        warned(f) = any(l -> occursin("underflow", string(l.message)), Test.collect_test_logs(f)[1])
        for (scale, expected) in ((1.0f-25, true), (rank == nranks - 1 ? 1.0f0 : 1.0f-25, false))
            prob = ODEProblem(fn, fill(scale, length(rows)), (0.0f0, 1.0f0))
            w = warned(() -> solve(prob, TSImplicit("bdf"; comm); dt = 0.01f0))
            @test w == expected
            @test same_everywhere(w)
        end
    end

    @testset "AutoForwardDiff colours the whole pattern for every rank" begin
        ad = AutoForwardDiff()
        heat(idx, f; jac = false) = ODEProblem(heat_function(f, idx; jac), heat0(idx), SPAN)
        calls = Ref(0)
        counting(f) = (du, u, p, t) -> (calls[] += 1; f(du, u, p, t))
        for (make, err) in METHODS
            calls[] = 0
            sol = compare(
                heat(rows, counting(heat!)), make(comm; autodiff = ad), heat(1:N, heat_serial!),
                make(MPI.COMM_SELF; autodiff = ad), heat_exact(SPAN[2]), err; TOL...,
            )
            @test sol.stats.njacs > 0
            @test same_everywhere(calls[])
            @test same_everywhere(sol.stats.nf)
            given = solve(heat(rows, heat!; jac = true), make(comm); TOL...)
            @test sol.t == given.t
            @test sol.u == given.u
        end
        residual(f) = (r, du, u, p, t) -> (f(r, u, p, t); r .= du .- r; nothing)
        function dae(idx, f)
            du0 = similar(heat0(idx))
            f(du0, heat0(idx), nothing, 0.0)
            fn = DAEFunction(residual(f); jac_prototype = heat_proto(idx))
            return DAEProblem(fn, du0, heat0(idx), SPAN)
        end
        compare(
            dae(rows, heat!), TSDAE("bdf"; comm, autodiff = ad), dae(1:N, heat_serial!),
            TSDAE("bdf"; autodiff = ad), heat_exact(SPAN[2]), 1.5e-6; TOL...,
        )
        fade!(du, u, p, t) = (du .= -u; nothing)
        halves(idx, f) = SplitODEProblem(heat_function(f, idx; jac = false), fade!, heat0(idx), SPAN)
        compare(
            halves(rows, heat!), TSARKIMEX(; comm, autodiff = ad), halves(1:N, heat_serial!),
            TSARKIMEX(; autodiff = ad), exp(-SPAN[2]) .* heat_exact(SPAN[2]), 4.0e-8; TOL...,
        )
        irk_counts = even(N)
        idx = owned(irk_counts)
        sol = solve(heat(idx, heat!), TSIRK(2; comm, autodiff = ad); dt = 1.0e-3)
        @test sol.retcode == ReturnCode.Success
        us = gathered(sol, irk_counts)
        if rank == 0
            ref = solve(heat(1:N, heat_serial!), TSIRK(2; autodiff = ad); dt = 1.0e-3)
            @test sol.t == ref.t
            @test maxdiff(us, ref.u) <= SERIAL_GAP
        end
        prob = heat(rows, throwing(heat!, "f"))
        @test raised(caught(() -> solve(prob, TSImplicit("bdf"; comm, autodiff = ad))), "f threw")
        if nranks > 1
            layout = copy(counts)
            layout[2] += layout[1]
            layout[1] = 0
            function skipping!(du, u, p, t)
                isempty(u) && return nothing
                left = rank == 1 ? MPI.PROC_NULL : rank - 1
                right = rank == nranks - 1 ? MPI.PROC_NULL : rank + 1
                gl, gr = zeros(eltype(u), 1), zeros(eltype(u), 1)
                MPI.Sendrecv!(u[1:1], gr, comm; dest = left, source = right)
                MPI.Sendrecv!(u[end:end], gl, comm; dest = right, source = left)
                return laplacian!(du, u, gl[1], gr[1])
            end
            calls[] = 0
            sol = compare(
                heat(owned(layout), counting(skipping!)), TSImplicit("bdf"; comm, autodiff = ad),
                heat(1:N, heat_serial!), TSImplicit("bdf"; autodiff = ad), heat_exact(SPAN[2]),
                1.5e-6; layout, TOL...,
            )
            @test isempty(sol.u[end]) == (rank == 0)
            @test same_everywhere(calls[])
        end
    end

    @testset "refusals" begin
        n = length(rows)
        heat(fn) = ODEProblem(fn, heat0(rows), SPAN)
        bdf = TSImplicit("bdf"; comm)
        coloured = heat(heat_function(heat!, rows; jac = false))
        @test refused(
            () -> solve(heat(ODEFunction(heat!; jac = heat_jac(rows))), bdf), "sparse `jac_prototype`",
        )
        @test refused(() -> solve(heat(ODEFunction(heat!)), bdf), "sparse `jac_prototype`")
        zygote = PETScDiffEq.ADTypes.AutoZygote()
        @test refused(
            () -> solve(coloured, TSImplicit("bdf"; comm, autodiff = zygote)),
            "AutoZygote()` on a communicator",
        )
        @test refused(
            () -> solve(heat(ODEFunction(heat!)), TSImplicit("bdf"; comm, autodiff = AutoForwardDiff())),
            "sparse `jac_prototype`",
        )
        for alg in (TSIRK(2; comm), TSGeneric("irk"; comm))
            @test refused(() -> solve(coloured, alg; dt = 1.0e-3), "TSIRK needs a `jac`")
        end
        dense_mass = [i == j ? 2.0 : 0.0 for i in 1:n, j in 1:n]
        @test refused(
            () -> solve(heat(heat_function(heat!, rows; jac = false, mass_matrix = dense_mass)), bdf),
            "only a `Diagonal` mass matrix",
        )
        @test refused(() -> solve(coloured, TSGeneric("glle"; comm); dt = 0.1), "cannot run")
        @test refused(
            () -> solve(coloured, TSImplicit("bdf", ["-ts_type", "glle"]; comm)), "`glle` cannot run",
        )
        proto = heat_proto(rows)
        wrong = rank == thrower ? [proto; spzeros(1, N)] : proto
        fn = ODEFunction(heat!; jac = heat_jac(rows), jac_prototype = wrong)
        e = caught(() -> solve(heat(fn), bdf))
        @test rank == thrower ? e isa ArgumentError && occursin("must be $n x $N", e.msg) : remote(e)
    end

    @testset "a Krylov dot product does not depend on where the vectors sit in memory: $(nameof(typeof(alg)))" for alg in (
            TSImplicit("bdf"; comm), TSIRK(2; comm),
        )
        pl = PETSc.getlib(; PetscScalar = Float64)
        lib = PETScDiffEq.LibPETSc
        integ = SciMLBase.init(
            ODEProblem(heat_function(heat!, rows; jac = true), heat0(rows), SPAN), alg;
            TOL..., (alg isa TSIRK ? (; dt = 1.0e-3, adaptive = false) : (;))...,
        )
        x = lib.VecDuplicate(pl, integ.h.u)
        len, m = Int(lib.VecGetLocalSize(pl, x)), 4
        wrap(buf, off) = lib.VecCreateMPIWithArray(
            pl, comm, lib.PetscInt(1), lib.PetscInt(len), lib.PetscInt(lib.PETSC_DECIDE),
            unsafe_wrap(Array, pointer(buf, off + 1), len),
        )
        function mdot(ys)
            z = zeros(m)
            PETScDiffEq._check_code(
                ccall(
                    PETScDiffEq._symbol(pl, :VecMDot), PETScDiffEq.LibPETSc.PetscErrorCode,
                    (Ptr{Cvoid}, PETScDiffEq.LibPETSc.PetscInt, Ptr{Ptr{Cvoid}}, Ptr{Float64}),
                    x.ptr, m, [y.ptr for y in ys], z,
                ),
            )
            return z
        end
        # These gaps fail PETSc's stride test, so only the packed copy could go through GEMV.
        offsets = [3len + 200, 0, len + 70, 2len + 140]
        agree = map(1:20) do trial
            PETScDiffEq.PETScCompat.with_local_array!(x; read = false, write = true) do a
                a .= cos.(7trial .+ 3 .* (1:len) .+ 11rank)
            end
            Y = [sin(17trial + 3i + 5k + 11rank) for i in 1:len, k in 1:m]
            packed, scattered = vec(Y), zeros(4len + 300)
            for k in 1:m
                scattered[offsets[k] .+ (1:len)] .= Y[:, k]
            end
            GC.@preserve packed scattered begin
                a = [wrap(packed, (k - 1) * len) for k in 1:m]
                b = [wrap(scattered, offsets[k]) for k in 1:m]
                same = mdot(a) == mdot(b)
                foreach(PETScDiffEq.PETScCompat.destroy!, [a; b])
                same
            end
        end
        @test all(agree)
        PETScDiffEq.PETScCompat.destroy!(x)
        SciMLBase.terminate!(integ)
    end

    @testset "every handle is freed" begin
        @test isempty(PETScDiffEq.PARALLEL_HANDLES)
        @test all(h -> h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
    end
end
