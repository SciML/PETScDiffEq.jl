using MPI, PETScDiffEq, SciMLBase, SparseArrays, Test
using LinearAlgebra: Diagonal
using PETScDiffEq: PETSc, AutoFiniteDiff, AutoForwardDiff
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
    gl, gr = zeros(1), zeros(1)
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
        function cell_problem(cells; jac)
            n, offset = 3length(cells), 3(first(cells) - 1)
            I = [3(c - 1) + i for c in 1:length(cells) for (i, _, _) in entries]
            J = [offset + 3(c - 1) + j for c in 1:length(cells) for (_, j, _) in entries]
            V = [v for _ in cells for (_, _, v) in entries]
            proto = sparse(I, J, ones(length(I)), n, 3sum(cell_counts))
            cell_jac!(Jm, u, p, t) = (foreach((i, j, v) -> Jm[i, j] = v, I, J, V); nothing)
            M = Diagonal(repeat([0.0, 0.0, 1.0], length(cells)))
            fn = jac ?
                ODEFunction(cell!; jac = cell_jac!, jac_prototype = proto, mass_matrix = M) :
                ODEFunction(cell!; jac_prototype = proto, mass_matrix = M)
            return ODEProblem(fn, repeat([2.0, 1.0, 1.0], length(cells)), (0.0, 1.0))
        end
        exact = repeat(exp(-1) .* [2.0, 1.0, 1.0], sum(cell_counts))
        for (jac, ad) in SOURCES
            compare(
                cell_problem(cells; jac), TSImplicit("bdf"; comm),
                cell_problem(1:sum(cell_counts); jac), TSImplicit("bdf"; autodiff = ad), exact,
                6.0e-6; layout = 3 .* cell_counts, TOL...,
            )
        end
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
        @test raised(caught(() -> solve(dae, TSDAE("bdf"; comm))), "residual")
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

    @testset "refusals" begin
        n = length(rows)
        heat(fn) = ODEProblem(fn, heat0(rows), SPAN)
        bdf = TSImplicit("bdf"; comm)
        coloured = heat(heat_function(heat!, rows; jac = false))
        @test refused(
            () -> solve(heat(ODEFunction(heat!; jac = heat_jac(rows))), bdf), "sparse `jac_prototype`",
        )
        @test refused(() -> solve(heat(ODEFunction(heat!)), bdf), "sparse `jac_prototype`")
        @test refused(
            () -> solve(coloured, TSImplicit("bdf"; comm, autodiff = AutoForwardDiff())),
            "cannot use `AutoForwardDiff()`",
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
        if nranks > 1
            skewed = even(N)
            skewed[end - 1] -= 1
            skewed[end] += 1
            idx = owned(skewed)
            prob = ODEProblem(heat_function(heat!, idx; jac = true), heat0(idx), SPAN)
            @test refused(() -> solve(prob, TSIRK(2; comm); dt = 1.0e-3), "PETSc's own share")
        end
    end

    @testset "every handle is freed" begin
        @test isempty(PETScDiffEq.PARALLEL_HANDLES)
        @test all(h -> h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
    end
end
