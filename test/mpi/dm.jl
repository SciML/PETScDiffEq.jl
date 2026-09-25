using MPI, PETScDiffEq, SciMLBase, SparseArrays, Test
using LinearAlgebra: Diagonal
using PETScDiffEq: PETSc, LibPETSc, PETScCompat, AutoForwardDiff, AutoFiniteDiff,
    reshape_local_array
using SciMLBase: ODEProblem, ODEFunction, DAEProblem, DAEFunction, SplitODEProblem,
    DiscreteCallback, ContinuousCallback, ReturnCode, init, solve, solve!, step!, get_du,
    terminate!

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
# PetscInitialize is collective, and the serial reference solves run on single ranks.
for S in (Float64, Float32)
    PETSc.initialize(PETSc.getlib(; PetscScalar = S))
end
const pl = PETSc.getlib(; PetscScalar = Float64)
const GHOSTED = LibPETSc.DM_BOUNDARY_GHOSTED
const thrower = nranks - 1
const TOL = (abstol = 1.0e-8, reltol = 1.0e-8)
const FIXED = (; dt = 1.0e-3, adaptive = false)
const ROUNDOFF = 5.0e-14

function uneven(n)
    counts = floor.(Int, n .* (1:nranks) ./ sum(1:nranks))
    counts[end] += n - sum(counts)
    return counts
end
owned(counts) = (lo = sum(counts[1:rank]) + 1; lo:(lo + counts[rank + 1] - 1))

same_everywhere(x) = MPI.bcast(x, 0, comm) == x
anywhere(b) = MPI.Allreduce(b, |, comm)
everywhere(b) = MPI.Allreduce(b, &, comm)
maxdiff(a, b) = maximum(maximum(abs, x - y) for (x, y) in zip(a, b))

function refs(obj)
    n = Ref{LibPETSc.PetscInt}(0)
    PETScDiffEq._check_code(
        ccall(
            PETScDiffEq._symbol(pl, :PetscObjectGetReference), LibPETSc.PetscErrorCode,
            (Ptr{Cvoid}, Ptr{LibPETSc.PetscInt}), obj, n,
        ),
    )
    return Int(n[])
end
held(dm) = PETScDiffEq._referenced(pl, dm)
function released(dm)
    n = refs(dm)
    PETScDiffEq._check_code(
        ccall(
            PETScDiffEq._symbol(pl, :DMDestroy), LibPETSc.PetscErrorCode, (Ptr{Ptr{Cvoid}},),
            Ref(dm),
        ),
    )
    return n == 1
end

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
refused_on_thrower(e, what) =
    rank == thrower ? e isa ArgumentError && occursin(what, e.msg) : remote(e)

function natural(u, dm, dims)
    full = zeros(dims)
    a = reshape_local_array(u, dm)
    for I in CartesianIndices(axes(a)[2:end])
        full[I] = a[1, I]
    end
    return MPI.Reduce(vec(full), +, comm; root = 0)
end
natural(sol::SciMLBase.AbstractODESolution, dm, dims) = [natural(u, dm, dims) for u in sol.u]

const N = 23
const dx = 1 / (N + 1)
const counts = uneven(N)
const rows = owned(counts)
const SPAN = (0.0, 0.1)
heat0(idx) = sinpi.(idx .* dx) .+ 0.5 .* sinpi.(3 .* idx .* dx)
eigval(k) = -4 / dx^2 * sinpi(k * dx / 2)^2
heat_exact(t) = exp(eigval(1) * t) .* sinpi.((1:N) .* dx) .+
    0.5 * exp(eigval(3) * t) .* sinpi.(3 .* (1:N) .* dx)

line_da(c; kw...) = PETSc.DMDA(pl, c, (GHOSTED,), (N,), 1, 1; kw...)
const da = line_da(comm; points_per_proc = (LibPETSc.PetscInt.(counts),))

function heat_dm!(du, u, da, t)
    U, D = reshape_local_array(u, da), reshape_local_array(du, da)
    for i in axes(D, 2)
        D[1, i] = (U[1, i - 1] - 2U[1, i] + U[1, i + 1]) / dx^2
    end
    return nothing
end

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

dm_heat(f = heat_dm!; kw...) = ODEProblem(ODEFunction(f; kw...), heat0(rows), SPAN, da)
comm_heat(f = heat!; kw...) =
    ODEProblem(ODEFunction(f; jac_prototype = heat_proto(rows), kw...), heat0(rows), SPAN)
serial_heat(f = heat_serial!; kw...) =
    ODEProblem(ODEFunction(f; jac_prototype = heat_proto(1:N), kw...), heat0(1:N), SPAN)

explicit(; kw...) = TSRK("5dp"; kw...)
implicit(; kw...) = TSImplicit("bdf"; autodiff = AutoFiniteDiff(), kw...)

const NX, NY = 7, 6
const hx, hy = 1 / (NX + 1), 1 / (NY + 1)
grid0(i, j) = sinpi(i * hx) * sinpi(j * hy) + 0.5 * sinpi(3i * hx) * sinpi(2j * hy)
eig2(k, l) = -4 / hx^2 * sinpi(k * hx / 2)^2 - 4 / hy^2 * sinpi(l * hy / 2)^2
grid_exact(t) = vec(
    [
        exp(eig2(1, 1) * t) * sinpi(i * hx) * sinpi(j * hy) +
            0.5 * exp(eig2(3, 2) * t) * sinpi(3i * hx) * sinpi(2j * hy)
            for i in 1:NX, j in 1:NY
    ],
)

grid_da(processors, ppp) = PETSc.DMDA(
    pl, comm, (GHOSTED, GHOSTED), (NX, NY), 1, 1, LibPETSc.DMDA_STENCIL_STAR;
    processors, points_per_proc = ppp,
)

function grid_heat_dm!(du, u, da, t)
    U, D = reshape_local_array(u, da), reshape_local_array(du, da)
    for j in axes(D, 3), i in axes(D, 2)
        D[1, i, j] = (U[1, i - 1, j] - 2U[1, i, j] + U[1, i + 1, j]) / hx^2 +
            (U[1, i, j - 1] - 2U[1, i, j] + U[1, i, j + 1]) / hy^2
    end
    return nothing
end

function grid_u0(da)
    u = zeros(PETScDiffEq._dm_local_size(pl, da))
    a = reshape_local_array(u, da)
    for j in axes(a, 3), i in axes(a, 2)
        a[1, i, j] = grid0(i, j)
    end
    return u
end

function grid_laplacian!(du, u, below, above)
    ny = length(u) ÷ NX
    U, D = reshape(u, NX, ny), reshape(du, NX, ny)
    at(i, j) = !(1 <= i <= NX) ? 0.0 : j == 0 ? below[i] : j == ny + 1 ? above[i] : U[i, j]
    for j in 1:ny, i in 1:NX
        D[i, j] = (at(i - 1, j) - 2U[i, j] + at(i + 1, j)) / hx^2 +
            (at(i, j - 1) - 2U[i, j] + at(i, j + 1)) / hy^2
    end
    return nothing
end

function grid_halo(u)
    down = rank == 0 ? MPI.PROC_NULL : rank - 1
    up = rank == nranks - 1 ? MPI.PROC_NULL : rank + 1
    below, above = zeros(NX), zeros(NX)
    MPI.Sendrecv!(u[1:NX], above, comm; dest = down, source = up)
    MPI.Sendrecv!(u[(end - NX + 1):end], below, comm; dest = up, source = down)
    return below, above
end

grid_heat!(du, u, p, t) = grid_laplacian!(du, u, grid_halo(u)...)
grid_heat_serial!(du, u, p, t) = grid_laplacian!(du, u, zeros(NX), zeros(NX))

function grid_proto(ys)
    point(i, j) = i + (j - 1) * NX
    near(i, j) = [
        point(a, b) for (a, b) in ((i, j - 1), (i - 1, j), (i, j), (i + 1, j), (i, j + 1))
            if 1 <= a <= NX && 1 <= b <= NY
    ]
    local_rows = [point(i, j) - point(1, first(ys)) + 1 for j in ys for i in 1:NX]
    cols = [near(i, j) for j in ys for i in 1:NX]
    I = [r for (r, c) in zip(local_rows, cols) for _ in c]
    return sparse(I, reduce(vcat, cols), ones(length(I)), NX * length(ys), NX * NY)
end

function throwing(f, when)
    return function (du, u, p, t)
        f(du, u, p, t)
        rank == thrower && when(t) && error("f threw on rank $rank")
        return nothing
    end
end

@testset "MPI DM, $nranks ranks" begin
    @testset "the algorithms take the DM's communicator" begin
        alg = TSRK(; dm = da)
        @test nranks == 1 ? alg.comm == MPI.COMM_SELF :
            MPI.Comm_compare(alg.comm, comm) == MPI.CONGRUENT
        @test TSRK(; dm = da, comm).comm == comm
        @test TSImplicit("bdf"; dm = da).autodiff isa AutoFiniteDiff
        @test TSRosW(; dm = da, comm).autodiff isa AutoFiniteDiff
        @test refused(() -> TSRK(; dm = 1), "takes a PETSc DM")
        nranks > 1 && @test refused(() -> TSRK(; dm = da, comm = MPI.COMM_SELF), "other ranks")
        integ = init(dm_heat(), TSImplicit("bdf"; dm = da); TOL...)
        ts_dm = held(PETScDiffEq._ts_dm(pl, integ.h.ts))
        @test ts_dm != da.ptr
        @test PETScDiffEq._dm_type(pl, LibPETSc.PetscDM(ts_dm, pl)) == "da"
        terminate!(integ)
        @test everywhere(released(ts_dm))
        @test everywhere(refs(da.ptr) == 1)
    end

    @testset "1-D heat, explicit and implicit" begin
        for (make, kw, err) in ((explicit, FIXED, 2.0e-12), (implicit, TOL, 1.5e-6))
            got = solve(dm_heat(), make(; dm = da, comm); TOL...)
            ref = solve(comm_heat(), make(; comm); TOL...)
            @test got.retcode == ReturnCode.Success
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
            @test same_everywhere(got.stats.nf)
            got = solve(dm_heat(), make(; dm = da); kw...)
            us = natural(got, da, N)
            if rank == 0
                serial = solve(serial_heat(), make(); kw...)
                @test got.t[end] == serial.t[end]
                @test maximum(abs, us[end] - serial.u[end]) <= ROUNDOFF
                @test maximum(abs, us[end] - heat_exact(SPAN[2])) <= err
            end
        end
        ssp(; kw...) = TSGeneric("ssp"; explicit = true, kw...)
        got = solve(dm_heat(), ssp(; dm = da, comm); FIXED...)
        ref = solve(comm_heat(), ssp(; comm); FIXED...)
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)
    end

    @testset "ghost points past the edge of the grid read zero on every call" begin
        function scribbling!(du, u, da, t)
            heat_dm!(du, u, da, t)
            u .= NaN
            return nothing
        end
        got = solve(dm_heat(scribbling!), explicit(; dm = da, comm); FIXED...)
        ref = solve(dm_heat(), explicit(; dm = da, comm); FIXED...)
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)
    end

    @testset "an implicit solve on a periodic grid of any size" begin
        ring = PETSc.DMDA(
            pl, comm, (LibPETSc.DM_BOUNDARY_PERIODIC,), (N,), 1, 1;
            points_per_proc = (LibPETSc.PetscInt.(counts),),
        )
        wave(idx) = cospi.(2 .* idx ./ N) .+ 0.5 .* sinpi.(6 .* idx ./ N)
        got = solve(ODEProblem(heat_dm!, wave(rows), SPAN, ring), implicit(; dm = ring); TOL...)
        @test got.retcode == ReturnCode.Success
        us = natural(got, ring, N)
        if rank == 0
            ring!(du, u, p, t) = laplacian!(du, u, u[end], u[1])
            I = repeat(1:N; inner = 3)
            J = [mod1(i + d, N) for i in 1:N for d in -1:1]
            fn = ODEFunction(ring!; jac_prototype = sparse(I, J, ones(3N), N, N))
            serial = solve(ODEProblem(fn, wave(1:N), SPAN), implicit(); TOL...)
            @test got.t[end] == serial.t[end]
            @test maximum(abs, us[end] - serial.u[end]) <= ROUNDOFF
        end
        PETScCompat.destroy!(ring)
    end

    @testset "2-D heat, explicit and implicit" begin
        slab = uneven(NY)
        across = grid_da((nranks, 1), (LibPETSc.PetscInt.(uneven(NX)), nothing))
        along = grid_da((1, nranks), (nothing, LibPETSc.PetscInt.(slab)))
        ys = owned(slab)
        span = (0.0, 0.05)
        serial = ODEProblem(
            ODEFunction(grid_heat_serial!; jac_prototype = grid_proto(1:NY)),
            vec([grid0(i, j) for i in 1:NX, j in 1:NY]), span,
        )
        for (make, kw, err) in ((explicit, FIXED, 2.0e-9), (implicit, TOL, 4.0e-6))
            got = solve(
                ODEProblem(grid_heat_dm!, grid_u0(across), span, across), make(; dm = across);
                saveat = 0.01, kw...,
            )
            @test got.retcode == ReturnCode.Success
            @test got.t == collect(0.0:0.01:0.05)
            us = natural(got, across, (NX, NY))
            if rank == 0
                ref = solve(serial, make(); saveat = 0.01, kw...)
                @test maxdiff(us, ref.u) <= ROUNDOFF
                @test maxdiff(us, grid_exact.(got.t)) <= err
            end
            got = solve(
                ODEProblem(grid_heat_dm!, grid_u0(along), span, along),
                make(; dm = along, comm); TOL...,
            )
            fn = ODEFunction(grid_heat!; jac_prototype = grid_proto(ys))
            u0 = vec([grid0(i, j) for i in 1:NX, j in ys])
            ref = solve(ODEProblem(fn, u0, span), make(; comm); TOL...)
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
        end
        PETScCompat.destroy!(across)
        PETScCompat.destroy!(along)
    end

    @testset "a DMDA on MPI.COMM_SELF, one to each rank" begin
        solo = line_da(MPI.COMM_SELF)
        for make in (explicit, implicit)
            alg = make(; dm = solo)
            @test alg.comm == MPI.COMM_SELF
            got = solve(ODEProblem(heat_dm!, heat0(1:N), SPAN, solo), alg; TOL...)
            ref = solve(serial_heat(), make(); TOL...)
            @test got.t == ref.t
            @test got.u == ref.u
        end
        back = (0.01, 0.0)
        got = solve(ODEProblem(heat_dm!, heat0(1:N), back, solo), explicit(; dm = solo); TOL...)
        ref = solve(ODEProblem(heat_serial!, heat0(1:N), back), explicit(); TOL...)
        @test got.retcode == ReturnCode.Success
        @test got.t == ref.t
        @test got.u == ref.u
        nan_after(f) = (du, u, p, t) -> (f(du, u, p, t); t > 0.05 && fill!(du, NaN); nothing)
        got = @test_logs (:warn, r"floating point exception") solve(
            ODEProblem(nan_after(heat_dm!), heat0(1:N), SPAN, solo), explicit(; dm = solo),
        )
        ref = @test_logs (:warn, r"floating point exception") solve(
            ODEProblem(nan_after(heat_serial!), heat0(1:N), SPAN), explicit(),
        )
        @test got.retcode == ReturnCode.Unstable
        @test got.stats.nreject > 10
        @test got.t == ref.t
        @test got.u == ref.u
        PETScCompat.destroy!(solo)
    end

    @testset "an adaptive step that turns NaN on every rank leaves the TS's DM free" begin
        nan_after(f) = (du, u, p, t) -> (f(du, u, p, t); t > 0.05 && fill!(du, NaN); nothing)
        for (prob, alg) in (
                (dm_heat(nan_after(heat_dm!)), explicit(; dm = da, comm)),
                (comm_heat(nan_after(heat!)), explicit(; comm)),
            )
            integ = init(prob, alg)
            ts_dm = held(PETScDiffEq._ts_dm(pl, integ.h.ts))
            sol = @test_logs (:warn, r"floating point exception") solve!(integ)
            @test sol.retcode == ReturnCode.Unstable
            @test everywhere(released(ts_dm))
        end
    end

    @testset "saveat and dense output" begin
        rk3(; kw...) = TSRK("3bs"; kw...)
        for (make, kw) in ((explicit, FIXED), (implicit, TOL), (rk3, FIXED))
            got = solve(dm_heat(), make(; dm = da, comm); saveat = 0.01, TOL...)
            ref = solve(comm_heat(), make(; comm); saveat = 0.01, TOL...)
            @test got.t == ref.t == collect(0.0:0.01:0.1)
            @test everywhere(got.u == ref.u)
            times = collect(0.0105:0.01:0.0905)
            got = solve(dm_heat(), make(; dm = da); saveat = times, kw...)
            @test got.t == times
            us = natural(got, da, N)
            if rank == 0
                serial = solve(serial_heat(), make(); saveat = times, kw...)
                @test maxdiff(us, serial.u) <= ROUNDOFF
            end
        end
        got = solve(dm_heat(), explicit(; dm = da, comm); FIXED...)
        ref = solve(comm_heat(), explicit(; comm); FIXED...)
        mids = (got.t[1:(end - 1)] .+ got.t[2:end]) ./ 2
        @test everywhere([got(t) for t in mids] == [ref(t) for t in mids])
    end

    @testset "callbacks" begin
        cases = (
            (;
                callback = DiscreteCallback((u, t, i) -> t == 0.05, i -> (i.u .*= 0.5)),
                tstops = [0.05],
            ),
            (;
                callback = ContinuousCallback(
                    (u, t, i) -> MPI.Allreduce(sum(u), +, comm) - 8.0, terminate!,
                ),
            ),
        )
        rk3(; kw...) = TSRK("3bs"; kw...)
        for kw in cases, make in (explicit, rk3, (; kw...) -> TSImplicit("bdf"; kw...))
            got = solve(dm_heat(), make(; dm = da, comm); kw..., FIXED...)
            ref = solve(comm_heat(), make(; comm); kw..., FIXED...)
            @test got.retcode == ref.retcode
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
        end
        @test solve(dm_heat(), explicit(; dm = da); cases[2]..., FIXED...).retcode ==
            ReturnCode.Terminated
        seen = Int[]
        sol = solve(
            dm_heat(), explicit(; dm = da);
            unstable_check = (dt, u, p, t) -> (push!(seen, length(u)); false),
            isoutofdomain = (u, p, t) -> (push!(seen, length(u)); false), TOL...,
        )
        @test sol.retcode == ReturnCode.Success
        @test !isempty(seen) && all(==(length(rows)), seen)
    end

    @testset "the integrator interface" begin
        function walk(prob, alg)
            integ = init(prob, alg; FIXED...)
            mids, dus = Vector{Float64}[], Vector{Float64}[]
            for _ in 1:30
                step!(integ)
                push!(mids, integ((integ.tprev + integ.t) / 2))
                push!(dus, get_du(integ))
            end
            terminate!(integ)
            return mids, dus
        end
        for make in (explicit, (; kw...) -> TSImplicit("bdf"; kw...))
            got = walk(dm_heat(), make(; dm = da, comm))
            ref = walk(comm_heat(), make(; comm))
            @test everywhere(got == ref)
        end
    end

    @testset "the caller can destroy the DM once the integrator has it" begin
        function ghosted!(du, u, p, t)
            for i in eachindex(du)
                du[i] = (u[i] - 2u[i + 1] + u[i + 2]) / dx^2
            end
            return nothing
        end
        prob = ODEProblem(ghosted!, heat0(rows), SPAN)
        fresh = line_da(comm; points_per_proc = (LibPETSc.PetscInt.(counts),))
        integ = init(prob, explicit(; dm = fresh); FIXED...)
        PETScCompat.destroy!(fresh)
        got = solve!(integ)
        ref = solve(prob, explicit(; dm = da); FIXED...)
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)
    end

    @testset "a mass matrix, a SplitODEProblem and a DAEProblem" begin
        d = 1 .+ (1:N) ./ N
        scaled(f) = (du, u, p, t) -> (f(du, u, p, t); du .*= d[rows]; nothing)
        for make in ((; kw...) -> TSImplicit("bdf"; kw...), (; kw...) -> TSRosW(; kw...))
            mass = Diagonal(d[rows])
            got = solve(dm_heat(scaled(heat_dm!); mass_matrix = mass), make(; dm = da, comm); TOL...)
            ref = solve(comm_heat(scaled(heat!); mass_matrix = mass), make(; comm); TOL...)
            @test got.retcode == ReturnCode.Success
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
        end

        function decay_dm!(du, u, da, t)
            U, D = reshape_local_array(u, da), reshape_local_array(du, da)
            for i in axes(D, 2)
                D[1, i] = -U[1, i]
            end
            return nothing
        end
        decay!(du, u, p, t) = (du .= -u; nothing)
        got = solve(
            SplitODEProblem(decay_dm!, heat_dm!, heat0(rows), SPAN, da), TSARKIMEX(; dm = da, comm);
            TOL...,
        )
        n = length(rows)
        f1 = ODEFunction(decay!; jac_prototype = sparse(1:n, rows, ones(n), n, N))
        ref = solve(SplitODEProblem(f1, heat!, heat0(rows), SPAN), TSARKIMEX(; comm); TOL...)
        @test got.retcode == ReturnCode.Success
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)
        u = natural(got.u[end], da, N)
        rank == 0 && @test maximum(abs, u - exp(-SPAN[2]) .* heat_exact(SPAN[2])) <= 5.0e-9

        residual(f) = (r, du, u, p, t) -> (f(r, u, p, t); r .= du .- r; nothing)
        du0 = zeros(length(rows))
        got = solve(
            DAEProblem(residual(heat_dm!), du0, heat0(rows), SPAN, da), TSDAE("bdf"; dm = da, comm);
            TOL...,
        )
        fn = DAEFunction(residual(heat!); jac_prototype = heat_proto(rows))
        ref = solve(DAEProblem(fn, du0, heat0(rows), SPAN), TSDAE("bdf"; comm); TOL...)
        @test got.retcode == ReturnCode.Success
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)
    end

    @testset "f throwing on one rank raises on every rank" begin
        after = t -> t > 0.02
        dense = (; FIXED..., saveat = [0.0105], dense = true)
        for (when, alg, kw) in (
                (t -> t == 0, explicit(; dm = da), (;)),
                (after, explicit(; dm = da), (;)),
                (after, implicit(; dm = da), (;)),
                (t -> t == 0.0105, explicit(; dm = da), dense),
            )
            e = caught(() -> solve(dm_heat(throwing(heat_dm!, when)), alg; kw...))
            @test raised(e, "f threw")
        end
        @test solve(dm_heat(throwing(heat_dm!, t -> false)), implicit(; dm = da)).retcode ==
            ReturnCode.Success
        @test everywhere(refs(da.ptr) == 1)
    end

    @testset "refusals" begin
        prob = dm_heat()
        for alg in (TSIRK(2; dm = da), TSMPRK([1]; dm = da), TSGeneric("alpha"; dm = da))
            @test refused(() -> solve(prob, alg; dt = 1.0e-3), "cannot run")
        end
        @test refused(() -> solve(prob, TSIRK(2; dm = da); dt = 1.0e-3), "TSIRK needs a `jac`")
        @test refused(
            () -> solve(dm_heat(; jac = (J, u, p, t) -> nothing), implicit(; dm = da)),
            "does not take a `jac`",
        )
        @test refused(
            () -> solve(dm_heat(; jac_prototype = heat_proto(rows)), implicit(; dm = da)),
            "leave out `jac_prototype`",
        )
        @test refused(
            () -> solve(prob, TSImplicit("bdf"; dm = da, autodiff = AutoForwardDiff())),
            "cannot use `AutoForwardDiff()`",
        )
        n = length(rows)
        dense_mass = [i == j ? 2.0 : 0.0 for i in 1:n, j in 1:n]
        @test refused(
            () -> solve(dm_heat(; mass_matrix = dense_mass), implicit(; dm = da)),
            "only a `Diagonal` mass matrix",
        )
        @test refused(
            () -> solve(prob, TSImplicit("bdf", ["-ts_type", "irk"]; dm = da)),
            "`irk` cannot run with a `dm`",
        )
        @test refused(
            () -> PETScDiffEq._discrete_adjoint(
                prob, TSRK("4"; dm = da), PETScAdjoint(); t = [0.1],
                dgdu_discrete = (out, u, p, t, i) -> (out .= u), dt = 0.01, adaptive = false,
            ),
            "PETScAdjoint",
        )
        shell = LibPETSc.DMShellCreate(pl, comm)
        @test refused(
            () -> solve(ODEProblem(heat_dm!, heat0(rows), SPAN, shell), explicit(; dm = shell)),
            "only a DMDA",
        )
        LibPETSc.DMDestroy(pl, shell)

        short = rank == thrower ? heat0(rows)[1:(end - 1)] : heat0(rows)
        e = caught(() -> solve(ODEProblem(heat_dm!, short, SPAN, da), explicit(; dm = da)))
        @test refused_on_thrower(e, "`u0` has $(length(short)) entries")
        e = caught(
            () -> solve(
                dm_heat(; mass_matrix = rank == thrower ? dense_mass : Diagonal(ones(n))),
                implicit(; dm = da),
            ),
        )
        @test refused_on_thrower(e, "only a `Diagonal` mass matrix")
        single = rank == thrower
        prob = ODEProblem(
            heat_dm!, single ? Float32.(heat0(rows)) : heat0(rows),
            single ? Float32.(SPAN) : SPAN, da,
        )
        e = caught(() -> solve(prob, explicit(; dm = da)))
        @test refused_on_thrower(e, "belongs to PETSc's")
        da32 = PETSc.DMDA(
            PETSc.getlib(; PetscScalar = Float32), comm, (GHOSTED,), (N,), 1, 1;
            points_per_proc = (LibPETSc.PetscInt.(counts),),
        )
        @test refused(
            () -> solve(
                ODEProblem(heat_dm!, Float32.(heat0(rows)), SPAN, da32), explicit(; dm = da32),
            ),
            "Float32 real build, but `u0` and `tspan` together run this problem in its " *
                "Float64 real one",
        )
        PETScCompat.destroy!(da32)
    end

    @testset "every handle is freed" begin
        @test isempty(PETScDiffEq.PARALLEL_HANDLES)
        @test all(h -> h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
    end
end

PETScCompat.destroy!(da)
