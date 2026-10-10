using MPI, PETScDiffEq, SciMLBase, SparseArrays, Test
using PETScDiffEq: PETSc, LibPETSc, PETScCompat, PETScAdjoint, AutoForwardDiff, reshape_local_array
using SciMLBase: ODEProblem, ODEFunction, SplitODEProblem, solve

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
# PetscInitialize is collective, and the serial reference solves run on single ranks.
const pl = PETSc.getlib(; PetscScalar = Float64)
PETSc.initialize(pl)

const N = 23
const P = [0.8, 1.5, 0.5]
const DT = 1.0e-3
const FORWARD = ((0.0, 0.1), collect(0.0:0.01:0.1), P)
const BACKWARD = ((0.1, 0.0), collect(0.1:-0.01:0.0), [-P[1], P[2], -P[3]])
const EXACT = ["-snes_rtol", "1e-13", "-snes_atol", "1e-15", "-ksp_type", "preonly"]
const SERIAL_GAP = 1.0e-15
# Measured at 1 to 3 ranks: a dm adjoint is within 1.3e-16 of the comm-mode one in 1-D and
# 1.1e-15 of a DMDA on MPI.COMM_SELF in 2-D.
const DM_GAP = 5.0e-15
# Measured at 1 and 2 ranks: a split problem is within 7.0e-16 of its serial adjoint.
const SPLIT_GAP = 5.0e-15
const FD_GAP = 2.0e-9
const thrower = nranks - 1

function uneven(n)
    counts = floor.(Int, n .* (1:nranks) ./ sum(1:nranks))
    counts[end] += n - sum(counts)
    return counts
end
owned(counts) = (lo = sum(counts[1:rank]) + 1; lo:(lo + counts[rank + 1] - 1))

const counts = uneven(N)
const rows = owned(counts)

function gathered(u::AbstractVector{<:Number}, counts)
    out = rank == 0 ? zeros(eltype(u), sum(counts)) : nothing
    MPI.Gatherv!(u, rank == 0 ? MPI.VBuffer(out, counts) : nothing, comm)
    return out
end

same_everywhere(x) = MPI.bcast(x, 0, comm) == x
relerr(a, b) = maximum(abs, a - b) / maximum(abs, b)

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
x(i) = i * dx
source(i) = x(i) * (1 - x(i))
heat0(idx) = sinpi.(x.(idx))

function halo(u)
    left = rank == 0 ? MPI.PROC_NULL : rank - 1
    right = rank == nranks - 1 ? MPI.PROC_NULL : rank + 1
    gl, gr = zeros(1), zeros(1)
    MPI.Sendrecv!(u[1:1], gr, comm; dest = left, source = right)
    MPI.Sendrecv!(u[end:end], gl, comm; dest = right, source = left)
    return gl[1], gr[1]
end
walls(u) = (0.0, 0.0)

laplacian(u, k, l, r) = ((k == 1 ? l : u[k - 1]) - 2u[k] + (k == length(u) ? r : u[k + 1])) / dx^2

heat(idx, ghosts) = function (du, u, p, t)
    l, r = ghosts(u)
    for (k, i) in enumerate(idx)
        du[k] = p[1] * laplacian(u, k, l, r) + p[2] * source(i) - p[3] * u[k]^3
    end
    return nothing
end

neighbours(i) = max(1, i - 1):min(N, i + 1)

function heat_proto(idx)
    I = [k for (k, i) in enumerate(idx) for _ in neighbours(i)]
    J = [j for i in idx for j in neighbours(i)]
    return sparse(I, J, ones(length(I)), length(idx), N)
end

heat_jac(idx) = function (J, u, p, t)
    for (k, i) in enumerate(idx), j in neighbours(i)
        J[k, j] = p[1] * (i == j ? -2 : 1) / dx^2 - (i == j) * 3 * p[3] * u[k]^2
    end
    return nothing
end

heat_paramjac(idx, ghosts) = function (pJ, u, p, t)
    l, r = ghosts(u)
    for (k, i) in enumerate(idx)
        pJ[k, 1] = laplacian(u, k, l, r)
        pJ[k, 2] = source(i)
        pJ[k, 3] = -u[k]^3
    end
    return nothing
end

function heat_problem(
        idx, ghosts; tspan = first(FORWARD), u0 = heat0(idx), p = copy(P),
        f = heat(idx, ghosts), jac = heat_jac(idx), paramjac = heat_paramjac(idx, ghosts),
        jac_prototype = heat_proto(idx),
    )
    return ODEProblem(ODEFunction(f; jac, paramjac, jac_prototype), u0, tspan, p)
end

# The heat problem as its stiff diffusion and the rest, each with its own Jacobians.
diffusion(idx, ghosts) = function (du, u, p, t)
    l, r = ghosts(u)
    foreach(k -> du[k] = p[1] * laplacian(u, k, l, r), eachindex(idx))
    return nothing
end
reaction(idx) = (du, u, p, t) -> (du .= p[2] .* source.(idx) .- p[3] .* u .^ 3; nothing)

diffusion_jac(idx) = function (J, u, p, t)
    for (k, i) in enumerate(idx), j in neighbours(i)
        J[k, j] = p[1] * (i == j ? -2 : 1) / dx^2
    end
    return nothing
end
reaction_proto(idx) = sparse(1:length(idx), idx, 1.0, length(idx), N)
reaction_jac(idx) = function (J, u, p, t)
    foreach(k -> J[k, idx[k]] = -3 * p[3] * u[k]^2, eachindex(idx))
    return nothing
end

diffusion_paramjac(idx, ghosts) = function (pJ, u, p, t)
    l, r = ghosts(u)
    fill!(pJ, 0.0)
    foreach(k -> pJ[k, 1] = laplacian(u, k, l, r), eachindex(idx))
    return nothing
end
reaction_paramjac(idx) = function (pJ, u, p, t)
    fill!(pJ, 0.0)
    pJ[:, 2] .= source.(idx)
    pJ[:, 3] .= .-u .^ 3
    return nothing
end

function split_problem(
        idx, ghosts; tspan = first(FORWARD), u0 = heat0(idx), p = copy(P),
        jac = reaction_jac(idx), paramjac = reaction_paramjac(idx),
        jac_prototype = reaction_proto(idx),
    )
    f1 = ODEFunction(
        diffusion(idx, ghosts); jac = diffusion_jac(idx),
        paramjac = diffusion_paramjac(idx, ghosts), jac_prototype = heat_proto(idx),
    )
    return SplitODEProblem(
        f1, ODEFunction(reaction(idx); jac, paramjac, jac_prototype), u0, tspan, p,
    )
end

cost(u, p, idx) = sum(abs2, u) / 2 + p[2] * sum(u .* x.(idx))
cost_du(idx) = (out, u, p, t, i) -> (out .= u .+ p[2] .* x.(idx); nothing)
cost_dp(idx) = (out, u, p, t, i) -> (fill!(out, 0.0); out[2] = sum(u .* x.(idx)); nothing)

function gradient(prob, alg, times, idx; kw...)
    return PETScDiffEq._discrete_adjoint(
        prob, alg, PETScAdjoint(); t = times, dgdu_discrete = cost_du(idx),
        dgdp_discrete = cost_dp(idx), dt = DT, adaptive = false, kw...,
    )
end

function loss(alg, tspan, times, u0, p, problem)
    sol = solve(problem(rows, halo; tspan, u0, p), alg; dt = DT, adaptive = false, saveat = times)
    return MPI.Allreduce(sum(cost(u, p, rows) for u in sol.u), +, comm)
end

function differenced(alg, tspan, times, p0; h = 1.0e-6, problem = heat_problem)
    g = zeros(N + length(p0))
    for j in eachindex(g)
        function at(s)
            u0, p = heat0(rows), copy(p0)
            if j > N
                p[j - N] += s
            elseif j in rows
                u0[j - first(rows) + 1] += s
            end
            return loss(alg, tspan, times, u0, p, problem)
        end
        g[j] = (at(h) - at(-h)) / 2h
    end
    return g
end

function throwing(f, what, when)
    return function (args...)
        f(args...)
        rank == thrower && when(args...) && error("$what threw on rank $rank")
        return nothing
    end
end

exact(c) = vcat(EXACT, ["-pc_type", c == MPI.COMM_SELF ? "lu" : "redundant"])

const METHODS = (
    ("TSRK", c -> TSRK("4"; comm = c)),
    ("backward Euler", c -> TSImplicit("beuler", exact(c); comm = c)),
    ("Crank-Nicolson", c -> TSImplicit("cn", exact(c); comm = c)),
    ("theta 0.7", c -> TSImplicit("theta", 0.7, exact(c); comm = c)),
    ("ARKIMEX l2", c -> TSARKIMEX("l2", exact(c); comm = c)),
    ("ARKIMEX 3", c -> TSARKIMEX("3", exact(c); comm = c)),
)

const GHOSTED = LibPETSc.DM_BOUNDARY_GHOSTED
const da = PETSc.DMDA(pl, comm, (GHOSTED,), (N,), 1, 1; points_per_proc = (LibPETSc.PetscInt.(counts),))

function dm_heat!(du, u, p, t)
    U, D = reshape_local_array(u, da), reshape_local_array(du, da)
    for i in axes(D, 2)
        D[1, i] = p[1] * ((U[1, i - 1] - 2U[1, i] + U[1, i + 1]) / dx^2) + p[2] * source(i) -
            p[3] * U[1, i]^3
    end
    return nothing
end

function dm_jac!(J, u, p, t)
    U = reshape_local_array(u, da)
    for i in rows
        row = (p[1] / dx^2, -2p[1] / dx^2 - 3 * p[3] * U[1, i]^2, p[1] / dx^2)
        set_stencil_values!(J, (1, i), ((1, i - 1), (1, i), (1, i + 1)), row)
    end
    return nothing
end

function dm_paramjac!(pJ, u, p, t)
    U = reshape_local_array(u, da)
    for (k, i) in enumerate(rows)
        pJ[k, 1] = (U[1, i - 1] - 2U[1, i] + U[1, i + 1]) / dx^2
        pJ[k, 2] = source(i)
        pJ[k, 3] = -U[1, i]^3
    end
    return nothing
end

dm_problem(;
    tspan = first(FORWARD), p = copy(P), f = dm_heat!, jac = dm_jac!, paramjac = dm_paramjac!,
) = ODEProblem(ODEFunction(f; jac, paramjac), heat0(rows), tspan, p)

with_dm(name) = Dict(
    "TSRK" => TSRK("4"; dm = da),
    "backward Euler" => TSImplicit("beuler", exact(comm); dm = da),
    "Crank-Nicolson" => TSImplicit("cn", exact(comm); dm = da),
    "theta 0.7" => TSImplicit("theta", 0.7, exact(comm); dm = da),
    "ARKIMEX l2" => TSARKIMEX("l2", exact(comm); dm = da),
    "ARKIMEX 3" => TSARKIMEX("3", exact(comm); dm = da),
)[name]

const NX, NY = 5, 4
const hx, hy = 1 / (NX + 1), 1 / (NY + 1)
const P2 = [0.8, 1.5, 0.5, 0.3]
plane(c, processors = (1, 1)) = PETSc.DMDA(
    pl, c, (GHOSTED, GHOSTED), (NX, NY), 2, 1, LibPETSc.DMDA_STENCIL_STAR; processors,
)
points(g) = CartesianIndices(
    axes(reshape_local_array(zeros(PETScDiffEq._dm_local_size(pl, g)), g))[2:3],
)
lap2(U, c, i, j) = (U[c, i - 1, j] - 2U[c, i, j] + U[c, i + 1, j]) / hx^2 +
    (U[c, i, j - 1] - 2U[c, i, j] + U[c, i, j + 1]) / hy^2

two_fields(g) = function (du, u, p, t)
    U, D = reshape_local_array(u, g), reshape_local_array(du, g)
    for I in points(g)
        i, j = Tuple(I)
        D[1, I] = p[1] * lap2(U, 1, i, j) + p[2] * i * j * hx * hy - p[3] * U[1, I]^3 + U[2, I]
        D[2, I] = p[4] * lap2(U, 2, i, j) + U[1, I] - p[3] * U[2, I]
    end
    return nothing
end

two_fields_jac(g) = function (J, u, p, t)
    U = reshape_local_array(u, g)
    for I in points(g), c in 1:2
        i, j = Tuple(I)
        d = c == 1 ? p[1] : p[4]
        cols = ((c, i, j), (c, i - 1, j), (c, i + 1, j), (c, i, j - 1), (c, i, j + 1), (3 - c, i, j))
        diagonal = -2d / hx^2 - 2d / hy^2 - (c == 1 ? 3 * p[3] * U[1, I]^2 : p[3])
        set_stencil_values!(J, (c, i, j), cols, (diagonal, d / hx^2, d / hx^2, d / hy^2, d / hy^2, 1.0))
    end
    return nothing
end

two_fields_paramjac(g) = function (pJ, u, p, t)
    U = reshape_local_array(u, g)
    fill!(pJ, 0.0)
    A = [reshape_local_array(view(pJ, :, k), g) for k in 1:4]
    for I in points(g)
        i, j = Tuple(I)
        A[1][1, I] = lap2(U, 1, i, j)
        A[2][1, I] = i * j * hx * hy
        A[3][1, I] = -U[1, I]^3
        A[3][2, I] = -U[2, I]
        A[4][2, I] = lap2(U, 2, i, j)
    end
    return nothing
end

function on_plane(value, g)
    u = zeros(PETScDiffEq._dm_local_size(pl, g))
    a = reshape_local_array(u, g)
    for I in points(g), c in 1:2
        a[c, I] = value(c, Tuple(I)...)
    end
    return u
end

function plane_gradient(g, alg)
    w = on_plane((c, i, j) -> c * i * hx * (1 + j * hy), g)
    u0 = on_plane((c, i, j) -> sinpi(c * i * hx) * sinpi(j * hy), g)
    fn = ODEFunction(two_fields(g); jac = two_fields_jac(g), paramjac = two_fields_paramjac(g))
    return PETScDiffEq._discrete_adjoint(
        ODEProblem(fn, u0, (0.0, 0.02), copy(P2)), alg, PETScAdjoint();
        t = collect(0.0:0.005:0.02), dgdu_discrete = (out, u, p, t, i) -> (out .= u .+ p[2] .* w),
        dgdp_discrete = (out, u, p, t, i) -> (fill!(out, 0.0); out[2] = sum(u .* w)), dt = DT,
        adaptive = false,
    )
end

function natural(x, g, c)
    full = zeros(2, NX, NY)
    a = reshape_local_array(x, g)
    for I in points(g), k in 1:2
        full[k, I] = a[k, I]
    end
    return MPI.Allreduce(vec(full), +, c)
end

@testset "MPI adjoint, $nranks ranks" begin
    @testset "matches the serial adjoint and finite differences: $name, $dir" for (name, make) in
            METHODS, (dir, (tspan, times, p)) in (("forward", FORWARD), ("backward", BACKWARD))
        alg = make(comm)
        du0, dp = gradient(heat_problem(rows, halo; tspan, p), alg, times, rows)
        @test same_everywhere(dp)
        mine = vcat(gathered(du0, counts), vec(dp))
        fd = differenced(alg, tspan, times, p)
        if rank == 0
            serial = gradient(heat_problem(1:N, walls; tspan, p), make(MPI.COMM_SELF), times, 1:N)
            reference = vcat(serial[1], vec(serial[2]))
            @test relerr(mine, reference) <= SERIAL_GAP
            @test relerr(mine, fd) <= FD_GAP
        end
    end

    @testset "a SplitODEProblem matches the serial adjoint and finite differences: $name, $dir" for (
                name, make,
            ) in METHODS[5:6], (dir, (tspan, times, p)) in (("forward", FORWARD), ("backward", BACKWARD))
        alg = make(comm)
        du0, dp = gradient(split_problem(rows, halo; tspan, p), alg, times, rows)
        @test same_everywhere(dp)
        mine = vcat(gathered(du0, counts), vec(dp))
        fd = differenced(alg, tspan, times, p; problem = split_problem)
        if rank == 0
            serial = gradient(split_problem(1:N, walls; tspan, p), make(MPI.COMM_SELF), times, 1:N)
            @test relerr(mine, vcat(serial[1], vec(serial[2]))) <= SPLIT_GAP
            @test relerr(mine, fd) <= FD_GAP
        end
    end

    @testset "a dm matches the comm-mode adjoint: $name, $dir" for (name, make) in METHODS,
            (dir, (tspan, times, p)) in (("forward", FORWARD), ("backward", BACKWARD))
        du0, dp = gradient(dm_problem(; tspan, p), with_dm(name), times, rows)
        @test same_everywhere(dp)
        ref = gradient(heat_problem(rows, halo; tspan, p), make(comm), times, rows)
        mine = vcat(gathered(du0, counts), vec(dp))
        theirs = vcat(gathered(ref[1], counts), vec(ref[2]))
        rank == 0 && @test relerr(mine, theirs) <= DM_GAP
    end

    @testset "a 2-D DMDA with two fields matches one on MPI.COMM_SELF: $name, $processors" for (
                name, make,
            ) in (
                ("TSRK", (c, g) -> TSRK("4"; dm = g)),
                ("Crank-Nicolson", (c, g) -> TSImplicit("cn", exact(c); dm = g)),
                ("ARKIMEX l2", (c, g) -> TSARKIMEX("l2", exact(c); dm = g)),
            ), processors in ((1, nranks), (nranks, 1))
        g = plane(comm, processors)
        du0, dp = plane_gradient(g, make(comm, g))
        @test same_everywhere(dp)
        mine = vcat(natural(du0, g, comm), vec(dp))
        PETScCompat.destroy!(g)
        if rank == 0
            solo = plane(MPI.COMM_SELF)
            sdu0, sdp = plane_gradient(solo, make(MPI.COMM_SELF, solo))
            @test relerr(mine, vcat(natural(sdu0, solo, MPI.COMM_SELF), vec(sdp))) <= DM_GAP
            PETScCompat.destroy!(solo)
        end
    end

    @testset "the cost sees the states solve saves" begin
        times = FORWARD[2]
        for alg in (TSRK("4"; comm), TSImplicit("beuler", ["-sub_pc_type", "jacobi"]; comm))
            seen = Vector{Float64}[]
            record = (out, u, p, t, i) -> (push!(seen, copy(u)); out .= u; nothing)
            PETScDiffEq._discrete_adjoint(
                heat_problem(rows, halo), alg, PETScAdjoint(); t = times, dgdu_discrete = record,
                dt = DT, adaptive = false,
            )
            sol = solve(heat_problem(rows, halo), alg; dt = DT, adaptive = false, saveat = times)
            @test reverse(seen) == sol.u
        end
        for alg in (TSRK("4"; dm = da), TSImplicit("beuler", ["-sub_pc_type", "jacobi"]; dm = da))
            seen = Vector{Float64}[]
            record = (out, u, p, t, i) -> (push!(seen, copy(u)); out .= u; nothing)
            PETScDiffEq._discrete_adjoint(
                dm_problem(), alg, PETScAdjoint(); t = times, dgdu_discrete = record, dt = DT,
                adaptive = false,
            )
            sol = solve(dm_problem(), alg; dt = DT, adaptive = false, saveat = times)
            @test reverse(seen) == sol.u
        end
    end

    @testset "a trajectory of states only" begin
        tspan, times, p = FORWARD
        only = TSRK("4", ["-ts_trajectory_solution_only", "1"]; comm)
        du0, dp = gradient(heat_problem(rows, halo), only, times, rows)
        full = gradient(heat_problem(rows, halo), TSRK("4"; comm), times, rows)
        mine, ref = vcat(gathered(du0, counts), vec(dp)), vcat(gathered(full[1], counts), vec(full[2]))
        rank == 0 && @test relerr(mine, ref) <= SERIAL_GAP
    end

    @testset "a function throwing on one rank raises on every rank: $what, $name" for (
            what, name, alg, pieces,
        ) in (
            ("jac", "TSRK", TSRK("4"; comm), :adjoint),
            ("jac", "backward Euler", TSImplicit("beuler", exact(comm); comm), :forward),
            ("jac", "backward Euler", TSImplicit("beuler", exact(comm); comm), :adjoint),
            ("paramjac", "TSRK", TSRK("4"; comm), :adjoint),
            ("paramjac", "Crank-Nicolson", TSImplicit("cn", exact(comm); comm), :adjoint),
            ("f", "TSRK", TSRK("4"; comm), :forward),
            ("f", "backward Euler", TSImplicit("beuler", exact(comm); comm), :forward),
            ("f", "TSRK", TSRK("4", ["-ts_trajectory_solution_only", "1"]; comm), :adjoint),
            ("dgdu", "TSRK", TSRK("4"; comm), :adjoint),
            ("dgdp", "TSRK", TSRK("4"; comm), :after),
            ("jac", "TSRK with a dm", TSRK("4"; dm = da), :adjoint),
            ("jac", "backward Euler with a dm", with_dm("backward Euler"), :adjoint),
            ("paramjac", "TSRK with a dm", TSRK("4"; dm = da), :adjoint),
            ("paramjac", "Crank-Nicolson with a dm", with_dm("Crank-Nicolson"), :adjoint),
            ("f", "TSRK with a dm", TSRK("4", ["-ts_trajectory_solution_only", "1"]; dm = da), :adjoint),
            ("dgdu", "TSRK with a dm", TSRK("4"; dm = da), :adjoint),
            ("dgdp", "TSRK with a dm", TSRK("4"; dm = da), :after),
        )
        started = Ref(false)
        when = pieces === :adjoint ? (args...) -> started[] :
            pieces === :forward ? (args...) -> args[end] > 0.05 : (args...) -> args[end] == 4
        pick(key, f) = key == what ? throwing(f, what, when) : f
        du = pick("dgdu", cost_du(rows))
        dgdu = (args...) -> (started[] = true; du(args...))
        prob = if alg.dm === nothing
            heat_problem(
                rows, halo; f = pick("f", heat(rows, halo)), jac = pick("jac", heat_jac(rows)),
                paramjac = pick("paramjac", heat_paramjac(rows, halo)),
            )
        else
            dm_problem(;
                f = pick("f", dm_heat!), jac = pick("jac", dm_jac!),
                paramjac = pick("paramjac", dm_paramjac!),
            )
        end
        e = caught() do
            PETScDiffEq._discrete_adjoint(
                prob, alg, PETScAdjoint(); t = FORWARD[2], dgdu_discrete = dgdu,
                dgdp_discrete = pick("dgdp", cost_dp(rows)), dt = DT, adaptive = false,
            )
        end
        @test raised(e, "$what threw")
    end

    @testset "f2's $what throwing on one rank raises on every rank" for what in ("jac", "paramjac")
        pick(key, f) = key == what ? throwing(f, what, (args...) -> true) : f
        prob = split_problem(
            rows, halo; jac = pick("jac", reaction_jac(rows)),
            paramjac = pick("paramjac", reaction_paramjac(rows)),
        )
        e = caught(() -> gradient(prob, TSARKIMEX("l2", exact(comm); comm), FORWARD[2], rows))
        @test raised(e, "$what threw")
    end

    @testset "refusals" begin
        times = FORWARD[2]
        rk = TSRK("4"; comm)
        heat_with(; kw...) = heat_problem(rows, halo; kw...)
        @test refused(
            () -> gradient(heat_with(jac = nothing), rk, times, rows), "needs the ODEFunction's `jac`",
        )
        for alg in (rk, TSImplicit("beuler"; comm, autodiff = AutoForwardDiff()))
            @test refused(
                () -> gradient(heat_with(paramjac = nothing), alg, times, rows),
                "needs the ODEFunction's `paramjac`",
            )
        end
        colouring = TSImplicit("beuler"; comm)
        for (prob, what) in (
                (heat_with(jac = nothing), "`jac`"), (heat_with(paramjac = nothing), "`paramjac`"),
            )
            @test refused(
                () -> gradient(prob, colouring, times, rows),
                "$what under `autodiff = AutoFiniteDiff()`",
            )
        end
        some = heat_with(paramjac = rank == thrower ? nothing : heat_paramjac(rows, halo))
        e = caught(() -> gradient(some, rk, times, rows))
        @test rank == thrower ? e isa ArgumentError && occursin("`paramjac`", e.msg) : remote(e)
        @test refused(
            () -> gradient(heat_with(jac_prototype = nothing), rk, times, rows),
            "sparse `jac_prototype`",
        )
        @test refused(
            () -> gradient(heat_with(), rk, times, rows; g = (u, p, t) -> sum(abs2, u)),
            "integral cost on MPI.COMM_SELF only",
        )
        imex = TSARKIMEX("l2"; comm, autodiff = AutoForwardDiff())
        for (kw, what) in (
                ((; jac = nothing), "needs `f2`'s `jac` on a communicator"),
                ((; paramjac = nothing), "needs `f2`'s `paramjac` on a communicator"),
                ((; jac_prototype = nothing), "to come with a sparse `jac_prototype`"),
            )
            @test refused(() -> gradient(split_problem(rows, halo; kw...), imex, times, rows), what)
        end
        taller = rank == thrower ? [reaction_proto(rows); spzeros(1, N)] : reaction_proto(rows)
        e = caught(() -> gradient(split_problem(rows, halo; jac_prototype = taller), imex, times, rows))
        @test rank == thrower ? e isa ArgumentError && occursin("so $(length(rows)) x $N", e.msg) :
            remote(e)
        halves = SplitODEProblem(
            dm_heat!, (du, u, p, t) -> (du .= 0; nothing), heat0(rows), first(FORWARD), copy(P),
        )
        nranks > 1 && @test refused(
            () -> gradient(halves, with_dm("ARKIMEX l2"), times, rows),
            "SplitODEProblem with a `dm` on a DM of a single rank only",
        )
        wrong = rank == thrower ? [heat_proto(rows); spzeros(1, N)] : heat_proto(rows)
        e = caught(() -> gradient(heat_with(jac_prototype = wrong), rk, times, rows))
        @test rank == thrower ? e isa ArgumentError && occursin("must be", e.msg) : remote(e)
        not_bool = rank == thrower ? nothing : false
        e = caught(() -> gradient(heat_with(), rk, times, rows; no_start = not_bool))
        @test rank == thrower ? e isa MethodError : remote(e)
        for (alg, what) in (
                (TSImplicit("bdf"; comm), "PETSc has no adjoint for TSImplicit(\"bdf\")"),
                (TSRosW(; comm), "PETSc has no adjoint for TSRosW"),
            )
            @test refused(() -> gradient(heat_with(), alg, times, rows), what)
        end
        @test refused(
            () -> gradient(heat_with(), rk, times, rows; maxiters = 5), "the forward solve stopped",
        )
        if nranks > 1
            short = rank == thrower ? times[1:(end - 1)] : times
            @test refused(() -> gradient(heat_with(), rk, short, rows), "the same cost times")
            @test refused(
                () -> gradient(heat_with(), rk, times, rows; no_start = rank == thrower),
                "the same cost times",
            )
            longer = heat_with(p = rank == thrower ? [P; 0.0] : copy(P))
            @test refused(() -> gradient(longer, rk, times, rows), "the same cost times")
            some_dgdp = rank == thrower ? nothing : cost_dp(rows)
            @test refused(
                () -> gradient(heat_with(), rk, times, rows; dgdp_discrete = some_dgdp),
                "the same cost times",
            )
        end
        grow!(du, u, p, t) = (du .= rank == thrower ? u .^ 2 : -u; nothing)
        grow_jac!(J, u, p, t) =
            (foreach(k -> J[k, rows[k]] = rank == thrower ? 2u[k] : -1.0, eachindex(rows)); nothing)
        diagonal = sparse(1:length(rows), rows, 1.0, length(rows), N)
        rk_dm = TSRK("4"; dm = da)
        for (prob, what) in (
                (dm_problem(jac = nothing), "the ODEFunction's `jac` with a `dm`"),
                (dm_problem(paramjac = nothing), "the ODEFunction's `paramjac` with a `dm`"),
            )
            @test refused(() -> gradient(prob, rk_dm, times, rows), "PETScAdjoint needs $what")
        end
        some = dm_problem(paramjac = rank == thrower ? nothing : dm_paramjac!)
        e = caught(() -> gradient(some, rk_dm, times, rows))
        @test rank == thrower ? e isa ArgumentError && occursin("`paramjac`", e.msg) : remote(e)
        nranks > 1 && @test refused(
            () -> gradient(dm_problem(), rk_dm, times, rows; g = (u, p, t) -> sum(abs2, u)),
            "integral cost on MPI.COMM_SELF only",
        )
        grows = ODEProblem(
            ODEFunction(grow!; jac = grow_jac!, jac_prototype = diagonal), ones(length(rows)),
            (0.0, 2.0),
        )
        @test refused(
            () -> PETScDiffEq._discrete_adjoint(
                grows, rk, PETScAdjoint(); t = [2.0], dgdu_discrete = (out, u, p, t, i) -> (out .= u),
                dt = 0.01, adaptive = false,
            ),
            "not finite",
        )
    end

    @testset "every handle is freed" begin
        @test isempty(PETScDiffEq.PARALLEL_HANDLES)
        @test all(h -> h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
    end
end

PETScCompat.destroy!(da)
