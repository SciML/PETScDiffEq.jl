using MPI, PETScDiffEq, SciMLBase, SparseArrays, Test
using PETScDiffEq: PETSc, PETScAdjoint, AutoForwardDiff
using SciMLBase: ODEProblem, ODEFunction, solve

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
# PetscInitialize is collective, and the serial reference solves run on single ranks.
PETSc.initialize(PETSc.getlib(; PetscScalar = Float64))

const N = 23
const P = [0.8, 1.5, 0.5]
const DT = 1.0e-3
const FORWARD = ((0.0, 0.1), collect(0.0:0.01:0.1), P)
const BACKWARD = ((0.1, 0.0), collect(0.1:-0.01:0.0), [-P[1], P[2], -P[3]])
const EXACT = ["-snes_rtol", "1e-13", "-snes_atol", "1e-15", "-ksp_type", "preonly"]
const SERIAL_GAP = 1.0e-15
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

cost(u, p, idx) = sum(abs2, u) / 2 + p[2] * sum(u .* x.(idx))
cost_du(idx) = (out, u, p, t, i) -> (out .= u .+ p[2] .* x.(idx); nothing)
cost_dp(idx) = (out, u, p, t, i) -> (fill!(out, 0.0); out[2] = sum(u .* x.(idx)); nothing)

function gradient(prob, alg, times, idx; kw...)
    return PETScDiffEq._discrete_adjoint(
        prob, alg, PETScAdjoint(); t = times, dgdu_discrete = cost_du(idx),
        dgdp_discrete = cost_dp(idx), dt = DT, adaptive = false, kw...,
    )
end

function loss(alg, tspan, times, u0, p)
    sol = solve(heat_problem(rows, halo; tspan, u0, p), alg; dt = DT, adaptive = false, saveat = times)
    return MPI.Allreduce(sum(cost(u, p, rows) for u in sol.u), +, comm)
end

function differenced(alg, tspan, times, p0; h = 1.0e-6)
    g = zeros(N + length(p0))
    for j in eachindex(g)
        function at(s)
            u0, p = heat0(rows), copy(p0)
            if j > N
                p[j - N] += s
            elseif j in rows
                u0[j - first(rows) + 1] += s
            end
            return loss(alg, tspan, times, u0, p)
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
)

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
        )
        started = Ref(false)
        when = pieces === :adjoint ? (args...) -> started[] :
            pieces === :forward ? (args...) -> args[end] > 0.05 : (args...) -> args[end] == 4
        pick(key, f) = key == what ? throwing(f, what, when) : f
        du = pick("dgdu", cost_du(rows))
        dgdu = (args...) -> (started[] = true; du(args...))
        prob = heat_problem(
            rows, halo; f = pick("f", heat(rows, halo)), jac = pick("jac", heat_jac(rows)),
            paramjac = pick("paramjac", heat_paramjac(rows, halo)),
        )
        e = caught() do
            PETScDiffEq._discrete_adjoint(
                prob, alg, PETScAdjoint(); t = FORWARD[2], dgdu_discrete = dgdu,
                dgdp_discrete = pick("dgdp", cost_dp(rows)), dt = DT, adaptive = false,
            )
        end
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
        some = heat_with(paramjac = rank == thrower ? nothing : heat_paramjac(rows, halo))
        e = caught(() -> gradient(some, rk, times, rows))
        @test rank == thrower ? e isa ArgumentError && occursin("`paramjac`", e.msg) : remote(e)
        @test refused(
            () -> gradient(heat_with(jac_prototype = nothing), rk, times, rows),
            "sparse `jac_prototype`",
        )
        wrong = rank == thrower ? [heat_proto(rows); spzeros(1, N)] : heat_proto(rows)
        e = caught(() -> gradient(heat_with(jac_prototype = wrong), rk, times, rows))
        @test rank == thrower ? e isa ArgumentError && occursin("must be", e.msg) : remote(e)
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
        end
        grow!(du, u, p, t) = (du .= rank == thrower ? u .^ 2 : -u; nothing)
        grow_jac!(J, u, p, t) =
            (foreach(k -> J[k, rows[k]] = rank == thrower ? 2u[k] : -1.0, eachindex(rows)); nothing)
        diagonal = sparse(1:length(rows), rows, 1.0, length(rows), N)
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
