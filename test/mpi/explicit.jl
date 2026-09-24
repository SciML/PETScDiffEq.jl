using MPI, PETScDiffEq, SciMLBase, Test
using PETScDiffEq: PETSc
using SciMLBase: ODEProblem, ODEFunction, DAEProblem, DiscreteCallback, ReturnCode, init, solve

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
# PetscInitialize is collective, and the serial reference solves run on single ranks.
PETSc.initialize(PETSc.getlib(; PetscScalar = Float64))

const N = 23
const DECAY_ERR = 1.0e-10
const HEAT_ERR = 1.0e-8
const ROUNDOFF = 1.0e-14

function block(n; first_empty = false)
    w = [first_empty && r == 0 && nranks > 1 ? 0 : r + 1 for r in 0:(nranks - 1)]
    counts = floor.(Int, n .* w ./ sum(w))
    counts[end] += n - sum(counts)
    lo = sum(counts[1:rank]) + 1
    return lo:(lo + counts[rank + 1] - 1), counts
end

const rows, counts = block(N)
const thrower = nranks - 1

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

rate(i) = 1 + i / N
decay!(du, u, idx, t) = (du .= -rate.(idx) .* u; nothing)
decay0(idx) = [1.0 + (i % 5) / 10 for i in idx]
decay_exact(idx, t) = decay0(idx) .* exp.(-rate.(idx) .* t)
decay_problem(idx) = ODEProblem(decay!, decay0(idx), (0.0, 1.0), idx)

const dx = 1 / (N + 1)
heat0(idx) = sinpi.(idx .* dx) .+ 0.5 .* sinpi.(3 .* idx .* dx)
eigval(k) = -4 / dx^2 * sinpi(k * dx / 2)^2
heat_exact(idx, t) = exp(eigval(1) * t) .* sinpi.(idx .* dx) .+
    0.5 * exp(eigval(3) * t) .* sinpi.(3 .* idx .* dx)

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

function heat_throwing(when)
    return function (du, u, p, t)
        heat!(du, u, p, t)
        rank == thrower && when(t) && error("f threw on rank $rank")
        return nothing
    end
end

@testset "MPI, $nranks ranks" begin
    tol = (abstol = 1.0e-10, reltol = 1.0e-10)

    @testset "decoupled decay" begin
        sol = solve(decay_problem(rows), TSRK("5dp"; comm); tol...)
        @test sol.retcode == ReturnCode.Success
        @test same_everywhere(sol.t)
        us = gathered(sol, counts)
        if rank == 0
            ref = solve(decay_problem(1:N), TSRK("5dp"); tol...)
            @test maximum(abs, us[end] - ref.u[end]) <= DECAY_ERR
            @test maximum(abs, us[end] - decay_exact(1:N, 1.0)) <= DECAY_ERR
        end
    end

    @testset "saveat" begin
        sol = solve(decay_problem(rows), TSRK("5dp"; comm); saveat = 0.1, tol...)
        @test sol.t == collect(0.0:0.1:1.0)
        us = gathered(sol, counts)
        if rank == 0
            ref = solve(decay_problem(1:N), TSRK("5dp"); saveat = 0.1, tol...)
            @test maxdiff(us, ref.u) <= DECAY_ERR
            @test maxdiff(us, [decay_exact(1:N, t) for t in sol.t]) <= DECAY_ERR
        end
    end

    @testset "1-D heat with a halo exchange" begin
        span = (0.0, 0.1)
        serial = ODEProblem(heat_serial!, heat0(1:N), span)
        for (alg, serial_alg) in (
                (TSRK("5dp"; comm), TSRK("5dp")),
                (TSGeneric("ssp"; explicit = true, comm), TSGeneric("ssp"; explicit = true)),
            )
            sol = solve(ODEProblem(heat!, heat0(rows), span), alg; dt = 1.0e-3, adaptive = false)
            @test sol.retcode == ReturnCode.Success
            mids = (sol.t[1:(end - 1)] .+ sol.t[2:end]) ./ 2
            us, ms = gathered(sol, counts), [gathered(sol(t), counts) for t in mids]
            if rank == 0
                ref = solve(serial, serial_alg; dt = 1.0e-3, adaptive = false)
                @test sol.t == ref.t
                @test maxdiff(us, ref.u) <= ROUNDOFF
                @test maxdiff(ms, [ref(t) for t in mids]) <= ROUNDOFF
            end
        end
        sol = solve(
            ODEProblem(heat!, heat0(rows), span), TSRK("5dp"; comm);
            abstol = 1.0e-8, reltol = 1.0e-8,
        )
        @test same_everywhere(sol.t)
        u = gathered(sol.u[end], counts)
        if rank == 0
            ref = solve(serial, TSRK("5dp"); abstol = 1.0e-8, reltol = 1.0e-8)
            @test maximum(abs, u - ref.u[end]) <= HEAT_ERR
            @test maximum(abs, u - heat_exact(1:N, span[2])) <= HEAT_ERR
        end
    end

    @testset "tstops, d_discontinuities, save_idxs and vector tolerances" begin
        for kw in ((; tstops = [0.25, 0.5]), (; d_discontinuities = [0.25, 0.5]))
            sol = solve(decay_problem(rows), TSRK("5dp"; comm); kw..., tol...)
            @test 0.25 in sol.t && 0.5 in sol.t
            @test same_everywhere(sol.t)
            u = gathered(sol.u[end], counts)
            rank == 0 && @test maximum(abs, u - decay_exact(1:N, 1.0)) <= DECAY_ERR
        end

        n = length(rows)
        sol = solve(
            decay_problem(rows), TSRK("5dp"; comm);
            abstol = fill(1.0e-10, n), reltol = fill(1.0e-10, n), save_idxs = [n],
        )
        ref = solve(
            decay_problem(1:N), TSRK("5dp");
            abstol = fill(1.0e-10, N), reltol = fill(1.0e-10, N), save_idxs = [rows[end]],
        )
        @test all(u -> length(u) == 1, sol.u)
        @test abs(sol.u[end][1] - ref.u[end][1]) <= DECAY_ERR
        @test abs(sol.u[end][1] - decay_exact(rows, 1.0)[end]) <= DECAY_ERR
    end

    @testset "a rank with no rows" begin
        rows0, counts0 = block(N; first_empty = true)
        sol = solve(decay_problem(rows0), TSRK("5dp"; comm); tol...)
        @test sol.retcode == ReturnCode.Success
        @test same_everywhere(sol.t)
        u = gathered(sol.u[end], counts0)
        rank == 0 && @test maximum(abs, u - decay_exact(1:N, 1.0)) <= DECAY_ERR
    end

    @testset "a check that fires on one rank stops every rank" begin
        low(u) = any(<(0.6), u)
        sol = solve(
            decay_problem(rows), TSRK("5dp"; comm); unstable_check = (dt, u, p, t) -> low(u),
        )
        @test sol.retcode == ReturnCode.Unstable
        @test same_everywhere(sol.t)
        @test anywhere(low(sol.u[end])) && !anywhere(low(sol.u[end - 1]))
        sol = solve(decay_problem(rows), TSRK("5dp"; comm); isoutofdomain = (u, p, t) -> low(u))
        @test sol.retcode == ReturnCode.Unstable
        @test same_everywhere(sol.t)
        @test !anywhere(any(low, sol.u))
    end

    @testset "a state that stops being finite on one rank" begin
        blowup!(du, u, idx, t) = (halo(u); du .= ifelse.(idx .> N - 3, u .^ 2, -u); nothing)
        prob = ODEProblem(blowup!, ones(length(rows)), (0.0, 1.5), rows)
        for kw in ((;), (; tstops = [1.25]))
            sol = solve(prob, TSRK("5dp"; comm); dt = 0.01, adaptive = false, kw...)
            @test sol.retcode == ReturnCode.Unstable
            @test same_everywhere(sol.t)
        end
    end

    @testset "f throwing on one rank raises on every rank" begin
        prob(when) = ODEProblem(heat_throwing(when), heat0(rows), (0.0, 0.1))
        after = t -> t > 0.02
        for (when, kw) in (
                (t -> t == 0, (;)),
                (after, (; dt = 1.0e-3)),
                (after, (; dt = 1.0e-3, adaptive = false)),
                (after, (; dt = 1.0e-3, tstops = [0.05])),
                (t -> t == 0.0105, (; dt = 1.0e-3, adaptive = false, saveat = [0.0105], dense = true)),
            )
            @test raised(caught(() -> solve(prob(when), TSRK("5dp"; comm); kw...)), "f threw")
        end
        never = prob(t -> false)
        check = (dt, u, p, t) -> (rank == thrower && t > 0.02 && error("check threw"); false)
        @test raised(
            caught(() -> solve(never, TSRK("5dp"; comm); dt = 1.0e-3, unstable_check = check)),
            "check threw",
        )
        domain = (u, p, t) -> (rank == thrower && t > 0.02 && error("domain threw"); false)
        @test raised(
            caught(() -> solve(never, TSRK("5dp"; comm); dt = 1.0e-3, isoutofdomain = domain)),
            "domain threw",
        )
        sol = solve(never, TSRK("5dp"; comm); dt = 1.0e-3)
        @test sol.retcode == ReturnCode.Success
    end

    @testset "refusals" begin
        prob = decay_problem(rows)
        for alg in (
                TSRosW(; comm), TSImplicit("beuler"; comm), TSIRK(2; comm), TSARKIMEX(; comm),
                TSMPRK([1]; comm), TSGeneric("alpha"; comm),
            )
            @test refused(() -> solve(prob, alg; dt = 0.1), "runs only TSRK")
        end
        dae = DAEProblem(
            (r, du, u, p, t) -> (r .= du .+ u; nothing), -decay0(rows), decay0(rows), (0.0, 1.0),
        )
        @test refused(() -> solve(dae, TSDAE(; comm); dt = 0.1), "runs only TSRK")
        n = length(rows)
        jac = ODEFunction(decay!; jac = (J, u, p, t) -> nothing)
        @test refused(
            () -> solve(ODEProblem(jac, decay0(rows), (0.0, 1.0), rows), TSRK("5dp"; comm)),
            "does not take a `jac`",
        )
        mass = ODEFunction(decay!; mass_matrix = [i == j ? 2.0 : 0.0 for i in 1:n, j in 1:n])
        @test refused(
            () -> solve(ODEProblem(mass, decay0(rows), (0.0, 1.0), rows), TSRK("5dp"; comm)),
            "mass matrix",
        )
        cb = DiscreteCallback((u, t, i) -> false, i -> nothing)
        @test refused(() -> solve(prob, TSRK("5dp"; comm); callback = cb), "callbacks")
        @test refused(() -> init(prob, TSRK("5dp"; comm); dt = 0.1), "integrator interface")
        @test refused(
            () -> PETScDiffEq._discrete_adjoint(
                prob, TSRK("4"; comm), PETScAdjoint(); t = [1.0],
                dgdu_discrete = (out, u, p, t, i) -> (out .= u), dt = 0.1, adaptive = false,
            ),
            "PETScAdjoint",
        )
        @test refused(
            () -> solve(prob, TSRK("5dp", ["-ts_type", "beuler"]; comm); dt = 0.1),
            "`beuler` cannot run",
        )
        bad = fill(1.0e-8, n + (rank == thrower))
        e = caught(() -> solve(prob, TSRK("5dp"; comm); abstol = bad))
        @test rank == thrower ? e isa ArgumentError && occursin("abstol", e.msg) : remote(e)
        e = caught(() -> solve(prob, TSRK("5dp"; comm); save_idxs = [rank == thrower ? n + 1 : 1]))
        @test rank == thrower ? e isa ArgumentError && occursin("save_idxs", e.msg) : remote(e)
    end

    @testset "every handle is freed" begin
        @test isempty(PETScDiffEq.PARALLEL_HANDLES)
        @test all(h -> h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
    end
end
