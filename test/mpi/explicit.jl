using MPI, PETScDiffEq, SciMLBase, Test
using PETScDiffEq: PETSc
using SciMLBase: ODEProblem, ODEFunction, DiscreteCallback, ReturnCode, init, solve
using SciMLBase: ContinuousCallback, VectorContinuousCallback, CallbackSet, terminate!, step!,
    solve!, reinit!, set_u!, get_du, add_tstop!, add_saveat!, savevalues!,
    change_t_via_interpolation!, set_proposed_dt!

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

const FIXED = (; dt = 1.0e-3, adaptive = false)
adaptive_step(kw) = !haskey(kw, :dt)
algorithm_pairs() = (
    (TSRK("5dp"; comm), TSRK("5dp")),
    (TSGeneric("ssp"; explicit = true, comm), TSGeneric("ssp"; explicit = true)),
)

parallel(alg) = alg.comm != MPI.COMM_SELF
heat_problem(idx, alg) =
    ODEProblem(parallel(alg) ? heat! : heat_serial!, heat0(idx), (0.0, 0.1))
local_row(idx, g) = findfirst(==(g), idx)
first_block(idx) = [g <= counts[1] ? 0.5 : 1.0 for g in idx]

function against_serial(run, alg, serial_alg)
    got = run(rows, alg)
    return got, rank == 0 ? run(1:N, serial_alg) : got
end

nan_gap(a, b) = isnan.(a) == isnan.(b) ? maximum(abs, replace(a - b, NaN => 0.0)) : Inf

function matches_serial(sol, ref; exact_t = true)
    n = length(sol.t)
    MPI.Allreduce(n, min, comm) == MPI.Allreduce(n, max, comm) || return false
    same = same_everywhere(sol.t)
    us = gathered(sol, counts)
    rank == 0 || return same
    length(sol.t) == length(ref.t) || return false
    times = exact_t ? sol.t == ref.t : maximum(abs, sol.t - ref.t) <= ROUNDOFF
    gap = maximum(nan_gap.(us, ref.u))
    return same && times && sol.retcode == ref.retcode && gap <= ROUNDOFF
end

halve_at(s) = DiscreteCallback((u, t, i) -> t == s, i -> (i.u .*= 0.5))
crossing_last_row(idx, level) =
    (u, t, i) -> (j = local_row(idx, N); j === nothing ? 1.0 : u[j] - level)

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

    @testset "a NaN on one rank's rows is retried or stops as a serial solve does" begin
        breaks!(du, u, idx, t) =
            (du .= ifelse.((idx .== N) .& (t > 0.5), NaN, -rate.(idx) .* u); nothing)
        exchanging!(du, u, idx, t) = (halo(u); breaks!(du, u, idx, t))
        breaks(idx, a) =
            ODEProblem(parallel(a) ? exchanging! : breaks!, decay0(idx), (0.0, 1.0), idx)
        never = DiscreteCallback((u, t, i) -> false, i -> nothing)
        warned(f) = (r = Test.collect_test_logs(f); (r[2], length(r[1])))
        unstable, small = ReturnCode.Unstable, ReturnCode.DtLessThanMin
        for (subtype, kw, retcode) in (
                ("5dp", (;), unstable),
                ("5dp", (; saveat = 0.05), unstable),
                ("3bs", (; callback = never), unstable),
                ("5dp", (; dtmin = 0.01), small),
                ("5dp", (; dtmin = 0.01, callback = never), small),
                ("5dp", FIXED, unstable),
                ("4", (; dt = 0.01), unstable),
                ("4", (; dt = 0.01, callback = never), unstable),
            )
            alg, serial_alg = TSRK(subtype; comm), TSRK(subtype)
            (sol, warnings), (ref, _) = against_serial(alg, serial_alg) do idx, a
                warned(() -> solve(breaks(idx, a), a; kw...))
            end
            @test sol.retcode == retcode
            @test warnings == (adaptive_step(kw) ? 1 : 0)
            @test matches_serial(sol, ref)
            @test anywhere(!all(isfinite, sol.u[end])) == !adaptive_step(kw)
            @test !anywhere(any(u -> !all(isfinite, u), sol.u[1:(end - 1)]))
            @test same_everywhere((sol.stats.naccept, sol.stats.nreject))
            rank == 0 && @test (sol.stats.naccept, sol.stats.nreject, sol.stats.nf) ==
                (ref.stats.naccept, ref.stats.nreject, ref.stats.nf)
            rank == 0 && adaptive_step(kw) && @test sol.stats.nreject > 5
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

    @testset "discrete callbacks" begin
        for (alg, serial_alg) in algorithm_pairs()
            sol, ref = against_serial(alg, serial_alg) do idx, a
                solve(heat_problem(idx, a), a; callback = halve_at(0.05), tstops = [0.05], FIXED...)
            end
            @test matches_serial(sol, ref)

            fired = Ref(0)
            sol, ref = against_serial(alg, serial_alg) do idx, a
                n = Ref(0)
                cb = DiscreteCallback(
                    (u, t, i) -> N in idx && t >= 0.05 && n[] == 0, i -> (n[] += 1; i.u .*= 0.5),
                )
                s = solve(heat_problem(idx, a), a; callback = cb, tstops = [0.05], FIXED...)
                parallel(a) && (fired[] = n[])
                s
            end
            @test fired[] == 1
            @test matches_serial(sol, ref)

            sol, ref = against_serial(alg, serial_alg) do idx, a
                function nudge!(i)
                    j = local_row(idx, N)
                    j === nothing ? SciMLBase.derivative_discontinuity!(i, false) : (i.u[j] += 0.1)
                    return nothing
                end
                cb = DiscreteCallback((u, t, i) -> t == 0.05, nudge!)
                solve(heat_problem(idx, a), a; callback = cb, tstops = [0.05], FIXED...)
            end
            @test matches_serial(sol, ref)
        end
    end

    @testset "continuous callbacks" begin
        for (alg, serial_alg) in algorithm_pairs()
            for interp_points in (10, 1)
                fired = Ref(0)
                sol, ref = against_serial(alg, serial_alg) do idx, a
                    n = Ref(0)
                    affect!(i) = (n[] += 1; i.u .*= 2)
                    cb = ContinuousCallback(crossing_last_row(idx, 0.15), affect!; interp_points)
                    s = solve(heat_problem(idx, a), a; callback = cb, FIXED...)
                    parallel(a) && (fired[] = n[])
                    s
                end
                @test fired[] == 2
                @test matches_serial(sol, ref; exact_t = false)
            end

            sol, ref = against_serial(alg, serial_alg) do idx, a
                total(u) = parallel(a) ? MPI.Allreduce(sum(u), +, comm) : sum(u)
                cb = ContinuousCallback((u, t, i) -> total(u) - 8.0, terminate!)
                solve(heat_problem(idx, a), a; callback = cb, FIXED...)
            end
            @test sol.retcode == ReturnCode.Terminated
            @test matches_serial(sol, ref; exact_t = false)

            got, ref = against_serial(alg, serial_alg) do idx, a
                events = Tuple{Float64, Vector{Int}}[]
                function conditions(out, u, t, i)
                    j, k = local_row(idx, N), local_row(idx, 1)
                    out[1] = j === nothing ? 1.0 : u[j] - 0.25
                    out[2] = k === nothing ? 1.0 : u[k] - 0.1
                    return nothing
                end
                affect!(i, mask) = push!(events, (i.t, findall(!iszero, mask)))
                cb = VectorContinuousCallback(conditions, affect!, 2)
                (; sol = solve(heat_problem(idx, a), a; callback = cb, FIXED...), events)
            end
            @test same_everywhere(got.events)
            @test last.(got.events) == last.(ref.events) == [[1], [2]]
            @test maximum(abs, first.(got.events) - first.(ref.events)) <= ROUNDOFF
            @test matches_serial(got.sol, ref.sol; exact_t = false)
        end
    end

    @testset "saveat and terminate! with callbacks" begin
        for (alg, serial_alg) in algorithm_pairs()
            sol, ref = against_serial(alg, serial_alg) do idx, a
                stop = DiscreteCallback((u, t, i) -> t >= 0.08, terminate!)
                cbs = CallbackSet(halve_at(0.05), stop)
                solve(heat_problem(idx, a), a; callback = cbs, tstops = [0.05], saveat = 0.01, FIXED...)
            end
            @test sol.retcode == ReturnCode.Terminated
            @test issubset(0.0:0.01:0.08, sol.t)
            @test matches_serial(sol, ref)

            sol, ref = against_serial(alg, serial_alg) do idx, a
                cb = ContinuousCallback(crossing_last_row(idx, 0.15), i -> (i.u .*= 2))
                solve(heat_problem(idx, a), a; callback = cb, saveat = 0.01, FIXED...)
            end
            @test matches_serial(sol, ref; exact_t = false)
        end
    end

    @testset "callback initialize and finalize" begin
        for (alg, serial_alg) in algorithm_pairs()
            finalized = Ref(0)
            sol, ref = against_serial(alg, serial_alg) do idx, a
                cb = DiscreteCallback(
                    (u, t, i) -> false, i -> nothing;
                    initialize = (c, u, t, i) -> (i.u ./= first_block(idx)),
                    finalize = (c, u, t, i) -> parallel(a) && (finalized[] += 1),
                )
                solve(heat_problem(idx, a), a; callback = cb, FIXED...)
            end
            @test finalized[] == 1
            @test matches_serial(sol, ref)
        end
    end

    @testset "the integrator interface" begin
        for (alg, serial_alg) in algorithm_pairs()
            got, ref = against_serial(alg, serial_alg) do idx, a
                integ = init(heat_problem(idx, a), a; FIXED...)
                add_tstop!(integ, 0.0425)
                add_saveat!(integ, 0.0333)
                mids, dus = Vector{Float64}[], Vector{Float64}[]
                while !SciMLBase.done(integ)
                    step!(integ)
                    push!(mids, integ((integ.tprev + integ.t) / 2))
                    push!(dus, get_du(integ))
                    k = length(mids)
                    k == 20 && set_u!(integ, 0.5 .* integ.u)
                    k == 30 && set_u!(integ, first_block(idx) .* integ.u)
                    k == 40 && (integ.u .*= first_block(idx))
                    k == 50 && savevalues!(integ, true)
                    k == 60 && change_t_via_interpolation!(integ, (integ.tprev + integ.t) / 2)
                end
                first_sol = integ.sol
                reinit!(integ, 2 .* heat0(idx))
                (; first_sol, mids, dus, sol = solve!(integ))
            end
            @test 0.0425 in got.first_sol.t && 0.0333 in got.first_sol.t
            @test matches_serial(got.first_sol, ref.first_sol)
            @test matches_serial(got.sol, ref.sol)
            mids = [gathered(u, counts) for u in got.mids]
            dus = [gathered(du, counts) for du in got.dus]
            if rank == 0
                @test length(mids) == length(ref.mids)
                @test maxdiff(mids, ref.mids) <= ROUNDOFF
                @test maxdiff(dus, ref.dus) <= ROUNDOFF / dx^2
            end
        end

        integ = init(heat_problem(rows, TSRK("5dp"; comm)), TSRK("5dp"; comm))
        step!(integ)
        set_proposed_dt!(integ, rank == 0 ? 1.0e-5 : 1.0e-4)
        step!(integ)
        @test same_everywhere(integ.t)
        @test abs(integ.dt - 1.0e-5) <= ROUNDOFF
        terminate!(integ)
        @test integ.sol.retcode == ReturnCode.Terminated
        @test same_everywhere(integ.sol.t)

        integ = init(heat_problem(rows, TSRK("5dp"; comm)), TSRK("5dp"; comm); dt = 1.0e-5)
        SciMLBase.auto_dt_reset!(integ)
        serial_dt = if rank == 0
            serial = init(heat_problem(1:N, TSRK("5dp")), TSRK("5dp"))
            terminate!(serial)
            serial.dt
        end
        @test abs(integ.dt - MPI.bcast(serial_dt, 0, comm)) <= ROUNDOFF * integ.dt
        step!(integ)
        @test SciMLBase.check_error!(integ) == ReturnCode.Success
        SciMLBase.postamble!(integ)
        @test integ.sol.retcode == ReturnCode.Success
        @test integ.sol.t[end] == integ.t
        @test same_everywhere(integ.sol.t)
    end

    @testset "a callback throwing on one rank raises on every rank" begin
        prob = heat_problem(rows, TSRK("5dp"; comm))
        late(t) = rank == thrower && t > 0.02
        for cb in (
                DiscreteCallback((u, t, i) -> late(t) ? error("condition threw") : false, i -> nothing),
                DiscreteCallback((u, t, i) -> t > 0.02, i -> rank == thrower && error("affect threw")),
                ContinuousCallback((u, t, i) -> late(t) ? error("condition threw") : 1.0, i -> nothing),
                ContinuousCallback(
                    crossing_last_row(rows, 0.15), i -> rank == thrower && error("affect threw"),
                ),
                DiscreteCallback(
                    (u, t, i) -> false, i -> nothing;
                    initialize = (c, u, t, i) -> rank == thrower && error("initialize threw"),
                ),
                DiscreteCallback(
                    (u, t, i) -> false, i -> nothing;
                    finalize = (c, u, t, i) -> rank == thrower && error("finalize threw"),
                ),
            )
            @test raised(caught(() -> solve(prob, TSRK("5dp"; comm); callback = cb, FIXED...)), "threw")
        end

        armed = Ref(false)
        f!(du, u, p, t) = (heat!(du, u, p, t); armed[] && rank == thrower && error("f threw"); nothing)
        integ = init(
            ODEProblem(f!, heat0(rows), (0.0, 0.1)), TSGeneric("ssp"; explicit = true, comm);
            save_everystep = false, FIXED...,
        )
        step!(integ)
        armed[] = true
        @test raised(caught(() -> integ((integ.tprev + integ.t) / 2)), "f threw")
        @test raised(caught(() -> get_du(integ)), "f threw")
        armed[] = false
        step!(integ)
        @test !anywhere(any(isnan, integ((integ.tprev + integ.t) / 2)))
        terminate!(integ)

        integ = init(
            ODEProblem(f!, heat0(rows), (0.0, 0.1)), TSRK("5dp"; comm); dense = false, FIXED...,
        )
        step!(integ)
        armed[] = true
        @test raised(caught(() -> reinit!(integ; reset_dt = true)), "f threw")
        armed[] = false
        terminate!(integ)
    end

    @testset "TSMPRK with its splits spread over the ranks" begin
        slow, medium = 1:11, 12:17
        mine(g, idx) = [k for (k, i) in enumerate(idx) if i in g]
        mprk(idx, sub, c) = sub in ("2a23", "2a33") ?
            TSMPRK(mine(slow, idx), mine(medium, idx), sub; comm = c) :
            TSMPRK(mine(slow, idx), sub; comm = c)
        for sub in ("p2", "p3", "2a22", "2a23", "2a33")
            sol, ref = against_serial(comm, MPI.COMM_SELF) do idx, c
                a = mprk(idx, sub, c)
                prob = ODEProblem(parallel(a) ? heat! : heat_serial!, heat0(idx), (0.0, 0.02))
                solve(prob, a; dt = 1.0e-4)
            end
            @test sol.retcode == ReturnCode.Success
            @test matches_serial(sol, ref)
            rank == 0 && @test sol.stats.nf == ref.stats.nf
            u = gathered(sol.u[end], counts)
            rank == 0 && @test maximum(abs, u - heat_exact(1:N, 0.02)) <= 5.0e-6
        end
        f = heat_throwing(t -> t > 0.01)
        prob = ODEProblem(f, heat0(rows), (0.0, 0.02))
        @test raised(caught(() -> solve(prob, mprk(rows, "p2", comm); dt = 1.0e-4)), "f threw")
    end

    @testset "a Threads.@threads loop on one thread" begin
        sols = Vector{Any}(undef, 2)
        Threads.@threads for i in 1:2
            sols[i] = solve(decay_problem(rows), TSRK("5dp"; comm))
        end
        ref = solve(decay_problem(rows), TSRK("5dp"; comm))
        @test all(s -> s.retcode == ReturnCode.Success && s.u == ref.u, sols)
    end

    @testset "refusals" begin
        prob = decay_problem(rows)
        n = length(rows)
        for (alg, what) in (
                (TSMPRK(Int[]; comm), "`slow` names no index on any rank"),
                (TSMPRK(collect(1:n); comm), "leaving nothing fast"),
                (TSMPRK([1], Int[], "2a23"; comm), "needs a `medium` index on some rank"),
                (TSGeneric("beuler"; explicit = true, comm), "cannot run"),
            )
            @test refused(() -> solve(prob, alg; dt = 0.1), what)
        end
        for (slow, medium, sub, what) in (
                ([n + 1], Int[], "p2", "names index"),
                ([0], Int[], "p2", "indices start at 1"),
                ([1, 1], Int[], "p2", "repeats an index"),
                ([1], [1], "2a23", "share an index"),
                ([1], [2], "p2", "takes only two splits"),
            )
            e = caught() do
                mine = rank == thrower ? (slow, medium) : ([1], Int[])
                solve(prob, TSMPRK(mine..., sub; comm); dt = 0.1)
            end
            @test rank == thrower ? e isa ArgumentError && occursin(what, e.msg) : remote(e)
        end
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
        @test refused(
            () -> solve(prob, TSRK("5dp", ["-ts_type", "beuler"]; comm); dt = 0.1),
            "`beuler` cannot run",
        )
        bad = fill(1.0e-8, n + (rank == thrower))
        e = caught(() -> solve(prob, TSRK("5dp"; comm); abstol = bad))
        @test rank == thrower ? e isa ArgumentError && occursin("abstol", e.msg) : remote(e)
        e = caught(() -> solve(prob, TSRK("5dp"; comm); save_idxs = [rank == thrower ? n + 1 : 1]))
        @test rank == thrower ? e isa ArgumentError && occursin("save_idxs", e.msg) : remote(e)
        integ = init(prob, TSRK("5dp"; comm))
        e = caught(() -> integ.opts.abstol = bad)
        @test rank == thrower ? e isa ArgumentError && occursin("abstol", e.msg) : remote(e)
        terminate!(integ)
    end

    @testset "every handle is freed" begin
        @test isempty(PETScDiffEq.PARALLEL_HANDLES)
        @test all(h -> h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
    end
end
