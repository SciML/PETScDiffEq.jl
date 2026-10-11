using MPI, PETScDiffEq, SciMLBase, SparseArrays, Test
using PETScDiffEq: PETSc, LibPETSc, PETScCompat, AutoFiniteDiff, AutoForwardDiff
using SciMLBase: SecondOrderODEProblem, DynamicalODEProblem, DynamicalODEFunction, ReturnCode,
    DiscreteCallback, ContinuousCallback, solve, step!, solve!, set_u!, reinit!, get_du

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
# PetscInitialize is collective, and the serial reference solves run on single ranks.
PETSc.initialize(PETSc.getlib(; PetscScalar = Float64))
const pl = PETSc.getlib(; PetscScalar = Float64)

const N = 23
const thrower = nranks - 1
# Measured at 1 to 3 ranks against a serial solve: 3.1e-15 at fixed steps, 1.2e-9 in the state
# and 4.9e-10 in the times at adaptive ones, and 2.5e-6 and 2.7e-7 under alpha2's colouring.
const EXACT = (u = 1.0e-13, t = 0.0)
const STEPPED = (u = 5.0e-9, t = 2.0e-9)
const COLOURED = (u = 1.0e-5, t = 1.0e-6)
const WAVE_ERR = Dict("sieuler" => 0.06, "velverlet" => 4.0e-3, "3" => 2.0e-5, "4" => 2.0e-5)
const ENERGY_ERR = Dict("velverlet" => 5.0e-7, "4" => 5.0e-11)
const ALPHA2_ERR = Dict("fixed" => 3.0e-3, "adaptive" => 1.0e-5)
const BDF_ERR = 1.5e-4
const SPRING_ERR = 5.0e-5

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
# Each part gathers on its own, into the serial [v; u].
function gathered_state(x, counts)
    v, u = gathered(x.x[1], counts), gathered(x.x[2], counts)
    return rank == 0 ? vcat(v, u) : nothing
end
gathered_sol(sol, counts) = [gathered_state(x, counts) for x in sol.u]

same_everywhere(x) = MPI.bcast(x, 0, comm) == x
everywhere(b) = MPI.Allreduce(b, &, comm)
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

parallel(alg) = alg.comm != MPI.COMM_SELF

const dx = 1 / (N + 1)
const C2 = 0.25
wave0(idx) = sinpi.(idx .* dx) .+ 0.5 .* sinpi.(3 .* idx .* dx)
omega(k) = sqrt(C2 * 4 / dx^2) * sinpi(k * dx / 2)
mode(k, t) = cos(omega(k) * t) .* sinpi.(k .* (1:N) .* dx)
speed(k, t) = -omega(k) * sin(omega(k) * t) .* sinpi.(k .* (1:N) .* dx)
wave_exact(t) = vcat(speed(1, t) .+ 0.5 .* speed(3, t), mode(1, t) .+ 0.5 .* mode(3, t))

function laplacian!(out, u, left, right)
    n = length(u)
    for i in 1:n
        l = i == 1 ? left : u[i - 1]
        r = i == n ? right : u[i + 1]
        out[i] = (l - 2u[i] + r) / dx^2
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

wave!(a, v, u, p, t) = (laplacian!(a, u, halo(u)...); a .*= C2; nothing)
wave_serial!(a, v, u, p, t) = (laplacian!(a, u, 0.0, 0.0); a .*= C2; nothing)
velocity!(du, v, u, p, t) = (du .= v; nothing)

neighbours(i) = max(1, i - 1):min(N, i + 1)

# This rank's v rows and then its u rows of the first-order system, with the columns of [v; u].
function wave_proto(idx)
    n = length(idx)
    I = vcat([k for (k, i) in enumerate(idx) for _ in neighbours(i)], n .+ (1:n))
    J = vcat([N + j for i in idx for j in neighbours(i)], collect(idx))
    return sparse(I, J, ones(length(I)), 2n, 2N)
end

wave_jac(idx) = function (J, x, p, t)
    n = length(idx)
    for (k, i) in enumerate(idx)
        for j in neighbours(i)
            J[k, N + j] = C2 * (i == j ? -2 : 1) / dx^2
        end
        J[n + k, i] = 1.0
    end
    return nothing
end

function wave_problem(idx, alg; span = (0.0, 1.0), jac = false, proto = jac, f = nothing)
    f = something(f, parallel(alg) ? wave! : wave_serial!)
    fn = if jac
        DynamicalODEFunction{true}(f, velocity!; jac = wave_jac(idx), jac_prototype = wave_proto(idx))
    elseif proto
        DynamicalODEFunction{true}(f, velocity!; jac_prototype = wave_proto(idx))
    else
        DynamicalODEFunction{true}(f, velocity!)
    end
    return SecondOrderODEProblem(fn, zeros(length(idx)), wave0(idx), span)
end

# A chain of unequal masses on nonlinear springs, with v the momentum.
const BETA = 0.5
masses(idx) = 1 .+ 0.5 .* sinpi.(idx ./ 7)
function chain_force!(dp, q, left, right)
    n = length(q)
    for i in 1:n
        l = i == 1 ? left : q[i - 1]
        r = i == n ? right : q[i + 1]
        a, b = r - q[i], q[i] - l
        dp[i] = a - b + BETA * (a^3 - b^3)
    end
    return nothing
end
kick!(dp, p, q, m, t) = chain_force!(dp, q, halo(q)...)
kick_serial!(dp, p, q, m, t) = chain_force!(dp, q, 0.0, 0.0)
drift!(dq, p, q, m, t) = (dq .= p ./ m; nothing)
chain_problem(idx, alg; span = (0.0, 50.0)) = DynamicalODEProblem(
    parallel(alg) ? kick! : kick_serial!, drift!, zeros(length(idx)),
    0.4 .* sinpi.(idx ./ (N + 1)) .+ 0.1 .* sinpi.(2 .* idx ./ (N + 1)), span, masses(idx),
)
# The kick's rows and then the drift's, in `wave_proto`'s pattern.
chain_jac(idx) = function (J, x, m, t)
    q, n = x.x[2], length(idx)
    left, right = halo(q)
    for (k, i) in enumerate(idx)
        a = (k == n ? right : q[k + 1]) - q[k]
        b = q[k] - (k == 1 ? left : q[k - 1])
        i > 1 && (J[k, N + i - 1] = 1 + 3 * BETA * b^2)
        J[k, N + i] = -2 - 3 * BETA * (a^2 + b^2)
        i < N && (J[k, N + i + 1] = 1 + 3 * BETA * a^2)
        J[n + k, i] = 1 / m[k]
    end
    return nothing
end
function chain_energy(x)
    p, q = x[1:N], x[(N + 1):end]
    d = diff(vcat(0.0, q, 0.0))
    return sum(p .^ 2 ./ (2 .* masses(1:N))) + sum(d .^ 2 ./ 2 .+ BETA .* d .^ 4 ./ 4)
end

# A direct linear solve on both sides, so an implicit solve can agree to round-off.
direct(c) = ["-ksp_type", "preonly", "-pc_type", c == MPI.COMM_SELF ? "lu" : "redundant"]

# Runs `run(idx, alg)` on the communicator and, on rank 0, on its own.
function against_serial(run, alg, serial_alg; layout = counts, idx = rows)
    sol = run(idx, alg)
    ref = rank == 0 ? run(1:N, serial_alg) : nothing
    return sol, ref, gathered_sol(sol, layout)
end

function matches(sol, ref, us, gap)
    n = length(sol.t)
    MPI.Allreduce(n, min, comm) == MPI.Allreduce(n, max, comm) || return false
    same = same_everywhere(sol.t)
    rank == 0 || return same
    length(sol.t) == length(ref.t) || return false
    times = maximum(abs, sol.t - ref.t) <= gap.t
    return same && times && sol.retcode == ref.retcode && maxdiff(us, collect.(ref.u)) <= gap.u
end

@testset "MPI second-order, $nranks ranks" begin
    @testset "TSBasicSymplectic on a wave: $sub" for sub in ("sieuler", "velverlet", "3", "4")
        alg, serial_alg = TSBasicSymplectic(sub; comm), TSBasicSymplectic(sub)
        sol, ref, us = against_serial(alg, serial_alg) do idx, a
            solve(wave_problem(idx, a), a; dt = 0.02)
        end
        @test sol.retcode == ReturnCode.Success
        @test sol.u[end] isa typeof(wave_problem(rows, alg).u0)
        @test matches(sol, ref, us, EXACT)
        if rank == 0
            @test (sol.stats.nf, sol.stats.nf2) == (ref.stats.nf, ref.stats.nf2)
            @test maximum(abs, us[end] - wave_exact(1.0)) <= WAVE_ERR[sub]
        end
    end

    @testset "a particle chain keeps its energy: $sub" for sub in ("velverlet", "4")
        alg, serial_alg = TSBasicSymplectic(sub; comm), TSBasicSymplectic(sub)
        sol, ref, us = against_serial(alg, serial_alg) do idx, a
            solve(chain_problem(idx, a), a; dt = 0.05)
        end
        @test matches(sol, ref, us, EXACT)
        if rank == 0
            energy = chain_energy.(us)
            drift = abs.(energy .- energy[1])
            half = length(drift) ÷ 2
            @test maximum(drift) <= ENERGY_ERR[sub]
            @test maximum(drift[half:end]) <= 1.5 * maximum(drift[1:half])
        end
    end

    @testset "TSAlpha2 with a jac and with colouring" begin
        steps = (
            ("fixed", (; dt = 0.02, adaptive = false)),
            ("adaptive", (; abstol = 1.0e-6, reltol = 1.0e-6)),
        )
        for (how, kw) in steps, span in ((0.0, 1.0), (1.0, 0.0)), jac in (true, false)
            alg = TSAlpha2(direct(comm); comm)
            serial_alg = TSAlpha2(direct(MPI.COMM_SELF); autodiff = AutoFiniteDiff())
            sol, ref, us = against_serial(alg, serial_alg) do idx, a
                SciMLBase.__solve(wave_problem(idx, a; span, jac, proto = true), a; kw...)
            end
            @test sol.retcode == ReturnCode.Success
            @test (sol.stats.njacs > 0) == jac
            @test matches(sol, ref, us, !jac ? COLOURED : how == "fixed" ? EXACT : STEPPED)
            exact = wave_exact(span[2] - span[1])
            rank == 0 && @test maximum(abs, us[end] - exact) <= ALPHA2_ERR[how]
        end
        sol, ref, us = against_serial(TSAlpha2(; comm, radius = 0.5), TSAlpha2(; radius = 0.5)) do idx, a
            SciMLBase.__solve(wave_problem(idx, a; jac = true), a; dt = 0.02, adaptive = false)
        end
        @test matches(sol, ref, us, STEPPED)
    end

    @testset "the first-order form" begin
        for (alg, serial_alg, kw) in (
                (TSRK("5dp"; comm), TSRK("5dp"), (; abstol = 1.0e-8, reltol = 1.0e-8)),
                (TSRK("4"; comm), TSRK("4"), (; dt = 0.01)),
            )
            sol, ref, us = against_serial(alg, serial_alg) do idx, a
                solve(wave_problem(idx, a), a; kw...)
            end
            @test sol.retcode == ReturnCode.Success
            @test matches(sol, ref, us, haskey(kw, :abstol) ? STEPPED : EXACT)
        end
        for jac in (true, false), span in ((0.0, 1.0), (1.0, 0.0))
            alg = TSImplicit("bdf", direct(comm); comm)
            serial_alg = TSImplicit("bdf", direct(MPI.COMM_SELF); autodiff = AutoFiniteDiff())
            sol, ref, us = against_serial(alg, serial_alg) do idx, a
                SciMLBase.__solve(
                    wave_problem(idx, a; span, jac, proto = true), a;
                    abstol = 1.0e-7, reltol = 1.0e-7,
                )
            end
            @test sol.retcode == ReturnCode.Success
            @test (sol.stats.njacs > 0) == jac
            @test matches(sol, ref, us, STEPPED)
            rank == 0 && @test maximum(abs, us[end] - wave_exact(span[2] - span[1])) <= BDF_ERR
        end
    end

    @testset "ForwardDiff seeds each rank's [v; u] block" begin
        forward = AutoForwardDiff()
        # This rank's rows of the Jacobian at `x`, then the solve.
        function jac_and_solve(pr, alg, x; kw...)
            integ = SciMLBase.__init(pr, alg; kw...)
            ctx = integ.h.ctx
            J = copy(ctx.J)
            ctx.jac!(J, x, ctx.p, 0.5)
            solve!(integ)
            return J, integ.sol
        end
        state(idx) = vcat(cospi.(idx .* dx), wave0(idx))
        entries(J, Jref) = maximum(abs, nonzeros(J) - nonzeros(Jref); init = 0.0)
        cases = (
            (c -> TSAlpha2(direct(comm); comm, c...), TSAlpha2, (; dt = 0.02, adaptive = false), EXACT, ALPHA2_ERR["fixed"]),
            (c -> TSAlpha2(direct(comm); comm, c...), TSAlpha2, (; abstol = 1.0e-6, reltol = 1.0e-6), STEPPED, ALPHA2_ERR["adaptive"]),
            (
                c -> TSImplicit("bdf", direct(comm); comm, c...), o -> TSImplicit("bdf", o),
                (; abstol = 1.0e-7, reltol = 1.0e-7), STEPPED, BDF_ERR,
            ),
        )
        for (make, serial, kw, gap, err) in cases, span in ((0.0, 1.0), (1.0, 0.0))
            alg = make((; autodiff = forward))
            jacs = Dict{Bool, Any}()
            sol, ref, us = against_serial(alg, serial(direct(MPI.COMM_SELF))) do idx, a
                jacs[parallel(a)], s = jac_and_solve(wave_problem(idx, a; span, proto = true), a, state(idx); kw...)
                s
            end
            @test sol.retcode == ReturnCode.Success
            @test sol.stats.njacs > 0
            @test matches(sol, ref, us, gap)
            if rank == 0
                @test maximum(abs, us[end] - wave_exact(span[2] - span[1])) <= err
                # The serial ForwardDiff Jacobian holds this rank's rows among its own.
                @test entries(jacs[true], jacs[false][vcat(rows, N .+ rows), :]) <= 1.0e-12
            end
            Jref, given = jac_and_solve(wave_problem(rows, alg; span, jac = true), make((;)), state(rows); kw...)
            @test entries(jacs[true], Jref) <= 1.0e-12
            @test sol.t == given.t
            @test sol.stats.njacs == given.stats.njacs
            @test maxdiff(sol.u, given.u) <= EXACT.u
        end

        # A DynamicalODEProblem, whose kick is nonlinear and whose drift divides by the masses.
        base = chain_problem(rows, TSRK(; comm); span = (0.0, 1.0))
        chain(; kw...) = DynamicalODEProblem(
            DynamicalODEFunction{true}(kick!, drift!; jac_prototype = wave_proto(rows), kw...),
            base.u0.x..., base.tspan, base.p,
        )
        x = vcat(0.3 .* cospi.(rows ./ 5), 0.4 .* sinpi.(rows ./ 9))
        fixed = (; dt = 0.05, adaptive = false)
        J, sol = jac_and_solve(chain(), TSImplicit("bdf", direct(comm); comm, autodiff = forward), x; fixed...)
        Jref, given = jac_and_solve(chain(; jac = chain_jac(rows)), TSImplicit("bdf", direct(comm); comm), x; fixed...)
        @test entries(J, Jref) <= 1.0e-12
        @test sol.retcode == given.retcode == ReturnCode.Success
        @test sol.t == given.t
        @test sol.stats.njacs == given.stats.njacs > 0
        @test maxdiff(sol.u, given.u) <= 1.0e-10
    end

    @testset "callbacks and the integrator: $(nameof(typeof(alg)))" for (alg, serial_alg, kw) in (
            (TSBasicSymplectic(; comm), TSBasicSymplectic(), (; dt = 0.02)),
            (
                TSAlpha2(direct(comm); comm),
                TSAlpha2(direct(MPI.COMM_SELF); autodiff = AutoFiniteDiff()),
                (; dt = 0.02, adaptive = false),
            ),
            (TSRK("5dp"; comm), TSRK("5dp"), (; abstol = 1.0e-8, reltol = 1.0e-8)),
        )
        gap = alg isa TSAlpha2 ? COLOURED : haskey(kw, :abstol) ? STEPPED : EXACT
        halve = DiscreteCallback((u, t, i) -> t == 0.5, i -> (i.u.x[1] .*= 0.5))
        sol, ref, us = against_serial(alg, serial_alg) do idx, a
            pr = wave_problem(idx, a; proto = true)
            SciMLBase.__solve(pr, a; kw..., callback = halve, tstops = [0.5])
        end
        @test 0.5 in sol.t
        @test matches(sol, ref, us, gap)

        # One rank's entry and the others' zeros add up exactly.
        events = Ref(0)
        sol, ref, us = against_serial(alg, serial_alg) do idx, a
            j = findfirst(==(12), idx)
            own(u) = j === nothing ? 0.0 : u.x[2][j]
            middle(u) = parallel(a) ? MPI.Allreduce(own(u), +, comm) : own(u)
            flip = ContinuousCallback(
                (u, t, i) -> middle(u) - 0.3, i -> (parallel(a) && (events[] += 1); i.u.x[1] .*= -1),
            )
            SciMLBase.__solve(wave_problem(idx, a; proto = true), a; kw..., callback = flip)
        end
        @test events[] > 0
        @test matches(sol, ref, us, gap === EXACT ? STEPPED : gap)

        slopes = Dict{Bool, Any}()
        sol, ref, us = against_serial(alg, serial_alg) do idx, a
            integ = SciMLBase.__init(wave_problem(idx, a; proto = true), a; kw...)
            for _ in 1:10
                step!(integ)
            end
            slopes[parallel(a)] = get_du(integ)
            set_u!(integ, 0.5 .* integ.u)
            step!(integ, 0.3, true)
            reinit!(integ, vcat(zeros(length(idx)), 2 .* wave0(idx)))
            solve!(integ)
            integ.sol
        end
        @test matches(sol, ref, us, gap)
        du = gathered_state(slopes[true], counts)
        rank == 0 && @test maximum(abs, du - collect(slopes[false])) <= gap.u
    end

    @testset "unstable_check gets this rank's state: $(nameof(typeof(alg)))" for alg in (
            TSBasicSymplectic(; comm), TSAlpha2(direct(comm); comm), TSRK("4"; comm),
        )
        prob = wave_problem(rows, alg; proto = true)
        seen = []
        low = (dt, u, p, t) -> (push!(seen, copy(u)); rank == thrower && u.x[2][end] < 0)
        sol = SciMLBase.__solve(prob, alg; dt = 0.02, adaptive = false, unstable_check = low)
        @test sol.retcode == ReturnCode.Unstable
        @test same_everywhere(sol.t)
        @test 0.3 < sol.t[end] < 0.6
        @test all(u -> u isa typeof(prob.u0), seen)
        @test seen == sol.u[2:end]
    end

    @testset "a rank with no rows" begin
        w = [r == 0 && nranks > 1 ? 0 : r + 1 for r in 0:(nranks - 1)]
        layout = floor.(Int, N .* w ./ sum(w))
        layout[end] += N - sum(layout)
        idx = owned(layout)
        stiffness(idx) = 1 .+ idx ./ N
        spring!(a, v, u, p, t) = (a .= -p .* u; nothing)
        spring_jac(idx) = function (J, x, p, t)
            n = length(idx)
            for (r, i) in enumerate(idx)
                J[r, N + i] = -p[r]
                J[n + r, i] = 1.0
            end
            return nothing
        end
        function springs(idx; jac = true)
            n = length(idx)
            proto = sparse([1:n; n .+ (1:n)], [N .+ idx; idx], ones(2n), 2n, 2N)
            fn = jac ?
                DynamicalODEFunction{true}(spring!, velocity!; jac = spring_jac(idx), jac_prototype = proto) :
                DynamicalODEFunction{true}(spring!, velocity!; jac_prototype = proto)
            return SecondOrderODEProblem(fn, zeros(n), ones(n), (0.0, 1.0), stiffness(idx))
        end
        root = sqrt.(stiffness(1:N))
        exact = vcat(-root .* sin.(root), cos.(root))
        for (alg, serial_alg, kw) in (
                (TSBasicSymplectic("4"; comm), TSBasicSymplectic("4"), (; dt = 0.01)),
                (
                    TSAlpha2(direct(comm); comm), TSAlpha2(direct(MPI.COMM_SELF)),
                    (; dt = 0.01, adaptive = false),
                ),
                (TSRK("5dp"; comm), TSRK("5dp"), (; abstol = 1.0e-10, reltol = 1.0e-10)),
            )
            sol, ref, us = against_serial(alg, serial_alg; layout, idx) do i, a
                SciMLBase.__solve(springs(i), a; kw...)
            end
            @test sol.retcode == ReturnCode.Success
            @test matches(sol, ref, us, haskey(kw, :abstol) ? STEPPED : EXACT)
            rank == 0 && @test maximum(abs, us[end] - exact) <= SPRING_ERR
        end
        forward = TSAlpha2(direct(comm); comm, autodiff = AutoForwardDiff())
        sol, ref, us = against_serial(forward, TSAlpha2(direct(MPI.COMM_SELF)); layout, idx) do i, a
            SciMLBase.__solve(springs(i; jac = false), a; dt = 0.01, adaptive = false)
        end
        @test sol.retcode == ReturnCode.Success
        @test sol.stats.njacs > 0
        @test matches(sol, ref, us, EXACT)
    end

    @testset "f1, f2 or jac throwing on one rank raises on every rank" begin
        late(f, what) = function (out, args...)
            f(out, args...)
            rank == thrower && args[end] > 0.2 && error("$what threw on rank $rank")
            return nothing
        end
        # A drift that communicates, which the throwing rank has to reach too.
        exchanging!(du, v, u, p, t) = (MPI.Allreduce(1, +, comm); du .= v; nothing)
        n = length(rows)
        thrown(f1, f2, alg; kw...) = caught(
            () -> SciMLBase.__solve(
                DynamicalODEProblem(f1, f2, zeros(n), wave0(rows), (0.0, 1.0)), alg; kw...,
            ),
        )
        fixed = (; dt = 0.02, adaptive = false)
        forward = AutoForwardDiff()
        for alg in (TSBasicSymplectic(; comm), TSRK("5dp"; comm))
            @test raised(thrown(late(wave!, "f1"), exchanging!, alg; dt = 0.02), "f1 threw")
            @test raised(thrown(wave!, late(exchanging!, "f2"), alg; dt = 0.02), "f2 threw")
        end
        for kw in (fixed, (; abstol = 1.0e-6, reltol = 1.0e-6))
            alg = TSAlpha2(; comm)
            e = caught(() -> SciMLBase.__solve(wave_problem(rows, alg; proto = true, f = late(wave!, "f1")), alg; kw...))
            @test raised(e, "f1 threw")
            for ad_alg in (TSAlpha2(; comm, autodiff = forward), TSImplicit("bdf"; comm, autodiff = forward))
                pr = wave_problem(rows, alg; proto = true, f = late(wave!, "f1"))
                @test raised(caught(() -> SciMLBase.__solve(pr, ad_alg; kw...)), "f1 threw")
            end
            fn = DynamicalODEFunction{true}(
                wave!, velocity!; jac = late(wave_jac(rows), "jac"), jac_prototype = wave_proto(rows),
            )
            pr = SecondOrderODEProblem(fn, zeros(n), wave0(rows), (0.0, 1.0))
            @test raised(caught(() -> SciMLBase.__solve(pr, alg; kw...)), "jac threw")
            @test raised(caught(() -> SciMLBase.__solve(pr, TSImplicit("bdf"; comm); kw...)), "jac threw")
        end
        @test SciMLBase.__solve(wave_problem(rows, TSAlpha2(; comm); jac = true), TSAlpha2(; comm); fixed...).retcode ==
            ReturnCode.Success
    end

    @testset "refusals" begin
        n = length(rows)
        alg = TSAlpha2(; comm)
        forward = AutoForwardDiff()
        zygote = PETScDiffEq.ADTypes.AutoZygote()
        for make in (c -> TSAlpha2(; comm, c...), c -> TSImplicit("bdf"; comm, c...))
            @test refused(
                () -> SciMLBase.__solve(wave_problem(rows, alg), make((; autodiff = forward)); dt = 0.02),
                "needs a sparse `jac_prototype`",
            )
            @test refused(
                () -> SciMLBase.__solve(wave_problem(rows, alg; proto = true), make((; autodiff = zygote)); dt = 0.02),
                "AutoZygote()` on a communicator",
            )
        end
        dense_jac = SecondOrderODEProblem(
            DynamicalODEFunction{true}(wave!, velocity!; jac = wave_jac(rows)), zeros(n), wave0(rows),
            (0.0, 1.0),
        )
        @test refused(() -> solve(dense_jac, alg; dt = 0.02), "needs a sparse `jac_prototype`")
        narrow = SecondOrderODEProblem(
            DynamicalODEFunction{true}(wave!, velocity!; jac_prototype = sparse(ones(2n, N))),
            zeros(n), wave0(rows), (0.0, 1.0),
        )
        @test refused(() -> SciMLBase.__solve(narrow, alg; dt = 0.02), "must be $(2n) x $(2N)")
        chain = chain_problem(rows, alg)
        @test refused(() -> solve(chain, alg; dt = 0.02), "TSAlpha2 needs a SecondOrderODEProblem")
        uneven_parts = SecondOrderODEProblem(
            wave!, zeros(rank == thrower ? n + 1 : n), wave0(rows), (0.0, 1.0),
        )
        e = caught(() -> solve(uneven_parts, alg; dt = 0.02))
        @test rank == thrower ? e isa ArgumentError && occursin("same length", e.msg) : remote(e)
        @test refused(
            () -> PETScDiffEq._discrete_adjoint(
                chain, TSRK("4"; comm), PETScAdjoint(); t = [1.0],
                dgdu_discrete = (out, u, p, t, i) -> (out .= u; nothing), dt = 0.05, adaptive = false,
            ),
            "without a `jac`, PETScAdjoint",
        )
        da = PETSc.DMDA(
            pl, comm, (LibPETSc.DM_BOUNDARY_NONE,), (2N,), 1, 1;
            points_per_proc = (LibPETSc.PetscInt.(2 .* counts),),
        )
        @test refused(() -> solve(chain, TSRK("5dp"; dm = da)), "with a `dm` yet")
        PETScCompat.destroy!(da)
    end

    @testset "every handle is freed" begin
        @test isempty(PETScDiffEq.PARALLEL_HANDLES)
        @test all(h -> h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
    end
end
