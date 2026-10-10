using PETScDiffEq
using SciMLBase
using DiffEqBase: DiffEqBase
using LinearAlgebra
using Logging
using SparseArrays
using DiffEqCallbacks
using DiffEqCallbacks: PresetTimeCallback
using MPI
using OrdinaryDiffEqTsit5: Tsit5
using RecipesBase: RecipesBase
using Test

RecipesBase.is_key_supported(::Symbol) = true

decay!(du, u, p, t) = (
    @inbounds for i in eachindex(du)
        du[i] = -u[i]
    end; nothing
)

function lotka_volterra!(du, u, p, t)
    a, b, c, d = p
    du[1] = a * u[1] - b * u[1] * u[2]
    return du[2] = -c * u[2] + d * u[1] * u[2]
end

decay_jac!(J, u, p, t) = (J[1, 1] = -1.0; nothing)
decay_wrong_jac!(J, u, p, t) = (J[1, 1] = 100.0; nothing)

function lotka_volterra_jac!(J, u, p, t)
    a, b, c, d = p
    J[1, 1] = a - b * u[2]
    J[1, 2] = -b * u[1]
    J[2, 1] = d * u[2]
    return J[2, 2] = -c + d * u[1]
end

function chain!(du, u, p, t)
    du[1] = -u[1] + u[2]
    du[2] = u[1] - 2u[2] + u[3]
    return du[3] = u[2] - u[3]
end
function chain_jac!(J, u, p, t)
    J[1, 1] = -1.0
    J[2, 1] = 1.0
    J[1, 2] = 1.0
    J[2, 2] = -2.0
    J[3, 2] = 1.0
    J[2, 3] = 1.0
    return J[3, 3] = -1.0
end
const CHAIN_PROTOTYPE = sparse(
    [1, 2, 1, 2, 3, 2, 3], [1, 1, 2, 2, 2, 3, 3], ones(7), 3, 3,
)

function damped_oscillator!(du, u, p, t)
    du[1] = u[2]
    return du[2] = -u[1] - 0.5 * u[2]
end
function damped_oscillator_jac!(J, u, p, t)
    J[1, 2] = 1.0
    J[2, 1] = -1.0
    return J[2, 2] = -0.5
end
const OSCILLATOR_PROTOTYPE = sparse([1, 2, 2], [2, 1, 2], ones(3), 2, 2)

# GROUP=MPI2 is MPI on 2 ranks, for a workflow that passes only GROUP.
const GROUP, GROUP_RANKS = let group = get(ENV, "GROUP", ""), ranks = match(r"^MPI([123])$", group)
    isempty(group) ? ("All", "") : ranks === nothing ? (group, "") : ("MPI", String(ranks[1]))
end
GROUP in ("All", "Core", "MPI") ||
    error("GROUP is All, Core, MPI, or MPI1, MPI2 or MPI3 for one rank count, not $GROUP")
is_mpi(s) = Meta.isexpr(s, :macrocall) && s.args[1] === Symbol("@testset") && s.args[3] == "MPI"

# TEST_PART=k/n runs every n-th top-level testset from the k-th on, so n processes cover the file.
const PART = let part = get(ENV, "TEST_PART", "")
    isempty(part) ? (1, 1) : Tuple(parse.(Int, split(part, '/')))
end
length(PART) == 2 && 1 <= PART[1] <= PART[2] ||
    error("TEST_PART is k/n with 1 <= k <= n, not $(ENV["TEST_PART"])")
is_testset(s) = Meta.isexpr(s, :&&) ? is_testset(s.args[end]) :
    Meta.isexpr(s, :macrocall) && s.args[1] === Symbol("@testset")

# Julia compiles a block as one thunk before running any of it, so each testset stands alone.
macro each_toplevel(ts, block)
    stmts = filter(s -> GROUP == "All" || (GROUP == "MPI") == is_mpi(s), block.args)
    tests = findall(is_testset, stmts)
    deleteat!(stmts, setdiff(tests, tests[PART[1]:PART[2]:end]))
    isdefined(Test, :push_testset) &&
        return esc(Expr(:toplevel, :(Test.push_testset($ts)), stmts..., :(Test.pop_testset())))
    wrap(s) = Meta.isexpr(s, :macrocall) && s.args[1] === Symbol("@testset") ? :(Test.@with_testset $ts $s) : s
    return esc(Expr(:toplevel, map(wrap, stmts)...))
end

const ALL_TESTS = Test.DefaultTestSet("PETScDiffEq.jl")
@each_toplevel ALL_TESTS begin
    @testset "TSRK convergence order" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        exact = exp(-1.0)
        for (subtype, expected_order) in (("3bs", 3), ("5dp", 5))
            errs = Float64[]
            for dt in (0.1, 0.05, 0.025, 0.0125)
                sol = SciMLBase.solve(
                    prob, PETScDiffEq.TSRK(subtype, ["-ts_adapt_type", "none"]); dt = dt,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                push!(errs, abs(sol.u[end][1] - exact))
            end
            orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
            @test all(o -> isapprox(o, expected_order; atol = 0.15), orders)
        end
    end

    @testset "TSRosW convergence order" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        exact = exp(-1.0)
        errs = Float64[]
        for dt in (0.1, 0.05, 0.025, 0.0125)
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRosW("ra34pw2", ["-ts_adapt_type", "none"]); dt = dt,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            push!(errs, abs(sol.u[end][1] - exact))
        end
        orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
        @test all(o -> isapprox(o, 3; atol = 0.15), orders)
    end

    @testset "TSImplicit convergence order" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        exact = exp(-1.0)
        cases = (
            ("beuler", PETScDiffEq.TSImplicit("beuler", ["-ts_adapt_type", "none"]), 1),
            ("cn", PETScDiffEq.TSImplicit("cn", ["-ts_adapt_type", "none"]), 2),
            (
                "theta(0.5)",
                PETScDiffEq.TSImplicit("theta", 0.5, ["-ts_adapt_type", "none"]), 2,
            ),
            ("bdf", PETScDiffEq.TSImplicit("bdf", ["-ts_adapt_type", "none"]), 2),
        )
        for (label, alg, expected_order) in cases
            errs = Float64[]
            for dt in (0.1, 0.05, 0.025, 0.0125)
                sol = SciMLBase.solve(prob, alg; dt = dt)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                push!(errs, abs(sol.u[end][1] - exact))
            end
            orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
            @test isapprox(orders[end], expected_order; atol = 0.2)
        end
        @test PETScDiffEq.TSImplicit("cn").theta === nothing
        @test PETScDiffEq.TSImplicit("theta", 0.5).theta == 0.5
        @test PETScDiffEq.TSImplicit("bdf", ["-ts_bdf_order", "3"]).petsc_options ==
            ["-ts_bdf_order", "3"]
    end

    @testset "TSARKIMEX convergence order" begin
        exact = exp(-1.0)
        errs = Float64[]
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        for dt in (0.1, 0.05, 0.025, 0.0125)
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSARKIMEX("3", ["-ts_adapt_type", "none"]); dt = dt,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            push!(errs, abs(sol.u[end][1] - exact))
        end
        orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
        @test all(o -> isapprox(o, 3; atol = 0.15), orders)
    end

    @testset "TSARKIMEX keeps order when tspan starts away from 0" begin
        f!(du, u, p, t) = (du[1] = cos(t); nothing)
        t0 = 1.0
        prob = SciMLBase.ODEProblem(f!, [sin(t0)], (t0, t0 + 1))
        for alg in (
                PETScDiffEq.TSARKIMEX("3"),
                PETScDiffEq.TSARKIMEX("l2", ["-ts_arkimex_type", "3"]),
                PETScDiffEq.TSGeneric("arkimex", ["-ts_arkimex_type", "3"]),
            )
            errs = map((0.01, 0.005)) do dt
                sol = SciMLBase.solve(prob, alg; dt, adaptive = false)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                abs(sol.u[end][1] - sin(t0 + 1))
            end
            @test errs[1] < 1.0e-6
            @test log2(errs[1] / errs[2]) ≈ 3 atol = 0.3
        end
    end

    @testset "TSARKIMEX IMEX split" begin
        stiff!(du, u, p, t) = (du[1] = -50.0 * u[1]; nothing)
        forcing!(du, u, p, t) = (du[1] = 1.0; nothing)
        exact(t) = (1.0 - 1 / 50) * exp(-50t) + 1 / 50

        prob = SciMLBase.SplitODEProblem(stiff!, forcing!, [1.0], (0.0, 0.1))
        errs = Float64[]
        for dt in (0.01, 0.005, 0.0025, 0.00125)
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSARKIMEX("3", ["-ts_adapt_type", "none"]); dt = dt,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            push!(errs, abs(sol.u[end][1] - exact(0.1)))
        end
        orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
        @test all(o -> isapprox(o, 3; atol = 0.2), orders)
    end

    Sys.WORD_SIZE == 64 && @testset "TSARKIMEX takes its first stage at the step's start" begin
        exact(t) = 1 / (2 - sin(t))
        seen = Float64[]
        rhs!(du, u, p, t) = (push!(seen, t); du[1] = u[1]^2 * cos(t); nothing)
        jac!(J, u, p, t) = (push!(seen, t); J[1, 1] = 2 * u[1] * cos(t); nothing)
        tight = ["-snes_rtol", "1e-13", "-snes_atol", "1e-14", "-ksp_rtol", "1e-13"]
        problem(span, f = rhs!) = SciMLBase.ODEProblem(f, [exact(span[1])], span)
        err(sol) = abs(sol.u[end][1] - exact(sol.t[end]))
        fixed(prob, alg, dt; kw...) = SciMLBase.solve(prob, alg; dt, adaptive = false, kw...)
        order(prob, alg) = log2(err(fixed(prob, alg, 0.1)) / err(fixed(prob, alg, 0.05)))
        fourth = PETScDiffEq.TSARKIMEX("4", tight)

        @testset "$sub keeps its order on $span" for sub in (
                    "1bee", "a2", "l2", "2c", "2d", "2e", "prssp2", "3", "ars443", "bpr3", "4", "5",
                ), span in ((0.0, 1.0), (1.0, 2.0), (1.0, 0.0), (-1.0, 0.0))
            alg = PETScDiffEq.TSARKIMEX(sub, tight)
            @test order(problem(span), alg) ≈ SciMLBase.alg_order(alg) atol = 0.3
        end
        esdirk = PETScDiffEq.TSGeneric("dirk", [tight; "-ts_dirk_type"; "es324sal"])
        @test order(problem((1.0, 2.0)), esdirk) ≈ 3 atol = 0.3

        @testset "f and jac are called inside $span" for span in ((1.0, 2.0), (2.0, 1.0))
            empty!(seen)
            sol = fixed(problem(span, SciMLBase.ODEFunction(rhs!; jac = jac!)), fourth, 0.05)
            @test all(t -> 1 <= t <= 2, seen)
            @test err(sol) < 1.0e-7
        end

        @testset "a mass matrix, a DAEProblem and an option that picks arkimex" begin
            span = (1.0, 2.0)
            x0 = exact(1.0)
            doubled = SciMLBase.ODEFunction(
                (du, u, p, t) -> (du[1] = 2 * u[1]^2 * cos(t); nothing); mass_matrix = Diagonal([2.0]),
            )
            constrained = SciMLBase.ODEFunction(
                (du, u, p, t) -> (du[1] = u[1]^2 * cos(t); du[2] = sin(t) * u[1] - u[2]; nothing);
                mass_matrix = Diagonal([1.0, 0.0]),
            )
            residual!(r, du, u, p, t) =
                (r[1] = du[1] - u[1]^2 * cos(t); r[2] = sin(t) * u[1] - u[2]; nothing)
            picked = [tight; "-ts_type"; "arkimex"; "-ts_arkimex_type"; "4"]
            slope = [x0^2 * cos(1.0), cos(1.0) * x0 + sin(1.0) * x0^2 * cos(1.0)]
            for (prob, alg) in (
                    (SciMLBase.ODEProblem(doubled, [x0], span), fourth),
                    (SciMLBase.ODEProblem(constrained, [x0, sin(1.0) * x0], span), fourth),
                    (
                        SciMLBase.DAEProblem(residual!, slope, [x0, sin(1.0) * x0], span),
                        PETScDiffEq.TSDAE("beuler", picked),
                    ),
                    (problem(span), PETScDiffEq.TSImplicit("beuler", picked)),
                )
                @test err(fixed(prob, alg, 0.05)) < 1.0e-7
                @test err(SciMLBase.solve!(SciMLBase.init(prob, alg; dt = 0.05, adaptive = false))) < 1.0e-7
            end
        end

        @testset "after a restart away from the last stage's time" begin
            span = (0.0, 1.0)
            root = SciMLBase.ContinuousCallback(
                (u, t, integ) -> sin(7.3 * (t - 0.11)), integ -> nothing; save_positions = (false, false),
            )
            @test err(fixed(problem(span), fourth, 0.025; callback = root)) < 1.0e-7

            integ = SciMLBase.init(problem(span), fourth; dt = 0.025, adaptive = false)
            SciMLBase.step!(integ)
            SciMLBase.step!(integ)
            SciMLBase.change_t_via_interpolation!(integ, 0.04)
            @test err(SciMLBase.solve!(integ)) < 1.0e-7

            SciMLBase.reinit!(integ, [exact(1.0)]; t0 = 1.0, tf = 2.0)
            @test err(SciMLBase.solve!(integ)) < 1.0e-7

            marks = [0.2, 0.5, 0.8]
            outside(u, p, t) = !isempty(marks) && t > marks[1] && (popfirst!(marks); true)
            rejected = SciMLBase.solve(
                problem(span), PETScDiffEq.TSARKIMEX("4"); dt = 0.05, abstol = 1.0e-7,
                reltol = 1.0e-7, isoutofdomain = outside,
            )
            @test isempty(marks)
            @test err(rejected) < 5.0e-6

            jump = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = t <= 0.5 ? 1.0 : 2.0; nothing), [0.0], span,
            )
            @test fixed(jump, fourth, 0.1; d_discontinuities = [0.5]).u[end][1] ≈ 1.5 atol = 1.0e-12
        end

        @testset "a right-hand side with no value at t = 0" begin
            root! = (du, u, p, t) -> (du[1] = 1 / (2 * sqrt(t - 0.5)); nothing)
            sol = fixed(SciMLBase.ODEProblem(root!, [sqrt(0.5)], (1.0, 2.0)), fourth, 0.05)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test abs(sol.u[end][1] - sqrt(1.5)) < 1.0e-6
        end
    end

    @testset "an IMEX split whose explicit part reads u" begin
        halves(a, b) = SciMLBase.SplitODEProblem(
            (du, u, p, t) -> (du[1] = a * u[1]; nothing),
            (du, u, p, t) -> (du[1] = b * u[1]; nothing), [1.0], (0.0, 0.1),
        )
        alg = PETScDiffEq.TSARKIMEX("3", ["-ts_adapt_type", "none"])
        errs = [
            abs(SciMLBase.solve(halves(-50.0, -1.0), alg; dt = dt).u[end][1] - exp(-5.1))
                for dt in (0.01, 0.005, 0.0025, 0.00125)
        ]
        orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
        @test all(o -> isapprox(o, 3; atol = 0.25), orders)
        @test isapprox(orders[end], 3; atol = 0.15)

        bounded = PETScDiffEq.TSARKIMEX(
            "3", ["-ts_adapt_type", "none", "-ts_max_snes_failures", "1"],
        )
        @test SciMLBase.solve(halves(-2000.0, -1.0), bounded; dt = 0.01).retcode ==
            SciMLBase.ReturnCode.Success
        # Reversed, it diverges, yet PETSc may still report Success.
        reversed = SciMLBase.solve(halves(-1.0, -2000.0), bounded; dt = 0.01)
        @test reversed.retcode != SciMLBase.ReturnCode.Success || abs(reversed.u[end][1]) > 1

        @testset "a zero pivot fails the solve rather than raising" begin
            printed(f) = mktemp() do path, io
                result = redirect_stderr(f, io)
                flush(io)
                return result, read(path, String)
            end
            # Uncapped, u grows until PETSc's FD Jacobian is zero: a zero pivot.
            pivots = PETScDiffEq.TSARKIMEX(
                "3", [bounded.petsc_options; "-snes_linesearch_maxstep"; "1e300"];
                autodiff = PETScDiffEq.AutoFiniteDiff(),
            )
            prob = halves(-1.0, -2000.0)
            upto = SciMLBase.solve(
                SciMLBase.remake(prob; tspan = (0.0, 0.04)), pivots; dt = 0.01,
            )
            @test upto.retcode == SciMLBase.ReturnCode.Success

            sol, text = printed(() -> SciMLBase.solve(prob, pivots; dt = 0.01))
            @test sol.retcode == SciMLBase.ReturnCode.Failure
            @test !occursin("PETSC ERROR", text)
            @test sol.t == upto.t
            @test sol.u == upto.u
            @test sol.stats.naccept == upto.stats.naccept
            ends = SciMLBase.solve(prob, pivots; dt = 0.01, save_everystep = false)
            @test ends.retcode == SciMLBase.ReturnCode.Failure
            @test ends.t == [0.0, upto.t[end]]
            @test ends.u[end] == upto.u[end]
            @test sol.stats.nreject == upto.stats.nreject
            @test_logs (:warn, r"zero pivot") match_mode = :any SciMLBase.solve(
                prob, pivots; dt = 0.01,
            )

            n_fin = Ref(0)
            hooked = SciMLBase.DiscreteCallback(
                (u, t, integ) -> false, integ -> nothing;
                finalize = (cb, u, t, integ) -> (n_fin[] += 1; nothing),
            )
            integ = SciMLBase.init(prob, pivots; dt = 0.01, callback = hooked)
            isol, text = printed(() -> SciMLBase.solve!(integ))
            @test isol.retcode == SciMLBase.ReturnCode.Failure
            @test !occursin("PETSC ERROR", text)
            @test isol.t == upto.t
            @test isol.u == upto.u
            @test isol.stats.naccept == upto.stats.naccept
            @test n_fin[] == 1
            @test_throws ArgumentError SciMLBase.step!(integ)
            @test_logs (:warn, r"zero pivot") match_mode = :any SciMLBase.solve!(
                SciMLBase.init(prob, pivots; dt = 0.01),
            )

            index1 = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    (du, u, p, t) -> (du[1] = -u[1]; du[2] = u[1] - u[2]; nothing);
                    mass_matrix = [1.0 0.0; 0.0 0.0],
                ), [1.0, 1.0], (0.0, 0.1),
            )
            unstepped = PETScDiffEq.TSARKIMEX(
                "3", ["-ts_adapt_type", "none"]; autodiff = PETScDiffEq.AutoFiniteDiff(),
            )
            start, text = printed(() -> SciMLBase.solve(index1, unstepped; dt = 0.01))
            @test start.retcode == SciMLBase.ReturnCode.Failure
            @test start.t == [0.0]
            @test start.u == [[1.0, 1.0]]
            @test !occursin("PETSC ERROR", text)
            @test_logs (:warn, r"zero pivot") match_mode = :any SciMLBase.solve(
                index1, unstepped; dt = 0.01,
            )
            exact = SciMLBase.solve(
                index1, PETScDiffEq.TSARKIMEX("3", ["-ts_adapt_type", "none"]); dt = 0.01,
            )
            @test exact.retcode == SciMLBase.ReturnCode.Success
            @test exact.u[end] ≈ fill(exp(-0.1), 2) rtol = 1.0e-7

            plain = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
            raising = PETScDiffEq.TSImplicit(
                "beuler", ["-snes_max_it", "0", "-snes_error_if_not_converged"],
            )
            for run in (
                    () -> SciMLBase.solve(plain, raising; dt = 0.1, adaptive = false),
                    () -> SciMLBase.solve!(
                        SciMLBase.init(plain, raising; dt = 0.1, adaptive = false),
                    ),
                )
                err, text = printed() do
                    try
                        run()
                    catch e
                        e
                    end
                end
                @test err isa PETScDiffEq.LibPETSc.PetscError
                @test occursin("PETSC ERROR", text)
            end

            singular = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    (du, u, p, t) -> (du[1] = p[1] * u[1]; nothing);
                    jac = (J, u, p, t) -> (J[1, 1] = p[1]; nothing),
                    paramjac = (pJ, u, p, t) -> (pJ[1, 1] = u[1]; nothing),
                ), [1.0], (0.0, 1.0), [10.0],
            )
            zrpvt = PETScDiffEq.TSImplicit(
                "beuler",
                ["-ksp_type", "preonly", "-pc_type", "lu", "-ksp_error_if_not_converged"],
            )
            for run in (
                    () -> SciMLBase.solve(singular, zrpvt; dt = 0.1, adaptive = false),
                    () -> SciMLBase.solve!(
                        SciMLBase.init(singular, zrpvt; dt = 0.1, adaptive = false),
                    ),
                    () -> PETScDiffEq._discrete_adjoint(
                        singular, zrpvt, PETScDiffEq.PETScAdjoint();
                        t = collect(0.0:0.1:1.0),
                        dgdu_discrete = (out, u, p, t, i) -> (out .= u; nothing),
                        dt = 0.1, adaptive = false,
                    ),
                )
                err, text = printed() do
                    try
                        run()
                    catch e
                        e
                    end
                end
                @test err isa PETScDiffEq.LibPETSc.PetscError
                @test occursin("Zero pivot", text)
            end

            dense_singular = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = 8.0 * u[1]; nothing), [1.0], (0.0, 1.0),
            )
            @test SciMLBase.solve(
                dense_singular, PETScDiffEq.TSImplicit("beuler"; autodiff = PETScDiffEq.AutoFiniteDiff());
                dt = 0.125, adaptive = false,
            ).retcode == SciMLBase.ReturnCode.Failure
            for opts in (
                    ["-snes_error_if_not_converged"],
                    ["-ts_error_if_step_fails"],
                    ["-ts_error_if_step_fails", "true"],
                    ["-TS_ERROR_IF_STEP_FAILS=1"],
                )
                err = try
                    SciMLBase.solve(
                        dense_singular,
                        PETScDiffEq.TSImplicit("beuler", opts; autodiff = PETScDiffEq.AutoFiniteDiff());
                        dt = 0.125, adaptive = false,
                    )
                catch e
                    e
                end
                @test err isa PETScDiffEq.LibPETSc.PetscError
            end
        end
    end

    @testset "an integrator dropped part-way still exits cleanly" begin
        script = """
        using PETScDiffEq, SciMLBase
        f!(du, u, p, t) = (du[1] = -u[1]; nothing)
        SciMLBase.init(
            SciMLBase.ODEProblem(f!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
            dt = 0.1,
        )
        """
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
        @test success(pipeline(cmd; stdout = devnull, stderr = devnull))
    end

    @testset "a first solve runs precompiled code and loading starts nothing" begin
        script = """
        using PETScDiffEq, SciMLBase
        P = PETScDiffEq
        all(isempty, (P.CALLBACKS, P.PETSC_SYMBOLS, P.EXIT_CLEANUP_ARMED, P.POST_STEP_CTX)) &&
            all(isempty, (P.PARALLEL_HANDLES, P.LIVE_HANDLES.ht)) &&
            !any(P.PETScCompat.isinitialized, P.PETSc.petsclibs) && !P.MPI.Initialized() ||
            exit(2)
        f!(du, u, p, t) = (du[1] = -u[1]; nothing)
        SciMLBase.solve(SciMLBase.ODEProblem(f!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK())
        """
        trace = tempname()
        # Coverage turns off the native code in package images.
        flags = `--code-coverage=none --track-allocation=none --pkgimages=yes`
        cmd = `$(Base.julia_cmd()) $flags --project=$(Base.active_project())
            --trace-compile=$trace -e $script`
        @test success(pipeline(cmd; stdout = devnull, stderr = devnull))
        @test count(l -> occursin("PETScDiffEq.", l), readlines(trace)) < 10
    end

    @testset "precompile workload" begin
        made = PETScDiffEq.HANDLES_MADE[]
        withenv(PETScDiffEq._run_workload, "PMI_RANK" => "0")
        @test PETScDiffEq.HANDLES_MADE[] == made
        @test PETScDiffEq._run_workload() === nothing
    end

    @testset "several PETSc builds in one process" begin
        PETSc = PETScDiffEq.PETSc
        builds = [PETSc.getlib(; PetscScalar = S) for S in (Float64, Float32)]
        found = [PETScDiffEq._symbol(pl, :TSGetSolution) for pl in builds]
        @test allunique(found)
        @test [PETScDiffEq._symbol(pl, :TSGetSolution) for pl in builds] == found
        ptrs = [PETScDiffEq._callbacks(pl) for pl in builds]
        @test ptrs[1].rhs != ptrs[2].rhs
        @test PETScDiffEq._callbacks(builds[1]) === ptrs[1]
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dtmin = 1.0e-8)
        SciMLBase.solve!(SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dtmin = 1.0e-8))
        @test isempty(PETScDiffEq.POST_STEP_CTX)
        span(S) = (zero(real(S)), one(real(S)))
        never = (dt, u, p, t) -> false
        runs = (
            S -> SciMLBase.solve(
                SciMLBase.ODEProblem(decay!, S[1, 2], span(S)), PETScDiffEq.TSRK("5dp");
                dtmin = real(S)(1.0e-6), unstable_check = never,
            ),
            S -> SciMLBase.solve(
                SciMLBase.ODEProblem(decay!, S[1, 2], span(S)), PETScDiffEq.TSMPRK([1], "p2");
                dt = real(S)(0.01),
            ),
            S -> SciMLBase.solve(
                SciMLBase.ODEProblem((du, u, p, t) -> (du[1] = 8 * u[1]; nothing), S[1], span(S)),
                PETScDiffEq.TSImplicit("beuler"); dt = real(S)(0.125), adaptive = false,
            ),
        )
        scalars = (Float32, Float64, ComplexF32, ComplexF64)
        quietly(f) = Logging.with_logger(f, Logging.NullLogger())
        alone = Dict(S => quietly(() -> [run(S) for run in runs]) for S in scalars)
        for _ in 1:2, (k, run) in enumerate(runs), S in scalars
            path, io = mktemp()
            sol = redirect_stderr(() -> quietly(() -> run(S)), io)
            close(io)
            @test !occursin("PETSC ERROR", read(path, String))
            @test sol.retcode == alone[S][k].retcode
            @test eltype(sol.u[end]) === S
            @test sol.u == alone[S][k].u
        end
        @test all(S -> last(alone[S]).retcode == SciMLBase.ReturnCode.Failure, scalars)
        # The live integrators are deliberate: exit frees each before its build tears down.
        script = """
        using PETScDiffEq, SciMLBase
        PETSc = PETScDiffEq.PETSc
        f!(du, u, p, t) = (du[1] = -u[1]; nothing)
        prob = SciMLBase.ODEProblem(f!, [1.0], (0.0, 1.0))
        single = SciMLBase.ODEProblem(f!, Float32[1], (0.0f0, 1.0f0))
        complex = SciMLBase.ODEProblem(f!, ComplexF64[1], (0.0, 1.0))
        ok(sol) = sol.retcode == SciMLBase.ReturnCode.Success || exit(2)
        ok(SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dtmin = 1.0e-8))
        PETSc.initialize(PETSc.getlib(; PetscScalar = ComplexF32))
        ok(SciMLBase.solve(single, PETScDiffEq.TSRK("5dp"); dtmin = 1.0f-6))
        ok(SciMLBase.solve(complex, PETScDiffEq.TSRK("5dp"); dtmin = 1.0e-8))
        ok(SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dtmin = 1.0e-8))
        SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1)
        SciMLBase.init(single, PETScDiffEq.TSRK("5dp"); dt = 0.1f0)
        SciMLBase.init(complex, PETScDiffEq.TSRK("5dp"); dt = 0.1)
        """
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
        @test success(pipeline(cmd; stdout = devnull, stderr = devnull))
    end

    @testset "a threaded ensemble runs its solves one at a time" begin
        script = """
        using PETScDiffEq, SciMLBase
        f!(du, u, p, t) = (du[1] = -p[1] * u[1]; nothing)
        prob = SciMLBase.ODEProblem(f!, [1.0], (0.0, 1.0), [1.0])
        id(c) = c isa Integer ? c : c.sim_id
        ens = SciMLBase.EnsembleProblem(
            prob; prob_func = (p, c...) -> SciMLBase.remake(p; p = [1.0 + id(c[1]) / 10]),
        )
        never = SciMLBase.DiscreteCallback((u, t, i) -> false, i -> nothing)
        for kw in ((;), (; callback = never))
            sim = SciMLBase.solve(
                ens, PETScDiffEq.TSRK("5dp"), SciMLBase.EnsembleThreads();
                trajectories = 32, dt = 0.1, kw...,
            )
            all(s -> s.retcode == SciMLBase.ReturnCode.Success, sim.u) || exit(2)
        end
        """
        cmd = `$(Base.julia_cmd()) -t 4 --project=$(Base.active_project()) -e $script`
        @test success(pipeline(cmd; stdout = devnull, stderr = devnull))
    end

    @testset "stops and saved points near an event or the start" begin
        decay_long = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0e6))
        hits = Float64[]
        early = SciMLBase.solve(
            decay_long, PETScDiffEq.TSRK("5dp"); dt = 1.0e-10,
            callback = DiffEqCallbacks.PresetTimeCallback([1.0e-9], integ -> push!(hits, integ.t)),
        )
        @test hits == [1.0e-9]
        @test 1.0e-9 in early.t

        ramp = SciMLBase.ODEProblem((du, u, p, t) -> (du[1] = 1.0; nothing), [0.0], (0.0, 1.0))
        jump = SciMLBase.ContinuousCallback((u, t, integ) -> u[1] - 0.5, integ -> (integ.u[1] += 10.0))
        after = SciMLBase.solve(
            ramp, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = jump, saveat = [0.5 + 1.0e-14, 0.6],
        )
        @test issorted(after.t)
        k = findfirst(==(0.5 + 1.0e-14), after.t)
        @test after.u[k][1] ≈ 10.5
    end

    @testset "an exception from the user's code leaves no PETSc traceback" begin
        throws!(du, u, p, t) = (t > 0.3 && error("boom"); du[1] = -u[1]; nothing)
        bad = SciMLBase.ODEProblem(throws!, [1.0], (0.0, 1.0))
        runs = (
            () -> SciMLBase.solve(bad, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false),
            () -> SciMLBase.solve(
                bad, PETScDiffEq.TSRK("5dp", ["-ksp_error_if_not_converged"]);
                dt = 0.1, adaptive = false,
            ),
            () -> SciMLBase.solve!(
                SciMLBase.init(bad, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false),
            ),
        )
        for run in runs, _ in 1:2
            path, io = mktemp()
            err = redirect_stderr(io) do
                try
                    run()
                catch e
                    e
                end
            end
            close(io)
            @test err isa ErrorException && err.msg == "boom"
            @test !occursin("PETSC ERROR", read(path, String))
        end
    end

    @testset "Float32" begin
        build(integ) = PETScDiffEq.PETSc.scalartype(integ.h.petsclib)
        none = ["-ts_adapt_type", "none"]
        # PETSc's single build is left out on 32-bit x86, so Float32 runs on the double clock.
        single_build = Float32 in PETScDiffEq._loaded_builds()
        clock32 = single_build ? Float32 : Float64

        @testset "the span picks the PETSc build, and the solution keeps the problem's types" begin
            seen = Set{Any}()
            typed!(du, u, p, t) = (push!(seen, (typeof(u), typeof(t))); du .= -u; nothing)
            typed(u, p, t) = (push!(seen, (typeof(u), typeof(t))); -u)
            cases = (
                (Float32[1], (0.0f0, 1.0f0), clock32, Float32),
                (Float32[1], (0.0, 1.0), Float64, Float32),
                ([1.0], (0.0f0, 1.0f0), Float64, Float64),
                ([1.0], (0.0, 1.0), Float64, Float64),
                ([1], (0.0, 1.0), Float64, Float64),
            )
            for (u0, tspan, R, U) in cases, f in (typed!, typed)
                T = eltype(tspan)
                prob = SciMLBase.ODEProblem(f, u0, tspan)
                empty!(seen)
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = T(0.1))
                @test build(integ) === R
                @test integ.u isa Vector{R} && integ.t isa R
                sol = SciMLBase.solve!(integ)
                @test eltype(sol.u[end]) === U && eltype(sol.t) === R
                @test seen == Set([(Vector{R}, R)])
                for kw in ((;), (; saveat = T(0.25)), (; tstops = [T(0.5)]))
                    empty!(seen)
                    sol = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = T(0.1), kw...)
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test eltype(sol.u[end]) === U && eltype(sol.t) === R
                    @test seen == Set([(Vector{R}, R)])
                end
            end
            integ = SciMLBase.init(
                SciMLBase.ODEProblem(decay!, [1.0], (0.0f0, 10.0f0)), PETScDiffEq.TSRK("5dp"),
            )
            foreach(_ -> SciMLBase.step!(integ), 1:3)
            @test length(integ.sol.t) == length(integ.sol.u) == 4
            @test integ.sol(integ.t) ≈ integ.u
            whole = SciMLBase.solve(
                SciMLBase.ODEProblem(decay!, Float32[1], (0, 1)), PETScDiffEq.TSRK("5dp"),
            )
            @test eltype(whole.t) === Float64 && eltype(whole.u[end]) === Float32
            single = SciMLBase.ODEProblem(decay!, Float32[1], (0.0f0, 1.0f0))
            @test PETScDiffEq._eltypes(single, [Float64]) == (Float64, Float64, Float32)
            @test PETScDiffEq._eltypes(
                SciMLBase.remake(single; u0 = ComplexF32[1]), [Float64, ComplexF64],
            ) == (Float64, ComplexF64, ComplexF32)
            @test_throws "ArgumentError: this problem needs PETSc's Float64 complex build" (
                PETScDiffEq._petsclib(ComplexF64, [Float64])
            )
            promoted = SciMLBase.init(
                SciMLBase.ODEProblem(decay!, Float32[1], (0.0f0, 1.0f0)),
                PETScDiffEq.TSRK("5dp"); dt = 0.1,
            )
            @test build(promoted) === Float64
        end

        @testset "a Float32 state with a Float64 span is solved in Float64" begin
            ref = SciMLBase.solve(
                SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
                dt = 0.1,
            )
            oop(u, p, t) = -u
            jac32!(J, u, p, t) = (J[1, 1] = -1.0; nothing)
            residual!(r, du, u, p, t) = (r[1] = du[1] + u[1]; nothing)
            runs = (
                () -> SciMLBase.solve(
                    SciMLBase.ODEProblem(decay!, Float32[1.0], (0.0, 1.0)),
                    PETScDiffEq.TSRK("5dp"); dt = 0.1,
                ),
                () -> SciMLBase.solve(
                    SciMLBase.ODEProblem(oop, Float32[1.0], (0.0, 1.0)),
                    PETScDiffEq.TSRK("5dp"); dt = 0.1,
                ),
                () -> SciMLBase.solve!(
                    SciMLBase.init(
                        SciMLBase.ODEProblem(decay!, Float32[1.0], (0.0, 1.0)),
                        PETScDiffEq.TSRK("5dp"); dt = 0.1,
                    ),
                ),
            )
            for run in runs
                sol = run()
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.u[end] == Float32.(ref.u[end])
            end
            withjac = SciMLBase.solve(
                SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(decay!; jac = jac32!), Float32[1.0], (0.0, 1.0),
                ),
                PETScDiffEq.TSImplicit("bdf"); dt = 0.01, abstol = 1.0e-8, reltol = 1.0e-8,
            )
            @test withjac.stats.njacs > 0
            @test abs(withjac.u[end][1] - exp(-1)) < 1.0e-5
            dae = SciMLBase.solve(
                SciMLBase.DAEProblem(residual!, Float32[-1.0], Float32[1.0], (0.0, 1.0)),
                PETScDiffEq.TSDAE("bdf"); dt = 0.001, abstol = 1.0e-8, reltol = 1.0e-8,
            )
            @test abs(dae.u[end][1] - exp(-1)) < 1.0e-5
            whole = SciMLBase.solve(
                SciMLBase.ODEProblem(decay!, [1], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
                dt = 0.1,
            )
            @test whole.u[end] == ref.u[end]
        end

        @testset "the single build converges at each method's order" begin
            forced!(du, u, p, t) = (du[1] = u[1] + cos(t); nothing)
            exact = 3 * exp(3.0) / 2 + (sin(3.0) - cos(3.0)) / 2
            for (alg, steps, order) in (
                    (PETScDiffEq.TSRK("4"), (6, 8, 12), 4),
                    (PETScDiffEq.TSRK("3", none), (6, 8, 12), 3),
                    (PETScDiffEq.TSImplicit("beuler"), (30, 60, 120), 1),
                    (PETScDiffEq.TSImplicit("cn"), (12, 24, 48), 2),
                    (PETScDiffEq.TSRosW("ra34pw2", none), (6, 8, 12), 3),
                    (PETScDiffEq.TSARKIMEX("3", none), (6, 8, 12), 3),
                    (PETScDiffEq.TSImplicit("bdf", none), (12, 24, 48), 2),
                    (PETScDiffEq.TSIRK(2), (3, 4, 6), 4),
                )
                orders = map((Float32, Float64)) do F
                    prob = SciMLBase.ODEProblem(forced!, F[1], (zero(F), F(3)))
                    dts = [F(3) / n for n in steps]
                    errs = map(dts) do dt
                        sol = SciMLBase.solve(prob, alg; dt = dt)
                        @test sol.retcode == SciMLBase.ReturnCode.Success
                        @test eltype(sol.u[end]) === F
                        abs(sol.u[end][1] - exact) / exact
                    end
                    F === Float32 && @test minimum(errs) > 5.0e-5
                    [log(errs[i] / errs[i + 1]) / log(dts[i] / dts[i + 1]) for i in 1:2]
                end
                @test isapprox(orders[1][end], order; atol = 0.3)
                @test maximum(abs, orders[1] .- orders[2]) < 0.05
            end
        end

        @testset "d_discontinuities and the step limits" begin
            alg = PETScDiffEq.TSRK("5dp")
            kink!(du, u, p, t) = (du[1] = t > 0.5f0 ? -1.0f0 : 1.0f0; nothing)
            back!(du, u, p, t) = (du[1] = t < 0.5f0 ? 1.0f0 : -1.0f0; nothing)
            for (f!, tspan) in ((kink!, (0.0f0, 1.0f0)), (back!, (1.0f0, 0.0f0))),
                    adaptive in (false, true)
                prob = SciMLBase.ODEProblem(f!, Float32[0], tspan)
                dt = 0.1f0 * sign(tspan[2] - tspan[1])
                plain = SciMLBase.solve(prob, alg; dt, adaptive, tstops = [0.5f0])
                kinked = SciMLBase.solve(prob, alg; dt, adaptive, d_discontinuities = [0.5f0])
                @test abs(kinked.u[end][1]) < 1.0e-6
                @test 0.5f0 in kinked.t
                adaptive || @test abs(plain.u[end][1]) > 1.0e-3
            end
            start!(du, u, p, t) = (du[1] = t > 0 ? -1.0f0 : 1.0f0; nothing)
            begins = SciMLBase.ODEProblem(start!, Float32[0], (0.0f0, 1.0f0))
            integ = SciMLBase.init(begins, alg; dt = 0.1f0, d_discontinuities = [0.0f0])
            @test integ.t === nextfloat(zero(clock32))
            slow = SciMLBase.ODEProblem(decay!, Float32[1], (0.0f0, 1.0f0))
            forced = SciMLBase.solve(
                slow, alg; dt = 0.01f0, dtmin = 0.2f0, dtmax = 0.1f0, force_dtmin = true,
            )
            @test forced.retcode == SciMLBase.ReturnCode.Success
            @test all(≈(0.2f0), diff(forced.t))
            capped = SciMLBase.solve(slow, alg; dt = 0.01f0, dtmax = 0.05f0)
            @test maximum(diff(capped.t)) <= 0.05f0 + eps(1.0f0)
        end

        @testset "a reversed span steps as its forward mirror" begin
            grow!(du, u, p, t) = (du[1] = u[1]; nothing)
            back = SciMLBase.ODEProblem(decay!, Float32[1], (1.0f0, 0.0f0))
            fwd = SciMLBase.ODEProblem(grow!, Float32[1], (-1.0f0, 0.0f0))
            loose = (reltol = 1.0f-5, abstol = 1.0f-6)
            for (alg, kw) in (
                    (PETScDiffEq.TSRK("5dp"), loose), (PETScDiffEq.TSRosW("ra34pw2"), loose),
                    (PETScDiffEq.TSImplicit("bdf"), loose), (PETScDiffEq.TSRK("3bs"), (adaptive = false,)),
                )
                b = SciMLBase.solve(back, alg; dt = 0.1f0, kw...)
                f = SciMLBase.solve(fwd, alg; dt = 0.1f0, kw...)
                @test b.retcode == SciMLBase.ReturnCode.Success
                @test b.t[end] === zero(clock32)
                @test b.t == -f.t
                @test b.u == f.u
            end
        end

        @testset "stops, saved times and events land where they are asked for" begin
            one!(du, u, p, t) = (du[1] = 1; nothing)
            line = SciMLBase.ODEProblem(one!, Float32[0], (0.0f0, 1.0f4))
            fixed = (; dt = 0.1f0, adaptive = false)
            never = SciMLBase.DiscreteCallback((u, t, i) -> false, i -> nothing)
            for kw in (
                    (; tstops = [5000.0f0], saveat = [5000.0f0, 1.0f4]),
                    (; tstops = [1.13f0], saveat = [1.13f0]),
                    (; saveat = [1.13f0], callback = never),
                    (; saveat = Float32[9000.05, 9000.13, 9500.07, 9999.95]),
                )
                sol = SciMLBase.solve(line, PETScDiffEq.TSRK("1fe"); fixed..., kw...)
                @test sol.t == kw.saveat
                @test [u[1] for u in sol.u] == sol.t
            end
            # On i686 the double build can hit PETSc's `bad hmax` over these spans.
            if single_build
                osc!(du, u, p, t) = (du[1] = u[2]; du[2] = -u[1]; nothing)
                ring = SciMLBase.ODEProblem(osc!, Float32[0, 1], (0.0f0, 1000.0f0))
                tight = (; reltol = 1.0f-6, abstol = 1.0f-6)
                want = Float32[0.001, 0.005, 0.5, 999.995]
                final = SciMLBase.solve(ring, PETScDiffEq.TSRK("5dp"); tight...).u[end]
                for kw in ((;), (; save_start = false, save_end = false), (; tstops = [500.0f0]))
                    sol = SciMLBase.solve(ring, PETScDiffEq.TSRK("5dp"); tight..., saveat = want, kw...)
                    @test sol.t == want
                    @test maximum(k -> abs(sol.u[k][1] - sin(Float64(want[k]))), 1:3) < 2.0e-6
                    @test sol.u[end] != final
                end
                fired = Float32[]
                kick = SciMLBase.DiscreteCallback((u, t, i) -> t == 999.995f0, i -> push!(fired, i.t))
                sol = SciMLBase.solve(ring, PETScDiffEq.TSRK("5dp"); tstops = [999.995f0], callback = kick)
                @test fired == [999.995f0]
                @test 999.995f0 in sol.t
                for span in ((100.0f0, 104.0f0), (0.0f0, 100.0f0)),
                        alg in (PETScDiffEq.TSRK("3bs"), PETScDiffEq.TSRK("5dp"))
                    crossings = Ref(0)
                    cb = SciMLBase.ContinuousCallback((u, t, i) -> u[1], i -> (crossings[] += 1))
                    start = Float32[sin(span[1]), cos(span[1])]
                    SciMLBase.solve(SciMLBase.ODEProblem(osc!, start, span), alg; callback = cb)
                    @test crossings[] == count(k -> span[1] < k * pi < span[2], 1:100)
                end
            end
        end

        @testset "steps the single-precision clock can take" begin
            if single_build
                implicit = (
                    PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSRosW(), PETScDiffEq.TSARKIMEX("3"),
                )
                for t0 in (1.0f4, 1.0f5), alg in implicit
                    sol = SciMLBase.solve(SciMLBase.ODEProblem(decay!, Float32[1], (t0, t0 + 10)), alg)
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test abs(sol.u[end][1] - exp(-10)) < 0.2 * exp(-10)
                end
                for span in ((0.0f0, 1.0f6), (0.0f0, 1.0f5)), alg in implicit
                    stops = [span[2] / 7 * k for k in 1:6]
                    sol = SciMLBase.solve(
                        SciMLBase.ODEProblem(decay!, Float32[1], span), alg;
                        tstops = stops, abstol = 1.0f-5, reltol = 1.0f-4,
                    )
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test sol.t[end] == span[2]
                    @test stops ⊆ sol.t
                end
                scaled!(du, u, p, t) = (du .= p .* u; nothing)
                for (F, p) in ((Float32, -0.5f0), (ComplexF32, -0.5f0 + 2.0f0im)),
                        alg in (PETScDiffEq.TSRK("4"), PETScDiffEq.TSImplicit("cn"))
                    sol = SciMLBase.solve(
                        SciMLBase.ODEProblem(scaled!, F[1], (0.0f0, 1.0f-3), p), alg; dt = 1.0f-6,
                    )
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test sol.t[end] == 1.0f-3
                    @test sol.stats.naccept == 1000
                    @test abs(sol.u[end][1] - exp(p * 1.0e-3)) < 1.0e-4
                    stiff = SciMLBase.solve(
                        SciMLBase.ODEProblem(scaled!, F[1], (0.0f0, 1.0f-4), -3.0f6),
                        PETScDiffEq.TSRK("5dp"),
                    )
                    @test stiff.retcode == SciMLBase.ReturnCode.Success
                    @test stiff.t[end] == 1.0f-4
                end
            end
        end

        @testset "a state too small for single precision's norms is warned about" begin
            tiny = SciMLBase.ODEProblem(decay!, Float32[1.0f-22], (0.0f0, 1.0f0))
            beuler = PETScDiffEq.TSImplicit("beuler")
            if single_build
                @test_logs (:warn, r"norms underflow") SciMLBase.solve(tiny, beuler; dt = 0.01f0)
            else
                sol = @test_logs min_level = Logging.Warn SciMLBase.solve(tiny, beuler; dt = 0.01f0)
                @test sol.u[end][1] ≈ 1.0f-22 / 1.01f0^100 rtol = 1.0e-5
            end
            # Decaying into the underflow range warns only if the solve fails there.
            logs, decayed = Test.collect_test_logs(min_level = Logging.Warn) do
                SciMLBase.solve(
                    SciMLBase.ODEProblem(decay!, Float32[1], (0.0f0, 1.0f5)),
                    PETScDiffEq.TSImplicit("bdf"),
                )
            end
            @test isempty(logs) == (decayed.retcode == SciMLBase.ReturnCode.Success)
            @test_logs min_level = Logging.Warn SciMLBase.solve(
                SciMLBase.remake(tiny; u0 = Float32[1.0f-15]), beuler; dt = 0.01f0,
            )
        end

        @testset "a DAE, a mass matrix and every Jacobian match the double build" begin
            residual!(r, du, u, p, t) = (r[1] = du[1] + u[1]; r[2] = u[2] - 2u[1]; nothing)
            mm!(du, u, p, t) = (du[1] = -2u[1]; du[2] = u[2] - u[1]; nothing)
            dae(F) = SciMLBase.DAEProblem(residual!, F[-1, -2], F[1, 2], (zero(F), one(F)))
            mass(F) = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(mm!; mass_matrix = F[2 0; 0 0]), F[1, 1], (zero(F), one(F)),
            )
            chain(F; kw...) = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(chain!; kw...), F[1, 0, 0], (zero(F), one(F)),
            )
            fd = PETScDiffEq.AutoFiniteDiff()
            for (make, alg) in (
                    (dae, PETScDiffEq.TSDAE("beuler")), (dae, PETScDiffEq.TSDAE("bdf", none)),
                    (mass, PETScDiffEq.TSImplicit("bdf", none)),
                    (mass, PETScDiffEq.TSRosW("ra34pw2", none)),
                    (chain, PETScDiffEq.TSImplicit("bdf", none)),
                    (chain, PETScDiffEq.TSImplicit("bdf", none; autodiff = fd)),
                    (F -> chain(F; jac = chain_jac!), PETScDiffEq.TSRosW("ra34pw2", none)),
                    (F -> chain(F; jac_prototype = CHAIN_PROTOTYPE), PETScDiffEq.TSImplicit("bdf", none)),
                    (
                        F -> chain(F; jac_prototype = Float32.(CHAIN_PROTOTYPE)),
                        PETScDiffEq.TSRosW("ra34pw2", none; autodiff = fd),
                    ),
                )
                single = SciMLBase.solve(make(Float32), alg; dt = 0.01f0)
                double = SciMLBase.solve(make(Float64), alg; dt = 0.01)
                @test single.retcode == SciMLBase.ReturnCode.Success
                @test eltype(single.u[end]) === Float32
                @test length(single.t) == length(double.t)
                @test maximum(abs, single.u[end] .- double.u[end]) < 1.0e-5
            end
        end
    end

    @testset "complex states" begin
        n = 8
        H = SymTridiagonal(collect(range(0.5, 2.0; length = n)), fill(-1.0, n - 1))
        schr!(du, u, p, t) = (mul!(du, H, u); du .*= -im; nothing)
        # Entry by entry: broadcasting into a sparse J can change its structure.
        function schr_jac!(J, u, p, t)
            for i in 1:n
                J[i, i] = -im * H[i, i]
                i < n && (J[i, i + 1] = J[i + 1, i] = -im * H[i, i + 1])
            end
            return nothing
        end
        u0 = ComplexF64[exp(-(k - 4.5)^2 / 2) * cis(0.3k) for k in 1:n]
        exact(t) = exp(-im * Matrix(H) * t) * u0
        lu = ["-ksp_type", "preonly", "-pc_type", "lu"]
        none = ["-ts_adapt_type", "none"]
        build(integ) = PETScDiffEq.PETSc.scalartype(integ.h.petsclib)
        order(errs, dts) = [log(errs[i] / errs[i + 1]) / log(dts[i] / dts[i + 1]) for i in 1:2]
        # PETSc's single complex build is left out on 32-bit x86.
        clock32 = ComplexF32 in PETScDiffEq._loaded_builds() ? Float32 : Float64

        @testset "the state picks a complex build and keeps its type" begin
            seen = Set{Any}()
            typed!(du, u, p, t) = (push!(seen, (typeof(u), typeof(t))); schr!(du, u, p, t))
            cases = (
                (u0, (0.0, 1.0), ComplexF64, ComplexF64),
                (ComplexF32.(u0), (0.0f0, 1.0f0), Complex{clock32}, ComplexF32),
                (ComplexF32.(u0), (0.0, 1.0), ComplexF64, ComplexF32),
            )
            for (v0, tspan, S, U) in cases
                T, R = eltype(tspan), real(S)
                prob = SciMLBase.ODEProblem(typed!, v0, tspan)
                empty!(seen)
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("4"); dt = T(0.05))
                @test build(integ) === S
                @test integ.u isa Vector{S} && integ.t isa R
                sol = SciMLBase.solve!(integ)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test eltype(sol.u[end]) === U && eltype(sol.t) === R
                @test seen == Set([(Vector{S}, R)])
            end
        end

        @testset "RK4 converges at order 4" begin
            prob = SciMLBase.ODEProblem(schr!, u0, (0.0, 2.0))
            dts = [2.0 / k for k in (10, 20, 40)]
            errs = [
                maximum(abs, SciMLBase.solve(prob, PETScDiffEq.TSRK("4"); dt).u[end] - exact(2.0))
                    for dt in dts
            ]
            @test all(o -> isapprox(o, 4; atol = 0.1), order(errs, dts))
        end

        @testset "Crank-Nicolson keeps the norm and converges at order 2" begin
            prob = SciMLBase.ODEProblem(schr!, u0, (0.0, 2.0))
            dts = [2.0 / k for k in (20, 40, 80)]
            sols = [SciMLBase.solve(prob, PETScDiffEq.TSImplicit("cn", lu); dt) for dt in dts]
            for sol in sols
                @test maximum(u -> abs(norm(u) - norm(u0)), sol.u) < 1.0e-13
            end
            @test all(o -> isapprox(o, 2; atol = 0.1), order([maximum(abs, s.u[end] - exact(2.0)) for s in sols], dts))
            damped = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("beuler", lu); dt = dts[1])
            @test norm(u0) - norm(damped.u[end]) > 1.0e-2
        end

        @testset "every source of the Jacobian gives the same solve" begin
            pattern = sparse(Matrix(H) .!= 0) * 1.0
            fd = PETScDiffEq.AutoFiniteDiff()
            solve_with(f, alg) = SciMLBase.solve(SciMLBase.ODEProblem(f, u0, (0.0, 1.0)), alg; dt = 0.05)
            ref = solve_with(SciMLBase.ODEFunction(schr!; jac = schr_jac!), PETScDiffEq.TSImplicit("cn", lu))
            @test ref.stats.njacs > 0
            for (f, alg, tol) in (
                    (SciMLBase.ODEFunction(schr!), PETScDiffEq.TSImplicit("cn", lu), 1.0e-14),
                    (
                        SciMLBase.ODEFunction(schr!; jac_prototype = pattern),
                        PETScDiffEq.TSImplicit("cn", lu), 1.0e-14,
                    ),
                    (
                        SciMLBase.ODEFunction((u, p, t) -> -im .* (H * u)),
                        PETScDiffEq.TSImplicit("cn", lu), 1.0e-14,
                    ),
                    (
                        SciMLBase.ODEFunction(schr!; jac = schr_jac!, jac_prototype = complex.(pattern)),
                        PETScDiffEq.TSImplicit("cn", lu), 1.0e-14,
                    ),
                    (SciMLBase.ODEFunction(schr!), PETScDiffEq.TSImplicit("cn", lu; autodiff = fd), 1.0e-8),
                    (
                        SciMLBase.ODEFunction(schr!; jac_prototype = pattern),
                        PETScDiffEq.TSImplicit("cn", lu; autodiff = fd), 1.0e-8,
                    ),
                )
                sol = solve_with(f, alg)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test maximum(abs, sol.u[end] - ref.u[end]) < tol
            end
            for alg in (
                    PETScDiffEq.TSRosW("ra34pw2", none), PETScDiffEq.TSIRK(2),
                    PETScDiffEq.TSImplicit("bdf", none), PETScDiffEq.TSARKIMEX("3", none),
                )
                sol = solve_with(SciMLBase.ODEFunction(schr!; jac_prototype = pattern), alg)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.stats.njacs > 0
                @test maximum(abs, sol.u[end] - exact(1.0)) < 5.0e-3
            end
        end

        @testset "a sparse backend's pattern, and a prototype that leaves an entry out" begin
            pattern = sparse(Matrix(H) .!= 0) * 1.0
            known = PETScDiffEq.ADTypes.AutoSparse(
                PETScDiffEq.AutoForwardDiff();
                sparsity_detector = PETScDiffEq.ADTypes.KnownJacobianSparsityDetector(pattern),
                coloring_algorithm = PETScDiffEq.SparseMatrixColorings.GreedyColoringAlgorithm(),
            )
            cn(ad = PETScDiffEq.AutoForwardDiff()) = PETScDiffEq.TSImplicit("cn", lu; autodiff = ad)
            schr = SciMLBase.ODEProblem(schr!, u0, (0.0, 1.0))
            @test SciMLBase.solve(schr, cn(known); dt = 0.05).u ==
                SciMLBase.solve(schr, cn(); dt = 0.05).u
            residual!(r, du, u, p, t) = (mul!(r, H, u); r .= du .+ im .* r; nothing)
            dae = SciMLBase.DAEProblem(residual!, -im .* (H * u0), u0, (0.0, 1.0))
            bdf(ad = PETScDiffEq.AutoForwardDiff()) =
                PETScDiffEq.TSDAE("bdf", [none; lu]; autodiff = ad)
            @test SciMLBase.solve(dae, bdf(known); dt = 0.01).u ==
                SciMLBase.solve(dae, bdf(); dt = 0.01).u
            chainc!(du, u, p, t) = (
                for i in 1:n
                    du[i] = -2u[i] + (i > 1 ? u[i - 1] : 0) + (i < n ? u[i + 1] : 0) + p * u[i]^2
                end; nothing
            )
            upper = sparse(Bidiagonal(ones(n), ones(n - 1), :U))
            for v0 in ([0.5 + 0.2 * k / n for k in 1:n], [0.5 + 0.2im * k / n for k in 1:n])
                full = SciMLBase.solve(
                    SciMLBase.ODEProblem(chainc!, v0, (0.0, 1.0), 0.3),
                    PETScDiffEq.TSImplicit("bdf", none); dt = 0.01,
                )
                partial = SciMLBase.solve(
                    SciMLBase.ODEProblem(
                        SciMLBase.ODEFunction(chainc!; jac_prototype = upper), v0, (0.0, 1.0), 0.3,
                    ),
                    PETScDiffEq.TSImplicit("bdf", none); dt = 0.01,
                )
                @test partial.retcode == SciMLBase.ReturnCode.Success
                @test maximum(abs, partial.u[end] - full.u[end]) < 1.0e-7
            end
            conj!(du, u, p, t) = (du .= -im .* conj.(u); nothing)
            @test_throws "not holomorphic" SciMLBase.solve(
                SciMLBase.ODEProblem(SciMLBase.ODEFunction(conj!; jac_prototype = upper), u0, (0.0, 1.0)),
                PETScDiffEq.TSImplicit("cn"); dt = 0.1,
            )
        end

        @testset "a DAE, a mass matrix and a reversed span" begin
            residual!(r, du, u, p, t) = (mul!(r, H, u); r .= du .+ im .* r; nothing)
            residual_jac!(J, du, u, p, gamma, t) = (J .= gamma .* I(n) .+ im .* H; nothing)
            du0 = -im .* (H * u0)
            bdf = PETScDiffEq.TSDAE("bdf", [none; lu])
            solve_dae(f, alg) =
                SciMLBase.solve(SciMLBase.DAEProblem(f, du0, u0, (0.0, 1.0)), alg; dt = 0.01)
            ref = solve_dae(SciMLBase.DAEFunction(residual!; jac = residual_jac!), bdf)
            @test maximum(abs, ref.u[end] - exact(1.0)) < 2.0e-4
            fd = PETScDiffEq.TSDAE("bdf", [none; lu]; autodiff = PETScDiffEq.AutoFiniteDiff())
            for (f, alg, tol) in (
                    (SciMLBase.DAEFunction(residual!), bdf, 1.0e-14),
                    (
                        SciMLBase.DAEFunction(residual!; jac_prototype = sparse(Matrix(H) .!= 0) * 1.0),
                        bdf, 1.0e-14,
                    ),
                    (SciMLBase.DAEFunction(residual!), fd, 1.0e-10),
                )
                @test maximum(abs, solve_dae(f, alg).u[end] - ref.u[end]) < tol
            end
            twice!(du, u, p, t) = (mul!(du, H, u); du .*= -2im; nothing)
            for (M, expected) in ((Matrix(2.0I, n, n), exact(1.0)), (Matrix(2.0im * I, n, n), exp(-Matrix(H)) * u0))
                sol = SciMLBase.solve(
                    SciMLBase.ODEProblem(SciMLBase.ODEFunction(twice!; mass_matrix = M), u0, (0.0, 1.0)),
                    PETScDiffEq.TSImplicit("cn", lu); dt = 0.01,
                )
                @test maximum(abs, sol.u[end] - expected) < 5.0e-5
            end
            back = SciMLBase.solve(
                SciMLBase.ODEProblem(schr!, exact(1.0), (1.0, 0.0)), PETScDiffEq.TSRK("4"); dt = -0.01,
            )
            @test back.t[end] === 0.0
            @test maximum(abs, back.u[end] - u0) < 1.0e-8
        end

        @testset "ComplexF32 in the single complex build" begin
            prob = SciMLBase.ODEProblem(schr!, ComplexF32.(u0), (0.0f0, 2.0f0))
            dts = [2.0f0 / k for k in (6, 9, 12)]
            sols = [SciMLBase.solve(prob, PETScDiffEq.TSRK("4"); dt) for dt in dts]
            @test all(s -> eltype(s.u[end]) === ComplexF32, sols)
            errs = [maximum(abs, s.u[end] - exact(2.0)) for s in sols]
            @test minimum(errs) > 1.0e-4
            @test all(o -> isapprox(o, 4; atol = 0.2), order(errs, dts))
            cn = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("cn", lu); dt = 0.05f0)
            @test maximum(u -> abs(norm(u) - norm(ComplexF32.(u0))), cn.u) < 1.0e-6
        end

        @testset "a callback's condition is real" begin
            for (F, tol) in ((ComplexF64, 1.0e-9), (ComplexF32, 1.0e-6))
                R = real(F)
                roots = R[]
                cb = SciMLBase.ContinuousCallback(
                    (u, t, integ) -> real(u[4]) - R(0.1), integ -> push!(roots, integ.t),
                )
                SciMLBase.solve(
                    SciMLBase.ODEProblem(schr!, F.(u0), (zero(R), R(2))), PETScDiffEq.TSRK("5dp");
                    abstol = R(1.0e-10), reltol = R(1.0e-10), callback = cb,
                )
                @test length(roots) == 1
                @test abs(real(exact(roots[1])[4]) - 0.1) < tol
            end
        end

        @testset "what it refuses" begin
            prob = SciMLBase.ODEProblem(schr!, u0, (0.0, 1.0))
            rk = PETScDiffEq.TSRK("5dp")
            conj!(du, u, p, t) = (du .= -im .* conj.(u); nothing)
            nonlinear!(du, u, p, t) = (mul!(du, H, u); du .= -im .* (du .+ abs2.(u) .* u); nothing)
            conj_residual!(r, du, u, p, t) = (r .= du .+ im .* conj.(u); nothing)
            holomorphic = "the problem's function is not holomorphic in the complex state"
            @testset "$message" for (message, call) in (
                    (
                        "`abstol` must be real",
                        () -> SciMLBase.solve(prob, rk; abstol = 1.0e-6 + 1.0e-9im),
                    ),
                    (
                        "`reltol` must be real",
                        () -> SciMLBase.solve(prob, rk; reltol = fill(1.0e-3 + 1.0e-9im, n)),
                    ),
                    ("`dt` must be real", () -> SciMLBase.__solve(prob, rk; dt = 0.1 + 0im)),
                    ("`saveat` must be real", () -> SciMLBase.solve(prob, rk; saveat = [0.5 + 0im])),
                    ("`saveat` must be real", () -> SciMLBase.solve(prob, rk; saveat = 0.1im)),
                    ("`tstops` must be real", () -> SciMLBase.solve(prob, rk; tstops = [0.5im])),
                    (
                        "`d_discontinuities` must be real",
                        () -> SciMLBase.solve(prob, rk; d_discontinuities = [0.5 + 0im]),
                    ),
                    (
                        "`abstol` must be real",
                        () -> (SciMLBase.init(prob, rk; dt = 0.1).opts.abstol = 1.0e-6im),
                    ),
                    (
                        holomorphic,
                        () -> SciMLBase.solve(
                            SciMLBase.ODEProblem(conj!, u0, (0.0, 1.0)), PETScDiffEq.TSImplicit("cn");
                            dt = 0.1,
                        ),
                    ),
                    (
                        holomorphic,
                        () -> SciMLBase.solve(
                            SciMLBase.ODEProblem(nonlinear!, 1.0e-3 .* u0, (0.0, 1.0)),
                            PETScDiffEq.TSRosW(); dt = 0.1,
                        ),
                    ),
                    (
                        holomorphic,
                        () -> SciMLBase.solve(
                            SciMLBase.DAEProblem(conj_residual!, zero(u0), u0, (0.0, 1.0)),
                            PETScDiffEq.TSDAE("beuler"); dt = 0.1,
                        ),
                    ),
                )
                @test_throws "ArgumentError: $message" call()
            end
            @test SciMLBase.solve(
                SciMLBase.ODEProblem(nonlinear!, u0, (0.0, 1.0)), PETScDiffEq.TSRK("4"); dt = 0.01,
            ).retcode == SciMLBase.ReturnCode.Success
            real_prob = SciMLBase.ODEProblem(decay!, [1.0, 2.0], (0.0, 1.0))
            plain = SciMLBase.solve(real_prob, rk; abstol = 1.0e-6, reltol = [1.0e-3, 1.0e-3])
            zero_im = SciMLBase.solve(
                real_prob, rk; abstol = 1.0e-6 + 0im, reltol = [1.0e-3, 1.0e-3] .+ 0im,
            )
            @test zero_im.t == plain.t && zero_im.u == plain.u
            integ = SciMLBase.init(real_prob, rk; dt = 0.1)
            integ.opts.abstol = 1.0e-7 + 0im
            @test SciMLBase.solve!(integ).retcode == SciMLBase.ReturnCode.Success
        end
    end

    @testset "d_discontinuities are times to step onto" begin
        kinked = SciMLBase.ODEProblem(
            (du, u, p, t) -> (du[1] = t < 0.5 ? 1.0 : -1.0; nothing), [0.0], (0.0, 1.0),
        )
        alg = PETScDiffEq.TSRK("5dp")
        stopped = SciMLBase.solve(kinked, alg; dt = 0.1, d_discontinuities = [0.5])
        @test 0.5 in stopped.t
        @test stopped.retcode == SciMLBase.ReturnCode.Success
        both = SciMLBase.init(
            kinked, alg; dt = 0.1, tstops = [0.25], d_discontinuities = [0.5, 0.75],
        )
        @test PETScDiffEq.DiffEqBase.get_tstops_array(both) == [0.25, 0.5, 0.75, 1.0]
        @test SciMLBase.solve!(both).t ⊇ [0.25, 0.5, 0.75]
        @test_logs min_level = Logging.Warn SciMLBase.solve(
            kinked, alg; dt = 0.1, d_discontinuities = [0.5],
        )
    end

    @testset "unstable_check sees what OrdinaryDiffEq's does" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        grow = SciMLBase.ODEProblem(
            (du, u, p, t) -> (du[1] = u[1]; nothing), [1.0], (1.0, 0.0),
        )
        alg = PETScDiffEq.TSRK("5dp")
        below(cut) = (dt, u, p, t) -> any(<(cut), u)
        on_solve(pr; kw...) = SciMLBase.solve(pr, alg; kw...)
        on_integ(pr; kw...) = SciMLBase.solve!(SciMLBase.init(pr, alg; kw...))
        for run in (on_solve, on_integ)
            sol = run(prob; dt = 0.1, unstable_check = below(0.7))
            @test sol.retcode == SciMLBase.ReturnCode.Unstable
            @test 0.0 < sol.t[end] < 1.0
            @test sol.u[end][1] < 0.7
            seen = Tuple{Float64, Float64}[]
            full = run(
                prob; dt = 0.01, abstol = 1.0e-8, reltol = 1.0e-8,
                unstable_check = (dt, u, p, t) -> (push!(seen, (dt, t)); false),
            )
            @test full.stats.nreject == 0
            @test last.(seen) == full.t[2:(end - 1)]
            @test first.(seen)[1:(end - 1)] ≈ diff(full.t)[2:(end - 1)]
            @test !(first.(seen) ≈ diff(full.t)[1:(end - 1)])
            @test run(prob; dt = 0.1, unstable_check = (dt, u, p, t) -> t >= 1.0).retcode ==
                SciMLBase.ReturnCode.Success
            empty!(seen)
            back = run(
                grow; dt = 0.1, adaptive = false,
                unstable_check = (dt, u, p, t) -> (push!(seen, (dt, t)); t < 0.45),
            )
            @test back.retcode == SciMLBase.ReturnCode.Unstable
            @test back.t[end] ≈ 0.4
            @test last.(seen) ≈ back.t[2:end]
            @test all(<(0), first.(seen))
            @test_throws "check threw" run(
                prob; dt = 0.1, unstable_check = (dt, u, p, t) -> t > 0.5 && error("check threw"),
            )
        end
        halve = SciMLBase.ContinuousCallback(
            (u, t, integ) -> u[1] - 0.8, integ -> (integ.u[1] /= 2),
        )
        after = SciMLBase.solve(
            prob, alg; dt = 0.1, adaptive = false, callback = halve, unstable_check = below(0.5),
        )
        @test after.retcode == SciMLBase.ReturnCode.Unstable
        @test after.t[end] ≈ log(1 / 0.8)
        @test after.u[end][1] ≈ 0.4
        ends = SciMLBase.DiscreteCallback((u, t, integ) -> t > 0.42, SciMLBase.terminate!)
        @test SciMLBase.solve(
            prob, alg; dt = 0.1, adaptive = false, callback = ends,
            unstable_check = (dt, u, p, t) -> t > 0.42,
        ).retcode == SciMLBase.ReturnCode.Terminated
        quiet = @test_logs min_level = Logging.Warn SciMLBase.solve(
            prob, alg; dt = 0.1, unstable_check = below(-1.0),
        )
        @test quiet.retcode == SciMLBase.ReturnCode.Success
        @test quiet.t[end] == 1.0
    end

    @testset "isoutofdomain takes a step again smaller, as OrdinaryDiffEq does" begin
        below(c) = (u, p, t) -> any(<(c), u)
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp")
        for scale in (1.0, 1.0e-12)
            scaled = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = -u[1] / scale; nothing), [1.0], (0.0, scale),
            )
            for run in (
                    () -> SciMLBase.solve(scaled, alg; dt = 0.1scale, isoutofdomain = below(0.6)),
                    () -> SciMLBase.solve!(
                        SciMLBase.init(scaled, alg; dt = 0.1scale, isoutofdomain = below(0.6)),
                    ),
                )
                sol = run()
                @test sol.retcode == SciMLBase.ReturnCode.Unstable
                @test all(u -> u[1] >= 0.6, sol.u)
                @test abs(sol.t[end] / scale - log(1 / 0.6)) < 1.0e-6
            end
        end
        integ = SciMLBase.init(prob, alg; dt = 0.1, isoutofdomain = below(0.6))
        SciMLBase.solve!(integ)
        @test integ.t - integ.tprev ≈ integ.dt
        @test integ(integ.t - integ.dt / 2) ≈ integ.sol(integ.t - integ.dt / 2)

        tried = Float64[]
        once = (u, p, t) -> (push!(tried, t); length(tried) == 1)
        integ = SciMLBase.init(prob, alg; dt = 0.1, isoutofdomain = once)
        SciMLBase.step!(integ)
        @test tried ≈ [0.1, 0.02]
        @test integ.t ≈ 0.02
        @test integ.sol.stats.nreject == 1
        @test !integ.finished
        every_other = Ref(0)
        alternate = (u, p, t) -> (every_other[] += 1; isodd(every_other[]))
        capped = SciMLBase.solve(prob, alg; dt = 1.0e-3, maxiters = 10, isoutofdomain = alternate)
        @test capped.retcode == SciMLBase.ReturnCode.MaxIters
        @test every_other[] == 10
        @test capped.stats.naccept == 5 == length(capped.t) - 1
        @test capped.stats.nreject == 5

        floored = SciMLBase.solve(prob, alg; dt = 0.1, dtmin = 0.01, isoutofdomain = below(0.6))
        @test floored.retcode == SciMLBase.ReturnCode.DtLessThanMin
        @test minimum(diff(floored.t)) >= 0.01
        forced = SciMLBase.solve(
            prob, alg; dt = 0.1, dtmin = 0.01, force_dtmin = true, isoutofdomain = below(0.6),
        )
        @test forced.retcode == SciMLBase.ReturnCode.Success
        @test forced.t[end] == 1.0
        @test minimum(diff(forced.t)[1:(end - 1)]) >= 0.01

        asked = Ref(0)
        SciMLBase.solve(
            prob, alg; dt = 0.25, adaptive = false,
            isoutofdomain = (u, p, t) -> (asked[] += 1; false),
        )
        @test asked[] == 0

        rate!(du, u, p, t) = (du[1] = -p[1] * u[1]; nothing)
        events = map((false, true)) do reject
            hits = Float64[]
            undone = Ref(false)
            change_p = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t == 0.5, integ -> (integ.p[1] = 20.0; nothing),
            )
            cross = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.7, integ -> push!(hits, integ.t),
            )
            first_after = (u, p, t) -> reject && p[1] == 20.0 && !undone[] && (undone[] = true)
            SciMLBase.solve(
                SciMLBase.ODEProblem(rate!, [1.0], (0.0, 1.0), [0.0]), PETScDiffEq.TSRK("3bs");
                dt = 0.1, abstol = 1.0e-8, reltol = 1.0e-8, tstops = [0.5],
                callback = SciMLBase.CallbackSet(change_p, cross), isoutofdomain = first_after,
            )
            (hits, undone[])
        end
        @test events[2][2]
        @test abs(only(events[2][1]) - only(events[1][1])) < 1.0e-8
        @test abs(only(events[2][1]) - (0.5 + log(1 / 0.7) / 20)) < 1.0e-8
    end

    @testset "d_discontinuities and force_dtmin as OrdinaryDiffEq has them" begin
        alg = PETScDiffEq.TSRK("5dp")
        kink!(du, u, p, t) = (du[1] = t > 0.5 ? -1.0 : 1.0; nothing)
        back!(du, u, p, t) = (du[1] = t < 0.5 ? 1.0 : -1.0; nothing)
        for (f!, tspan) in ((kink!, (0.0, 1.0)), (back!, (1.0, 0.0))), adaptive in (false, true)
            prob = SciMLBase.ODEProblem(f!, [0.0], tspan)
            dt = 0.1 * sign(tspan[2] - tspan[1])
            plain = SciMLBase.solve(prob, alg; dt, adaptive, tstops = [0.5])
            kinked = SciMLBase.solve(prob, alg; dt, adaptive, d_discontinuities = [0.5])
            @test abs(kinked.u[end][1]) < 1.0e-12
            @test 0.5 in kinked.t
            adaptive || @test abs(plain.u[end][1]) > 1.0e-3
        end
        start!(du, u, p, t) = (du[1] = t > 0 ? -1.0 : 1.0; nothing)
        begins = SciMLBase.ODEProblem(start!, [0.0], (0.0, 1.0))
        integ = SciMLBase.init(begins, alg; dt = 0.1, d_discontinuities = [0.0])
        @test integ.t == nextfloat(0.0)
        @test abs(SciMLBase.solve!(integ).u[end][1] + 1) < 1.0e-12
        prob = SciMLBase.ODEProblem(kink!, [0.0], (0.0, 1.0))
        integ = SciMLBase.init(prob, alg; dt = 0.1, d_discontinuities = [0.5])
        SciMLBase.solve!(integ)
        SciMLBase.reinit!(integ; tstops = [0.25])
        @test PETScDiffEq.DiffEqBase.get_tstops_array(integ) == [0.25, 0.5, 1.0]
        @test abs(SciMLBase.solve!(integ).u[end][1]) < 1.0e-12
        SciMLBase.reinit!(integ; d_discontinuities = [0.75])
        @test PETScDiffEq.DiffEqBase.get_tstops_array(integ) == [0.75, 1.0]
        SciMLBase.reinit!(integ)
        @test PETScDiffEq.DiffEqBase.get_tstops_array(integ) == [0.5, 1.0]
        @test 0.25 in SciMLBase.solve(prob, alg; dt = 0.1, tstops = 0.25).t
        @test 0.5 in SciMLBase.solve(prob, alg; dt = 0.1, d_discontinuities = 0.5).t

        decay = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        for mark in (false, true)
            integ = SciMLBase.init(decay, alg; dt = 0.1, adaptive = false)
            SciMLBase.step!(integ)
            integ.u[1] = 5.0
            mark && SciMLBase.u_modified!(integ, true)
            SciMLBase.step!(integ)
            @test integ.u[1] ≈ 5exp(-0.1) rtol = 1.0e-6
        end

        fast = SciMLBase.ODEProblem((du, u, p, t) -> (du[1] = -50.0 * u[1]; nothing), [1.0], (0.0, 1.0))
        tight = (; abstol = 1.0e-10, reltol = 1.0e-10, force_dtmin = true)
        steps(sol) = diff(sol.t)[2:(end - 1)]
        over = SciMLBase.solve(fast, alg; dt = 0.01, dtmin = 0.2, dtmax = 0.1, tight...)
        @test over.retcode == SciMLBase.ReturnCode.Success
        @test all(≈(0.2), steps(over))
        integ = SciMLBase.init(fast, alg; dt = 0.01, dtmin = 0.1, tight...)
        SciMLBase.step!(integ)
        integ.opts.dtmin = 0.2
        integ.opts.dtmax = 0.05
        moved = SciMLBase.solve!(integ)
        @test moved.retcode == SciMLBase.ReturnCode.Success
        @test moved.t[end] == 1.0
        @test minimum(diff(moved.t)[3:(end - 1)]) >= 0.2 - 1.0e-12
        stopped = SciMLBase.solve(fast, alg; dt = 0.001, dtmin = 0.01, tstops = [0.5], tight...)
        @test minimum(diff(stopped.t)) >= 0.01 - 1.0e-12
        @test SciMLBase.solve(fast, alg; dt = 0.01, dtmin = -0.1, tight...).t ==
            SciMLBase.solve(fast, alg; dt = 0.01, dtmin = 0.1, tight...).t
        own = SciMLBase.solve(
            fast, PETScDiffEq.TSRK("5dp", ["-ts_adapt_dt_min", "0.05"]);
            dt = 0.01, dtmin = 0.1, tight...,
        )
        @test all(≈(0.05), diff(own.t)[1:(end - 3)])
    end

    @testset "running out of steps is MaxIters" begin
        sol = SciMLBase.solve(
            SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
            dt = 0.01, adaptive = false, maxiters = 5,
        )
        @test sol.retcode == SciMLBase.ReturnCode.MaxIters
        @test sol.t[end] ≈ 0.05
        @test sol.stats.naccept == 5
    end

    @testset "maxiters counts rejected and failed attempts, as OrdinaryDiffEq's does" begin
        breaks = SciMLBase.ODEProblem(
            (du, u, p, t) -> (du[1] = t > 0.5 ? NaN : -u[1]; nothing), [1.0], (0.0, 1.0),
        )
        vdp = SciMLBase.ODEProblem(
            (du, u, p, t) -> (du[1] = u[2]; du[2] = 1000 * (1 - u[1]^2) * u[2] - u[1]; nothing),
            [2.0, 0.0], (0.0, 3000.0),
        )
        counts(sol) = (sol.stats.naccept, sol.stats.nreject, sol.stats.nnonlinconvfail)
        for prob in (breaks, vdp), alg in (
                    PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRosW(), PETScDiffEq.TSImplicit("bdf"),
                    PETScDiffEq.TSARKIMEX(),
                )
            sol = @test_logs SciMLBase.solve(prob, alg; maxiters = 60)
            @test sol.retcode == SciMLBase.ReturnCode.MaxIters
            @test sum(counts(sol)) == 60
            @test sol.stats.naccept < 60
            integ = SciMLBase.init(prob, alg; maxiters = 60)
            @test_logs SciMLBase.solve!(integ)
            @test integ.iter == 60
            @test integ.sol.retcode == SciMLBase.ReturnCode.MaxIters
            @test counts(integ.sol) == counts(sol)
            @test integ.sol.t == sol.t && integ.sol.u == sol.u
        end
    end

    @testset "the nonlinear counters come from the solver, not the stepper" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        implicit = SciMLBase.solve(
            prob, PETScDiffEq.TSImplicit("bdf"); dt = 0.01, adaptive = false,
        )
        @test implicit.stats.nnonliniter > implicit.stats.naccept
        @test implicit.stats.nnonlinconvfail == 0
        explicit = SciMLBase.solve(
            prob, PETScDiffEq.TSRK("5dp"); dt = 0.01, adaptive = false,
        )
        @test explicit.stats.nnonliniter == 0
    end

    @testset "maxiters that no PetscInt can hold" begin
        @test PETScDiffEq._maxsteps(typemax(Int)) ==
            min(typemax(Int), typemax(PETScDiffEq.LibPETSc.PetscInt))
        @test PETScDiffEq._maxsteps(100) == 100
        @test PETScDiffEq._maxsteps(typemax(Int)) isa PETScDiffEq.LibPETSc.PetscInt
        @test SciMLBase.solve(
            SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
            dt = 0.1, maxiters = typemax(Int),
        ).retcode == SciMLBase.ReturnCode.Success
    end

    @testset "the loaded PETSc has the index width the wrappers assume" begin
        lib = PETScDiffEq.PETSc.getlib(PetscScalar = Float64)
        @test PETScDiffEq._check_inttype(lib) === nothing
        @test PETScDiffEq.PETSc.inttype(lib) === PETScDiffEq.LibPETSc.PetscInt
    end

    @testset "a Krylov dot product does not depend on where the vectors sit in memory" begin
        pl = PETScDiffEq.PETSc.getlib(; PetscScalar = Float64)
        lib = PETScDiffEq.LibPETSc
        len, m = 40, 4
        wrap(buf, off) = lib.VecCreateSeqWithArray(
            pl, MPI.COMM_SELF, lib.PetscInt(1), lib.PetscInt(len),
            unsafe_wrap(Array, pointer(buf, off + 1), len),
        )
        # These gaps fail PETSc's stride test, so only the packed copy could go through GEMV.
        offsets = [3len + 200, 0, len + 70, 2len + 140]
        function layout_free(state)
            x = lib.VecDuplicate(pl, state)
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
            agree = map(1:20) do trial
                PETScDiffEq.PETScCompat.with_local_array!(x; read = false, write = true) do a
                    a .= cos.(7trial .+ 3 .* (1:len))
                end
                Y = [sin(17trial + 3i + 5k) for i in 1:len, k in 1:m]
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
            PETScDiffEq.PETScCompat.destroy!(x)
            return all(agree)
        end
        first_order = SciMLBase.init(
            SciMLBase.ODEProblem(decay!, ones(len), (0.0, 1.0)), PETScDiffEq.TSImplicit("bdf"),
        )
        @test layout_free(first_order.h.u)
        SciMLBase.terminate!(first_order)
        second_order = SciMLBase.init(
            SciMLBase.SecondOrderODEProblem(
                (dv, v, u, p, t) -> (dv .= -u; nothing), zeros(len), ones(len), (0.0, 1.0),
            ),
            PETScDiffEq.TSAlpha2(); dt = 0.01,
        )
        @test layout_free(second_order.h.solution)
        SciMLBase.terminate!(second_order)
    end

    @testset "which side of the bracket the root lands on" begin
        rootprob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        function crossed(rootfind)
            conds = Float64[]
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.5,
                integ -> (push!(conds, integ.u[1] - 0.5); nothing); rootfind = rootfind,
            )
            SciMLBase.solve(rootprob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = cb)
            return conds
        end
        left, right = crossed(SciMLBase.LeftRootFind), crossed(SciMLBase.RightRootFind)
        @test length(left) == 1
        @test length(right) == 1
        @test left[1] >= 0
        @test right[1] <= 0
        @test abs(left[1]) < 1.0e-14
        @test abs(right[1]) < 1.0e-14
    end

    @testset "save_idxs keeps the order it was asked for" begin
        three!(du, u, p, t) = (
            du[1] = -u[1]; du[2] = -2u[2]; du[3] = -3u[3]; nothing
        )
        sol = SciMLBase.solve(
            SciMLBase.ODEProblem(three!, [1.0, 2.0, 3.0], (0.0, 1.0)),
            PETScDiffEq.TSRK("5dp"); dt = 0.1, save_idxs = [3, 1],
        )
        @test sol.u[1] == [3.0, 1.0]
        @test sol.u[end] ≈ [3exp(-3), exp(-1)] rtol = 1.0e-3
    end

    @testset "tstops at the ends of the span are dropped" begin
        integ = SciMLBase.init(
            SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
            dt = 0.1, tstops = [0.0, 0.5, 1.0],
        )
        @test integ.tstops == [0.5, 1.0]
    end

    @testset "dense output" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp")
        sol = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false)
        lin = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false, dense = false)
        @test sol.dense
        @test !lin.dense
        @test sol.interp isa SciMLBase.HermiteInterpolation
        @test lin.interp isa SciMLBase.LinearInterpolation
        @test length(sol.interp.du) == length(sol.u)
        @test sol.stats.nf == lin.stats.nf + length(sol.u)
        @test abs(sol(0.55)[1] - exp(-0.55)) < 1.0e-6
        @test abs(lin(0.55)[1] - exp(-0.55)) > 1.0e-4
        @test abs(sol(0.55, Val{1})[1] + exp(-0.55)) < 1.0e-4
        @test sol(sol.t[4]) == sol.u[4]

        @testset "through the integrator" begin
            isol = SciMLBase.solve!(SciMLBase.init(prob, alg; dt = 0.1, adaptive = false))
            @test isol.dense
            @test isol(0.55) == sol(0.55)
        end

        @testset "off by default with saveat, on by request" begin
            s = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false, saveat = 0.25)
            @test !s.dense
            d = SciMLBase.solve(
                prob, alg; dt = 0.1, adaptive = false, saveat = 0.25, dense = true,
            )
            @test d.dense
            @test length(d.interp.du) == length(d.u)
            @test abs(d(0.6)[1] - exp(-0.6)) < 1.0e-4
            @test abs(s(0.6)[1] - exp(-0.6)) > 1.0e-3
        end

        @testset "off with save_everystep = false" begin
            s = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false, save_everystep = false)
            @test !s.dense
        end

        @testset "split problems save f1 + f2" begin
            stiff!(du, u, p, t) = (du[1] = -1000.0 * (u[1] - cos(t)); nothing)
            forcing!(du, u, p, t) = (du[1] = -sin(t); nothing)
            sprob = SciMLBase.SplitODEProblem(stiff!, forcing!, [1.0], (0.0, 0.1))
            s = SciMLBase.solve(
                sprob, PETScDiffEq.TSARKIMEX("3", ["-ts_adapt_type", "none"]); dt = 0.01,
            )
            @test s.dense
            k = 5
            uk, tk = s.u[k][1], s.t[k]
            f1 = -1000.0 * (uk - cos(tk))
            f2 = -sin(tk)
            @test s.interp.du[k][1] ≈ f1 + f2 atol = 1.0e-12
            @test !(s.interp.du[k][1] ≈ f1)
            @test abs(s(0.055)[1] - cos(0.055)) < 1.0e-4
        end

        @testset "not available with a mass matrix" begin
            mprob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(decay!; mass_matrix = fill(2.0, 1, 1)), [1.0], (0.0, 1.0),
            )
            s = SciMLBase.solve(mprob, PETScDiffEq.TSImplicit("bdf"); dt = 0.1)
            @test !s.dense
            @test_throws ArgumentError SciMLBase.solve(
                mprob, PETScDiffEq.TSImplicit("bdf"); dt = 0.1, dense = true,
            )
        end

        @testset "the post-callback point carries the post-jump derivative" begin
            fired = Ref(false)
            jump = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t >= 0.5 && !fired[],
                integ -> (integ.u[1] += 1.0; fired[] = true),
            )
            s = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false, callback = jump)
            i = findfirst(==(0.5), s.t)
            @test i !== nothing && s.t[i + 1] == 0.5
            @test s.interp.du[i] ≈ -s.u[i]
            @test s.interp.du[i + 1] ≈ -s.u[i + 1]
        end
    end

    @testset "reinit!" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        for alg in (PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSImplicit("bdf"))
            fresh = SciMLBase.solve(prob, alg; dt = 0.1)
            integ = SciMLBase.init(prob, alg; dt = 0.1)
            SciMLBase.step!(integ)
            SciMLBase.step!(integ)
            SciMLBase.reinit!(integ)
            @test integ.t == 0.0
            @test integ.u == [1.0]
            @test length(integ.sol.t) == 1
            again = SciMLBase.solve!(integ)
            @test again.t == fresh.t
            @test again.u == fresh.u
            @test again.stats.nf == fresh.stats.nf
            @test again.stats.naccept == fresh.stats.naccept

            @testset "after the integrator has finished" begin
                SciMLBase.reinit!(integ)
                @test !integ.finished
                third = SciMLBase.solve!(integ)
                @test third.u == fresh.u
                @test third.stats.nf == fresh.stats.nf
            end
        end

        @testset "new state and span" begin
            alg = PETScDiffEq.TSRK("5dp")
            integ = SciMLBase.init(prob, alg; dt = 0.1, reltol = 1.0e-8, abstol = 1.0e-10)
            SciMLBase.solve!(integ)
            SciMLBase.reinit!(integ, [2.0]; t0 = 1.0, tf = 2.5)
            sol = SciMLBase.solve!(integ)
            @test sol.t[1] == 1.0
            @test sol.u[1] == [2.0]
            @test sol.t[end] == 2.5
            @test abs(sol.u[end][1] - 2exp(-1.5)) < 1.0e-6
            @test sol.dense
            @test abs(sol(1.75)[1] - 2exp(-0.75)) < 1.0e-6
        end

        @testset "erase_sol = false keeps the earlier points" begin
            alg = PETScDiffEq.TSRK("5dp")
            integ = SciMLBase.init(prob, alg; dt = 0.1, adaptive = false)
            first = SciMLBase.solve!(integ)
            n = length(first.t)
            SciMLBase.reinit!(integ, first.u[end]; t0 = 1.0, tf = 2.0, erase_sol = false)
            sol = SciMLBase.solve!(integ)
            @test length(sol.t) == 2n
            @test sol.t[1:n] == first.t
            @test sol.t[n + 1] == 1.0
            @test length(sol.interp.du) == length(sol.u)
        end

        @testset "erase_sol = false across a change of saving" begin
            alg = PETScDiffEq.TSRK("5dp")
            integ = SciMLBase.init(
                prob, alg; dt = 0.1, saveat = 0.5, abstol = 1.0e-8, reltol = 1.0e-8,
            )
            kept = SciMLBase.solve!(integ)
            @test !kept.dense
            SciMLBase.reinit!(
                integ, kept.u[end]; t0 = 1.0, tf = 2.0, saveat = Float64[],
                erase_sol = false,
            )
            sol = SciMLBase.solve!(integ)
            @test sol.dense
            @test length(sol.interp.du) == length(sol.u)
            @test abs(sol(0.25)[1] - exp(-0.25)) < 1.0e-3
            @test abs(sol(1.5)[1] - kept.u[end][1] * exp(-0.5)) < 1.0e-5
        end

        @testset "erase_sol = false after save_idxs stays non-dense" begin
            rot!(du, u, p, t) = (du[1] = -u[2]; du[2] = u[1]; nothing)
            rot = SciMLBase.ODEProblem(rot!, [1.0, 0.0], (0.0, 1.0))
            for idxs in ([1], 1)
                integ = SciMLBase.init(
                    rot, PETScDiffEq.TSRK("4"); dt = 0.1, adaptive = false, saveat = 0.5,
                    save_idxs = idxs,
                )
                kept = SciMLBase.solve!(integ)
                SciMLBase.reinit!(
                    integ, [cos(1.0), sin(1.0)]; t0 = 1.0, tf = 2.0, erase_sol = false,
                    saveat = Float64[],
                )
                sol = SciMLBase.solve!(integ)
                fresh = SciMLBase.solve(
                    SciMLBase.remake(rot; u0 = [cos(1.0), sin(1.0)], tspan = (1.0, 2.0)),
                    PETScDiffEq.TSRK("4"); dt = 0.1, adaptive = false, save_idxs = idxs,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test !sol.dense
                @test sol.t[1:3] == kept.t
                @test sol.u[1:3] == kept.u
                @test sol.t[4:end] == fresh.t
                @test sol.u[4:end] == fresh.u
                @test sol(0.25) == (sol.u[1] + sol.u[2]) / 2
            end
            osc!(dv, v, u, p, t) = (dv .= -u; nothing)
            osc = SciMLBase.SecondOrderODEProblem(osc!, [0.0], [1.0], (0.0, 1.0))
            integ = SciMLBase.init(
                osc, PETScDiffEq.TSBasicSymplectic("1"); dt = 0.1, saveat = 0.5,
                save_idxs = [2],
            )
            kept = SciMLBase.solve!(integ)
            SciMLBase.reinit!(integ; erase_sol = false, saveat = Float64[])
            sol = SciMLBase.solve!(integ)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test !sol.dense
            @test length(sol.t) == length(kept.t) + 11
            @test sol.u[1:3] == kept.u
            @test all(u -> length(u) == 1, sol.u)
        end

        @testset "saveat can be replaced" begin
            alg = PETScDiffEq.TSRK("5dp")
            integ = SciMLBase.init(prob, alg; dt = 0.1, saveat = 0.5)
            @test SciMLBase.solve!(integ).t == [0.0, 0.5, 1.0]
            SciMLBase.reinit!(integ; saveat = 0.25)
            @test SciMLBase.solve!(integ).t == [0.0, 0.25, 0.5, 0.75, 1.0]
        end

        @testset "callbacks are initialised again unless told not to" begin
            n_init = Ref(0)
            cb = SciMLBase.DiscreteCallback(
                (u, t, integ) -> false, integ -> nothing;
                initialize = (c, u, t, integ) -> (n_init[] += 1; nothing),
            )
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = cb)
            @test n_init[] == 1
            SciMLBase.reinit!(integ)
            @test n_init[] == 2
            SciMLBase.reinit!(integ; reinit_callbacks = false)
            @test n_init[] == 2
            SciMLBase.terminate!(integ)
        end

        @testset "initialize_save = false leaves the start out" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1)
            SciMLBase.reinit!(integ; initialize_save = false)
            @test isempty(integ.sol.t)
            sol = SciMLBase.solve!(integ)
            @test sol.t[1] > 0.0
        end

        @testset "initialize_save = false leaves an initialize callback's change unsaved" begin
            doubled = SciMLBase.DiscreteCallback(
                (u, t, integ) -> false, integ -> nothing;
                initialize = (c, u, t, integ) -> (integ.u .*= 2; nothing),
            )
            for (save, ts, us) in ((true, [0.0, 0.0], [[1.0], [2.0]]), (false, [0.0], [[1.0]]))
                sol = SciMLBase.solve(
                    prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = doubled,
                    save_everystep = false, initialize_save = save,
                )
                @test sol.t[1:(end - 1)] == ts
                @test sol.u[1:(end - 1)] == us
                @test sol.t[end] == 1.0
            end
        end

        if isdefined(SciMLBase, :has_reinit)
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1)
            @test SciMLBase.has_reinit(integ)
            SciMLBase.terminate!(integ)
        end
    end

    @testset "TSMPRK" begin
        two!(du, u, p, t) = (du[1] = -u[1]; du[2] = -100.0 * u[2]; nothing)
        mprob = SciMLBase.ODEProblem(two!, [1.0, 1.0], (0.0, 1.0))

        @testset "every supported subtype integrates the slow part" begin
            for st in ("2a22", "2a32", "p2", "p3")
                sol = SciMLBase.solve(
                    mprob, PETScDiffEq.TSMPRK([1], st); dt = 0.001, adaptive = false,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-6
                @test abs(sol.u[end][2]) < 1.0e-40
            end
        end

        @testset "p3 is more accurate than p2" begin
            err(st) = abs(
                SciMLBase.solve(
                    mprob, PETScDiffEq.TSMPRK([1], st); dt = 0.001, adaptive = false,
                ).u[end][1] - exp(-1.0),
            )
            @test err("p3") < err("p2") / 100
        end

        @testset "a dt that does not divide the span is shortened at the end" begin
            for (st, dt, tol) in (
                    ("p2", 0.3, 1.0e-2), ("p2", 0.03, 1.0e-4),
                    ("p3", 0.3, 5.0e-4), ("p3", 0.1, 1.0e-5),
                )
                sol = SciMLBase.solve(
                    mprob, PETScDiffEq.TSMPRK([1], st); dt = dt, adaptive = false,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.t[end] == 1.0
                @test all(<=(1.0), sol.t)
                @test abs(sol.u[end][1] - exp(-1.0)) < tol
            end
            ends = SciMLBase.solve(
                mprob, PETScDiffEq.TSMPRK([1], "p3"); dt = 0.3, adaptive = false,
                save_everystep = false,
            )
            @test ends.t == [0.0, 1.0]
            @test abs(ends.u[end][1] - exp(-1.0)) < 5.0e-4
            three_way!(du, u, p, t) = (
                du[1] = -u[1]; du[2] = -10.0 * u[2]; du[3] = -100.0 * u[3]; nothing
            )
            split3 = SciMLBase.solve(
                SciMLBase.ODEProblem(three_way!, [1.0, 1.0, 1.0], (0.0, 1.0)),
                PETScDiffEq.TSMPRK([1], [2], "2a33"); dt = 0.3, adaptive = false,
            )
            @test split3.t[end] == 1.0
            @test abs(split3.u[end][1] - exp(-1.0)) < 1.0e-2
            # Backward in time, +10u2 is the decaying fast row.
            back!(du, u, p, t) = (du[1] = -u[1]; du[2] = 10.0 * u[2]; nothing)
            rev = SciMLBase.solve(
                SciMLBase.ODEProblem(back!, [1.0, 1.0], (1.0, 0.0)),
                PETScDiffEq.TSMPRK([1], "p3"); dt = 0.3, adaptive = false,
            )
            @test rev.t[end] == 0.0
            @test all(>=(0.0), rev.t)
            @test abs(rev.u[end][1] - exp(1.0)) < 1.0e-3
        end

        @testset "which half is slow is the caller's to say" begin
            sol = SciMLBase.solve(
                mprob, PETScDiffEq.TSMPRK([2]); dt = 0.001, adaptive = false,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-5
        end

        @testset "the index list is checked" begin
            @test_throws ArgumentError PETScDiffEq.TSMPRK(Int[])
            @test_throws ArgumentError PETScDiffEq.TSMPRK([0])
            @test_throws ArgumentError PETScDiffEq.TSMPRK([1, 1])
            @test PETScDiffEq.TSMPRK([2, 1]).slow == [1, 2]
            @test_throws ArgumentError PETScDiffEq.TSMPRK([1], "2a23")
            @test_throws ArgumentError PETScDiffEq.TSMPRK([1], "nonsense")
            @test_throws ArgumentError SciMLBase.solve(
                mprob, PETScDiffEq.TSMPRK([3]); dt = 0.01,
            )
            @test_throws ArgumentError SciMLBase.solve(
                mprob, PETScDiffEq.TSMPRK([1, 2]); dt = 0.01,
            )
        end

        @testset "the fast part is actually substepped" begin
            stiff!(du, u, p, t) = (du[1] = -u[1]; du[2] = -100.0 * u[2]; nothing)
            sprob = SciMLBase.ODEProblem(stiff!, [1.0, 1.0], (0.0, 1.0))
            settled(alg, dt) = abs(
                SciMLBase.solve(sprob, alg; dt = dt, adaptive = false).u[end][2],
            ) < 1.0e-6
            @test !settled(PETScDiffEq.TSRK("5dp"), 0.0325)
            @test settled(PETScDiffEq.TSMPRK([1], "p2"), 0.0325)
            @test settled(PETScDiffEq.TSMPRK([1], "p3"), 0.0325)
            @test settled(PETScDiffEq.TSRK("5dp"), 0.02)
        end

        @testset "each stage is evaluated once, not once per part" begin
            counted = Ref(0)
            counting!(du, u, p, t) = (
                counted[] += 1; du[1] = -u[1]; du[2] = -100.0 * u[2]; nothing
            )
            cprob = SciMLBase.ODEProblem(counting!, [1.0, 1.0], (0.0, 1.0))
            calls(alg) = (
                counted[] = 0;
                SciMLBase.solve(cprob, alg; dt = 0.01, adaptive = false);
                counted[]
            )
            @test calls(PETScDiffEq.TSMPRK([1], "p2")) <
                calls(PETScDiffEq.TSRK("5dp"))
        end

        @testset "a three-way split" begin
            three!(du, u, p, t) = (
                du[1] = -u[1]; du[2] = -10.0 * u[2]; du[3] = -100.0 * u[3]; nothing
            )
            tprob = SciMLBase.ODEProblem(three!, [1.0, 1.0, 1.0], (0.0, 1.0))
            for st in ("2a23", "2a33")
                sol = SciMLBase.solve(
                    tprob, PETScDiffEq.TSMPRK([1], [2], st); dt = 0.001,
                    adaptive = false,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-6
                @test abs(sol.u[end][2] - exp(-10.0)) < 1.0e-6
            end
            @test PETScDiffEq.TSMPRK([1], [2]).medium == [2]
        end

        @testset "the subtype has to match the number of splits" begin
            @test_throws ArgumentError PETScDiffEq.TSMPRK([1], "2a23")
            @test_throws ArgumentError PETScDiffEq.TSMPRK([1], [2], "p2")
            @test_throws ArgumentError PETScDiffEq.TSMPRK([1], [1], "2a23")
            @test_throws ArgumentError PETScDiffEq.TSMPRK([1], [0], "2a23")
            @test_throws ArgumentError SciMLBase.solve(
                SciMLBase.ODEProblem(two!, [1.0, 1.0], (0.0, 1.0)),
                PETScDiffEq.TSMPRK([1], [2], "2a23"); dt = 0.01,
            )
        end

        @testset "explicit, so a mass matrix is refused" begin
            @test_throws ArgumentError SciMLBase.solve(
                SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(two!; mass_matrix = [2.0 0.0; 0.0 1.0]),
                    [1.0, 1.0], (0.0, 1.0),
                ), PETScDiffEq.TSMPRK([1]); dt = 0.01,
            )
        end
    end

    @testset "PETSc types this package cannot drive are refused" begin
        for t in ("alpha2", "basicsymplectic", "discgrad", "eimex", "mimex", "mprk", "pseudo")
            @test_throws ArgumentError PETScDiffEq.TSGeneric(t)
            @test_throws ArgumentError PETScDiffEq.TSGeneric(t; explicit = true)
        end

        for t in ("euler", "glee", "rk", "ssp")
            @test_throws ArgumentError PETScDiffEq.TSGeneric(t)
            @test PETScDiffEq.TSGeneric(t; explicit = true).ts_type == t
        end

        for t in ("alpha", "beuler", "bdf", "cn", "dirk", "glle", "irk", "rosw", "theta")
            @test PETScDiffEq.TSGeneric(t).ts_type == t
        end
        @test_throws "`irk` is an implicit PETSc type" PETScDiffEq.TSGeneric("irk"; explicit = true)
    end

    @testset "the same types are refused when an option selects them" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        solve_at(alg; kw...) = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false, kw...)
        for t in ("alpha2", "basicsymplectic", "discgrad", "eimex", "mimex", "mprk", "pseudo"), alg in (
                    PETScDiffEq.TSImplicit("beuler", ["-ts_type", t]),
                    PETScDiffEq.TSRK("4", ["-ts_type=$t"]),
                    PETScDiffEq.TSGeneric("glee", ["-TS_TYPE", t]; explicit = true),
                )
            @test_throws "PETScDiffEq cannot drive `$t`" solve_at(alg)
        end
        for t in ("euler", "glee", "rk", "ssp"), alg in (
                    PETScDiffEq.TSImplicit("beuler", ["-ts_type", t]),
                    PETScDiffEq.TSRosW("ra34pw2", ["-ts_type=$t"]),
                    PETScDiffEq.TSGeneric("beuler", ["-TS_TYPE", t]),
                )
            @test_throws "`$t` is an explicit PETSc type" solve_at(alg)
            @test_throws "`$t` is an explicit PETSc type" SciMLBase.init(
                prob, alg; dt = 0.1, adaptive = false,
            )
        end
        for alg in (
                PETScDiffEq.TSRK("4", ["-ts_type", "irk"]),
                PETScDiffEq.TSGeneric("euler", ["-ts_type=irk"]; explicit = true),
            )
            @test_throws "`irk` is an implicit PETSc type" solve_at(alg)
        end
        sol = solve_at(PETScDiffEq.TSRK("4", ["-ts_type", "glee"]))
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test abs(sol.u[end][1] - exp(-1)) < 1.0e-3
    end

    @testset "subtype refusals hold when an option selects the subtype" begin
        pair_jac!(J, u, p, t) = (J .= 0.0; J[1, 1] = -1.0; J[2, 2] = -1.0; nothing)
        mass = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = pair_jac!, mass_matrix = Diagonal([2.0, 1.0])),
            [1.0, 1.0], (0.0, 1.0),
        )
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        split = SciMLBase.SplitODEProblem(decay!, decay!, [1.0], (0.0, 1.0))
        halved!(r, du, u, p, t) = (r[1] = 2du[1] + u[1]; nothing)
        dae = SciMLBase.DAEProblem(halved!, [-0.5], [1.0], (0.0, 1.0))
        pb = ["-pc_type", "pbjacobi"]
        for (pr, alg, msg) in (
                (mass, PETScDiffEq.TSImplicit("beuler", ["-ts_type", "irk", pb...]), "a mass matrix with TSIRK"),
                (mass, PETScDiffEq.TSGeneric("beuler", ["-ts_type", "irk", pb...]), "a mass matrix with TSIRK"),
                (mass, PETScDiffEq.TSRosW("ra34pw2", ["-ts_rosw_type", "assp3p3s1c"]), "cannot take a mass matrix"),
                (mass, PETScDiffEq.TSGeneric("rosw", ["-ts_rosw_type", "assp3p3s1c"]), "cannot take a mass matrix"),
                (prob, PETScDiffEq.TSRosW("ra34pw2", ["-ts_rosw_type", "ark3"]), "cannot be used"),
                (prob, PETScDiffEq.TSARKIMEX("3", ["-ts_arkimex_type", "ars122"]), "needs a SplitODEProblem"),
                (split, PETScDiffEq.TSARKIMEX("3", ["-ts_arkimex_type", "bpr3"]), "converges at first order"),
                (dae, PETScDiffEq.TSDAE("irk", pb), "a DAEProblem with TSIRK"),
                (dae, PETScDiffEq.TSDAE("beuler", ["-ts_type", "irk", pb...]), "a DAEProblem with TSIRK"),
            )
            @test_throws msg SciMLBase.solve(pr, alg; dt = 0.01, adaptive = false)
            @test_throws msg SciMLBase.init(pr, alg; dt = 0.01, adaptive = false)
        end
    end

    @testset "a subtype an option replaces is not refused" begin
        pair_jac!(J, u, p, t) = (J .= 0.0; J[1, 1] = -1.0; J[2, 2] = -1.0; nothing)
        mass = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = pair_jac!, mass_matrix = Diagonal([2.0, 1.0])),
            [1.0, 1.0], (0.0, 1.0),
        )
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        split = SciMLBase.SplitODEProblem(decay!, decay!, [1.0], (0.0, 1.0))
        fd = PETScDiffEq.AutoFiniteDiff()
        ra34pw2 = ["-ts_rosw_type", "ra34pw2"]
        for (pr, alg, runs) in (
                (mass, PETScDiffEq.TSRosW("assp3p3s1c", ra34pw2), PETScDiffEq.TSRosW("ra34pw2")),
                (
                    prob, PETScDiffEq.TSRosW("assp3p3s1c", ra34pw2; autodiff = fd),
                    PETScDiffEq.TSRosW("ra34pw2"; autodiff = fd),
                ),
                (prob, PETScDiffEq.TSRosW("ark3", ra34pw2), PETScDiffEq.TSRosW("ra34pw2")),
                (
                    prob, PETScDiffEq.TSARKIMEX("ars122", ["-ts_arkimex_type", "3"]),
                    PETScDiffEq.TSARKIMEX("3"),
                ),
                (
                    split, PETScDiffEq.TSARKIMEX("bpr3", ["-ts_arkimex_type", "3"]),
                    PETScDiffEq.TSARKIMEX("3"),
                ),
                (mass, PETScDiffEq.TSIRK(2, ["-ts_type", "bdf"]), PETScDiffEq.TSImplicit("bdf")),
                (
                    prob, PETScDiffEq.TSIRK(2, ["-ts_type", "bdf"]; autodiff = fd),
                    PETScDiffEq.TSImplicit("bdf"; autodiff = fd),
                ),
                (mass, PETScDiffEq.TSGeneric("irk", ["-ts_type", "bdf"]), PETScDiffEq.TSGeneric("bdf")),
            )
            sol = SciMLBase.solve(pr, alg; dt = 0.01)
            ran = SciMLBase.solve(pr, runs; dt = 0.01)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.t == ran.t
            @test sol.u == ran.u
        end
    end

    @testset "a solve that never uses the Jacobian is called out" begin
        prob = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = decay_jac!), [1.0], (0.0, 1.0),
        )
        plain = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))

        @testset "and the same type is fine when told it is explicit" begin
            sol = SciMLBase.solve(
                plain, PETScDiffEq.TSGeneric("glee"; explicit = true); dt = 0.1,
                adaptive = false,
            )
            @test abs(sol.u[end][1] - exp(-1)) < 1.0e-3
        end

        @testset "an empty callback set leaves the solve to TSSolve" begin
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSGeneric("glle"); dt = 0.1, adaptive = false,
                callback = SciMLBase.CallbackSet(),
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test abs(sol.u[end][1] - exp(-1)) < 1.0e-3
        end

        @testset "no implicit method that works trips it" begin
            for alg in (
                    PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSImplicit("cn"),
                    PETScDiffEq.TSRosW("ra34pw2"), PETScDiffEq.TSARKIMEX("3"),
                    PETScDiffEq.TSIRK(2), PETScDiffEq.TSGeneric("glle"),
                    PETScDiffEq.TSGeneric("alpha"),
                )
                sol = @test_logs min_level = Logging.Warn SciMLBase.solve(
                    prob, alg; dt = 0.1, adaptive = false,
                )
                @test sol.stats.njacs > 0
            end
        end
    end

    @testset "BDF order" begin
        prob = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = decay_jac!), [1.0], (0.0, 1.0),
        )
        kw = (dt = 1.0e-3, reltol = 1.0e-8, abstol = 1.0e-10)

        @testset "the keyword matches the PETSc option" begin
            a = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("bdf"; order = 5); kw...)
            b = SciMLBase.solve(
                prob, PETScDiffEq.TSImplicit("bdf", ["-ts_bdf_order", "5"]); kw...,
            )
            @test a.t == b.t
            @test a.u == b.u
        end

        @testset "it changes the integration" begin
            low = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("bdf"; order = 1); kw...)
            high = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("bdf"; order = 5); kw...)
            @test low.stats.naccept > high.stats.naccept
            @test abs(high.u[end][1] - exp(-1)) < 1.0e-6
            @test abs(low.u[end][1] - exp(-1)) < 1.0e-4
        end

        @testset "only bdf takes an order, and only 1 through 6" begin
            @test_throws ArgumentError PETScDiffEq.TSImplicit("theta"; order = 3)
            @test_throws ArgumentError PETScDiffEq.TSImplicit("bdf"; order = 0)
            @test_throws ArgumentError PETScDiffEq.TSImplicit("bdf"; order = 7)
        end

        @testset "the positional constructors are unchanged" begin
            @test PETScDiffEq.TSImplicit().subtype == "beuler"
            @test PETScDiffEq.TSImplicit("theta", 0.3).theta == 0.3
            @test length(PETScDiffEq.TSImplicit("bdf", ["-x", "1"]).petsc_options) == 2
            @test PETScDiffEq.TSImplicit("theta", 0.3, ["-x", "1"]).theta == 0.3
            @test PETScDiffEq.TSImplicit("bdf").order === nothing
        end
    end

    @testset "DAEProblem" begin
        function resid!(r, du, u, p, t)
            r[1] = du[1] + u[1]
            return r[2] = u[2] - u[1]
        end
        function dae_jac!(J, du, u, p, gamma, t)
            J[1, 1] = gamma + 1.0
            J[1, 2] = 0.0
            J[2, 1] = -1.0
            return J[2, 2] = 1.0
        end
        function wrong_dae_jac!(J, du, u, p, gamma, t)
            J[1, 1] = 99.0
            J[1, 2] = 0.0
            J[2, 1] = 5.0
            return J[2, 2] = 3.0
        end
        u0, du0, tspan = [1.0, 1.0], [-1.0, -1.0], (0.0, 1.0)
        withjac = SciMLBase.DAEFunction(resid!; jac = dae_jac!)

        @testset "both components solve, with and without a jac" begin
            for (f, name) in ((SciMLBase.DAEFunction(resid!), "no jac"), (withjac, "jac"))
                for (subtype, tol) in (("bdf", 1.0e-6), ("cn", 1.0e-6), ("beuler", 1.0e-3))
                    sol = SciMLBase.solve(
                        SciMLBase.DAEProblem(f, du0, u0, tspan), PETScDiffEq.TSDAE(subtype);
                        dt = 1.0e-3, adaptive = false,
                    )
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test abs(sol.u[end][1] - exp(-1)) < tol
                    @test abs(sol.u[end][2] - exp(-1)) < tol
                end
            end
        end

        @testset "the analytic Jacobian reaches PETSc" begin
            plain = SciMLBase.solve(
                SciMLBase.DAEProblem(SciMLBase.DAEFunction(resid!), du0, u0, tspan),
                PETScDiffEq.TSDAE(; autodiff = PETScDiffEq.AutoFiniteDiff()); dt = 1.0e-3, adaptive = false,
            )
            given = SciMLBase.solve(
                SciMLBase.DAEProblem(withjac, du0, u0, tspan), PETScDiffEq.TSDAE();
                dt = 1.0e-3, adaptive = false,
            )
            @test plain.stats.njacs == 0
            @test given.stats.njacs > 0
            @test SciMLBase.solve(
                SciMLBase.DAEProblem(
                    SciMLBase.DAEFunction(resid!; jac = wrong_dae_jac!), du0, u0, tspan,
                ), PETScDiffEq.TSDAE("bdf", ["-ts_max_snes_failures", "1"]); dt = 1.0e-3,
                adaptive = false,
            ).retcode != SciMLBase.ReturnCode.Success
        end

        @testset "the Jacobian is asked about the same point as the residual" begin
            seen_r = Tuple{Float64, Vector{Float64}, Vector{Float64}}[]
            seen_j = Tuple{Float64, Vector{Float64}, Vector{Float64}, Float64}[]
            function recording_resid!(r, du, u, p, t)
                push!(seen_r, (t, copy(u), copy(du)))
                r[1] = du[1] + u[1]
                return r[2] = u[2] - u[1]
            end
            function recording_jac!(J, du, u, p, gamma, t)
                push!(seen_j, (t, copy(u), copy(du), gamma))
                J[1, 1] = gamma + 1.0
                J[1, 2] = 0.0
                J[2, 1] = -1.0
                return J[2, 2] = 1.0
            end
            recording = SciMLBase.DAEFunction(recording_resid!; jac = recording_jac!)
            for dt in (0.1, 0.05)
                empty!(seen_r)
                empty!(seen_j)
                SciMLBase.solve(
                    SciMLBase.DAEProblem(recording, du0, u0, tspan),
                    PETScDiffEq.TSDAE("beuler"); dt = dt, adaptive = false,
                )
                @test length(seen_j) == round(Int, 1.0 / dt)
                @test all(
                    j -> any(
                        s -> s[1] == j[1] && s[2] == j[2] && s[3] == j[3], seen_r
                    ), seen_j,
                )
                @test all(j -> isapprox(j[4], 1 / dt; rtol = 1.0e-9), seen_j)
                @test extrema(j -> j[1], seen_j) == (dt, 1.0)
            end
        end

        @testset "a Jacobian that depends on du" begin
            function cubic_resid!(r, du, u, p, t)
                r[1] = du[1]^3 + u[1]
                return r[2] = u[2] - u[1]
            end
            function cubic_jac!(J, du, u, p, gamma, t)
                J[1, 1] = gamma * 3 * du[1]^2 + 1.0
                J[1, 2] = 0.0
                J[2, 1] = -1.0
                return J[2, 2] = 1.0
            end
            cubic = SciMLBase.DAEProblem(
                SciMLBase.DAEFunction(cubic_resid!; jac = cubic_jac!), du0, u0, tspan,
            )
            exact = (1 - 2 / 3)^1.5
            errs = [
                abs(
                    SciMLBase.solve(
                        cubic, PETScDiffEq.TSDAE("beuler"); dt = dt, adaptive = false,
                    ).u[end][1] - exact,
                ) for dt in (0.02, 0.01)
            ]
            @test errs[1] / errs[2] > 1.7
            @test errs[2] < 2.0e-3
        end

        @testset "the BDF order carries to the DAE side" begin
            steps(alg) = SciMLBase.solve(
                SciMLBase.DAEProblem(withjac, du0, u0, tspan), alg;
                dt = 1.0e-3, reltol = 1.0e-8, abstol = 1.0e-10,
            ).stats.naccept
            @test steps(PETScDiffEq.TSDAE("bdf"; order = 5)) <
                steps(PETScDiffEq.TSDAE("bdf")) / 2
            @test_throws ArgumentError PETScDiffEq.TSDAE("beuler"; order = 3)
            @test_throws ArgumentError PETScDiffEq.TSDAE("bdf"; order = 9)
        end

        @testset "backward Euler is first order on the residual" begin
            errs = [
                abs(
                    SciMLBase.solve(
                        SciMLBase.DAEProblem(withjac, du0, u0, tspan),
                        PETScDiffEq.TSDAE("beuler"); dt = dt, adaptive = false,
                    ).u[end][1] - exp(-1),
                ) for dt in (0.02, 0.01, 0.005)
            ]
            orders = [log2(errs[i] / errs[i + 1]) for i in 1:2]
            @test all(o -> isapprox(o, 1; atol = 0.1), orders)
        end

        @testset "what a DAE cannot ask for" begin
            prob = SciMLBase.DAEProblem(withjac, du0, u0, tspan)
            @test_throws Exception SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 1.0e-3,
            )
            @test_throws ArgumentError SciMLBase.solve(
                prob, PETScDiffEq.TSDAE(); dt = 1.0e-3, dense = true,
            )
            @test !SciMLBase.solve(prob, PETScDiffEq.TSDAE(); dt = 1.0e-3).dense
        end

        @testset "saving and the integrator work as for an ODE" begin
            prob = SciMLBase.DAEProblem(withjac, du0, u0, tspan)
            at = SciMLBase.solve(
                prob, PETScDiffEq.TSDAE(); dt = 1.0e-3, adaptive = false, saveat = 0.25,
            )
            @test at.t == [0.0, 0.25, 0.5, 0.75, 1.0]
            @test abs(at.u[3][1] - exp(-0.5)) < 1.0e-6
            integ = SciMLBase.init(prob, PETScDiffEq.TSDAE(); dt = 1.0e-3, adaptive = false)
            SciMLBase.step!(integ)
            @test integ.t > 0.0
            sol = SciMLBase.solve!(integ)
            @test abs(sol.u[end][1] - exp(-1)) < 1.0e-6
        end
    end

    @testset "DAE initialization" begin
        function rober!(du, u, p, t)
            du[1] = -0.04 * u[1] + 1.0e4 * u[2] * u[3]
            du[2] = 0.04 * u[1] - 3.0e7 * u[2]^2 - 1.0e4 * u[2] * u[3]
            du[3] = u[1] + u[2] + u[3] - 1
            return nothing
        end
        function rober_residual!(r, du, u, p, t)
            rober!(r, u, p, t)
            r[1] -= du[1]
            r[2] -= du[2]
            return nothing
        end
        function rober_jac!(J, u, p, t)
            J[1, 1], J[1, 2], J[1, 3] = -0.04, 1.0e4 * u[3], 1.0e4 * u[2]
            J[2, 1], J[2, 2], J[2, 3] = 0.04, -6.0e7 * u[2] - 1.0e4 * u[3], -1.0e4 * u[2]
            J[3, 1], J[3, 2], J[3, 3] = 1.0, 1.0, 1.0
            return nothing
        end
        good, bad = [1.0, 0.0, 0.0], [1.0, 0.0, 0.2]
        span, tol = (0.0, 100.0), (abstol = 1.0e-8, reltol = 1.0e-8)
        mass(u0; kw...) = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(rober!; mass_matrix = Diagonal([1.0, 1.0, 0.0]), kw...), u0,
            span,
        )
        dae(u0, du0; differential_vars = [true, true, false]) = SciMLBase.DAEProblem(
            rober_residual!, du0, u0, span; differential_vars,
        )
        brown = DiffEqBase.BrownFullBasicInit()
        shampine = DiffEqBase.ShampineCollocationInit()
        with_mass = (
            PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSRosW(), PETScDiffEq.TSARKIMEX(),
        )
        checks(prob, alg; kw...) = (SciMLBase.solve(prob, alg; maxiters = 1, kw...); true)

        @testset "the default checks the start and throws, as OrdinaryDiffEq's does" begin
            for alg in with_mass
                @test_throws SciMLBase.CheckInitFailureError SciMLBase.solve(mass(bad), alg)
                @test checks(mass(good), alg)
            end
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.init(
                mass(bad), PETScDiffEq.TSImplicit("bdf"),
            )
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.solve(
                dae(bad, zeros(3)), PETScDiffEq.TSDAE(),
            )
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.solve(
                dae(good, zeros(3)), PETScDiffEq.TSDAE(),
            )
            @test checks(dae(good, [-0.04, 0.04, 0.0]), PETScDiffEq.TSDAE())
            @test checks(mass(bad), PETScDiffEq.TSImplicit("bdf"); initializealg = SciMLBase.NoInit())
        end

        @testset "abstol bounds the RMS of the algebraic residual" begin
            rms = 0.2 / sqrt(3)
            alg = PETScDiffEq.TSImplicit("bdf")
            @test_throws SciMLBase.CheckInitFailureError checks(mass(bad), alg; abstol = 0.999rms)
            @test checks(mass(bad), alg; abstol = 1.001rms)
            @test checks(mass(bad), alg; abstol = [1.0e-6, 1.0e-6, 0.2])
            @test_throws SciMLBase.CheckInitFailureError checks(
                mass(bad), alg; abstol = [1.0e-6, 1.0e-6, 0.1],
            )
            zero_entry = [1.0e-6, 0.0, 1.0e-6]
            @test checks(mass(good), alg; abstol = zero_entry)
            sol = SciMLBase.solve(
                mass(good), alg; abstol = zero_entry, initializealg = shampine, maxiters = 1,
            )
            @test sol.u[1] == good
            @test_throws SciMLBase.CheckInitFailureError checks(
                mass(bad), alg; abstol = [1.0, 1.0, 0.0],
            )
            @test checks(mass(good), alg; abstol = 1.0e-6 + 0im)
            @test checks(
                dae(good, [-0.04, 0.04, 0.0]), PETScDiffEq.TSDAE(); abstol = 1.0e-6 + 0im,
            )
        end

        # PETSc's absolute-eps step check fails these Robertson spans on 32-bit x86.
        Sys.WORD_SIZE == 64 && @testset "BrownFullBasicInit solves for the algebraic variables" begin
            for (prob, alg) in (
                    ((mass(bad), alg) for alg in with_mass)...,
                    (dae(bad, zeros(3)), PETScDiffEq.TSDAE()),
                )
                sol = SciMLBase.solve(prob, alg; initializealg = brown, tol...)
                ref = SciMLBase.solve(
                    prob isa SciMLBase.DAEProblem ? dae(good, [-0.04, 0.04, 0.0]) : mass(good),
                    alg; tol...,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.u[1][1:2] == bad[1:2]
                @test abs(sum(sol.u[1]) - 1) <= 1.0e-10
                @test all(isapprox.(sol.u[end], ref.u[end]; rtol = 1.0e-13))
            end
            g = 9.81
            function pendulum!(du, u, p, t)
                x, y, vx, vy, lambda = u
                du[1], du[2] = vx, vy
                du[3], du[4] = -lambda * x, -lambda * y - g
                du[5] = vx^2 + vy^2 - lambda * (x^2 + y^2) - g * y
                return nothing
            end
            u0 = [sqrt(0.5), -sqrt(0.5), 0.3, 0.3, 0.0]
            exact = (u0[3]^2 + u0[4]^2 - g * u0[2]) / (u0[1]^2 + u0[2]^2)
            for alg in with_mass
                sol = SciMLBase.solve(
                    SciMLBase.ODEProblem(
                        SciMLBase.ODEFunction(
                            pendulum!; mass_matrix = Diagonal([1.0, 1.0, 1.0, 1.0, 0.0]),
                        ), u0, (0.0, 1.0),
                    ), alg; initializealg = brown, tol...,
                )
                @test sol.u[1][1:4] == u0[1:4]
                @test isapprox(sol.u[1][5], exact; rtol = 1.0e-14)
            end
            function pendulum_residual!(r, du, u, p, t)
                pendulum!(r, u, p, t)
                r[1:4] .-= du[1:4]
                return nothing
            end
            prob = SciMLBase.DAEProblem(
                pendulum_residual!, zeros(5), u0, (0.0, 1.0);
                differential_vars = [true, true, true, true, false],
            )
            fd = PETScDiffEq.AutoFiniteDiff()
            for alg in (PETScDiffEq.TSDAE(), PETScDiffEq.TSDAE(; autodiff = fd))
                sol = SciMLBase.solve(prob, alg; initializealg = brown, tol...)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.u[1][1:4] == u0[1:4]
                @test isapprox(sol.u[1][5], exact; rtol = 1.0e-14)
            end
        end

        Sys.WORD_SIZE == 64 && @testset "ShampineCollocationInit takes one backward Euler step" begin
            fbdf = [0.9961513330874654, 3.5651156852644935e-5, 0.0038130157556819193]
            for alg in with_mass
                sol = SciMLBase.solve(mass(bad), alg; initializealg = shampine, tol...)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test all(isapprox.(sol.u[1], fbdf; rtol = 1.0e-6))
            end
            dfbdf = [0.8818094150587155, 1.9846976089331387e-5, 0.1181707379651953]
            sol = SciMLBase.solve(
                dae(bad, zeros(3)), PETScDiffEq.TSDAE(); initializealg = shampine, tol...,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test all(isapprox.(sol.u[1], dfbdf; rtol = 1.0e-10))
            function step_residual(u, h)
                r = zeros(3)
                rober_residual!(r, (u .- bad) ./ h, u, nothing, 0.0)
                return maximum(abs, r)
            end
            later = SciMLBase.remake(dae(bad, zeros(3)); tspan = (1.0, 100.0))
            for (prob, alg, kw, h) in (
                    (mass(bad), PETScDiffEq.TSImplicit("bdf"), (; dt = 1.0e-3), 2.0e-4),
                    (
                        mass(bad), PETScDiffEq.TSImplicit("bdf"),
                        (; initializealg = DiffEqBase.ShampineCollocationInit(1.0e-2)), 1.0e-2,
                    ),
                    (later, PETScDiffEq.TSDAE(), (;), 1.0e-3),
                )
                sol = SciMLBase.solve(prob, alg; initializealg = shampine, kw..., tol...)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test step_residual(sol.u[1], h) <= 1.0e-12
            end
        end

        @testset "sparse LU on an algebraic block with a zero diagonal" begin
            function cell!(du, u, p, t)
                du[1], du[2], du[3] = u[2] - u[3], u[1] - 2u[3], u[2] - u[1]
                return nothing
            end
            rows, cols = [1, 1, 2, 2, 3, 3], [2, 3, 1, 3, 1, 2]
            vals = [1.0, -1.0, 1.0, -2.0, -1.0, 1.0]
            cell_jac!(J, u, p, t) = (foreach((i, j, v) -> J[i, j] = v, rows, cols, vals); nothing)
            proto = sparse(rows, cols, ones(6), 3, 3)
            h = 1.0e-3
            ads = (PETScDiffEq.AutoForwardDiff(), PETScDiffEq.AutoFiniteDiff())
            for jac in (nothing, cell_jac!), ad in ads
                fn = SciMLBase.ODEFunction(
                    cell!; jac, jac_prototype = proto, mass_matrix = Diagonal([0.0, 0.0, 1.0]),
                )
                prob = SciMLBase.ODEProblem(fn, [3.0, 1.0, 1.0], (0.0, 1.0))
                for (init, start) in ((brown, [2.0, 1.0, 1.0]), (shampine, [2.0, 1.0, 1.0] / (1 + h)))
                    sol = SciMLBase.solve(
                        prob, PETScDiffEq.TSImplicit("bdf"; autodiff = ad); initializealg = init,
                        tol...,
                    )
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test isapprox(sol.u[1], start; rtol = 1.0e-15)
                end
            end
        end

        Sys.WORD_SIZE == 64 && @testset "every source of the Jacobian gives the same start" begin
            proto = sparse(ones(3, 3))
            fd = PETScDiffEq.AutoFiniteDiff()
            for init in (brown, shampine)
                ref = SciMLBase.solve(
                    mass(bad), PETScDiffEq.TSImplicit("bdf"); initializealg = init, tol...,
                ).u[1]
                for (prob, alg) in (
                        (mass(bad; jac = rober_jac!), PETScDiffEq.TSImplicit("bdf")),
                        (mass(bad; jac_prototype = proto), PETScDiffEq.TSImplicit("bdf")),
                        (
                            mass(bad; jac = rober_jac!, jac_prototype = proto),
                            PETScDiffEq.TSImplicit("bdf"),
                        ),
                        (mass(bad), PETScDiffEq.TSImplicit("bdf"; autodiff = fd)),
                        (mass(bad; jac_prototype = proto), PETScDiffEq.TSImplicit("bdf"; autodiff = fd)),
                    )
                    u = SciMLBase.solve(prob, alg; initializealg = init, tol...).u[1]
                    @test maximum(abs, u - ref) <= 1.0e-14
                end
            end
        end

        Sys.WORD_SIZE == 64 && @testset "Float32 and complex states" begin
            f32 = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(rober!; mass_matrix = Diagonal(Float32[1, 1, 0])),
                Float32.(bad), (0.0f0, 100.0f0),
            )
            sol = SciMLBase.solve(f32, PETScDiffEq.TSImplicit("bdf"); initializealg = brown)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test eltype(sol.u[1]) == Float32
            @test abs(sol.u[1][3]) <= eps(Float32)
            real_start = SciMLBase.solve(
                mass(bad), PETScDiffEq.TSImplicit("bdf"); initializealg = shampine, tol...,
            ).u[1]
            sol = SciMLBase.solve(
                mass(complex.(bad)), PETScDiffEq.TSImplicit("bdf"); initializealg = shampine,
                tol...,
            )
            @test sol.u[1] == real_start
            sol = SciMLBase.solve(
                mass(bad), PETScDiffEq.TSImplicit("bdf"); initializealg = shampine,
                abstol = fill(1.0e-8 + 0im, 3), reltol = 1.0e-8,
            )
            @test sol.u[1] == real_start
        end

        @testset "a nonlinear solve that fails is InitialFailure" begin
            noroot!(du, u, p, t) = (du[1] = -u[1]; du[2] = u[2]^2 + 1; nothing)
            prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(noroot!; mass_matrix = Diagonal([1.0, 0.0])), [1.0, 0.0],
                (0.0, 1.0),
            )
            for init in (brown, shampine)
                sol = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("bdf"); initializealg = init)
                @test sol.retcode == SciMLBase.ReturnCode.InitialFailure
                @test sol.t == [0.0]
                @test sol.u == [[1.0, 0.0]]
            end
            integ = SciMLBase.init(prob, PETScDiffEq.TSRosW(); initializealg = brown)
            @test integ.sol.retcode == SciMLBase.ReturnCode.InitialFailure
            SciMLBase.step!(integ)
            @test integ.t == 0.0
            @test SciMLBase.solve!(integ).retcode == SciMLBase.ReturnCode.InitialFailure
            noroot_residual!(r, du, u, p, t) = (r[1] = du[1] + u[1]; r[2] = u[2]^2 + 1; nothing)
            sol = SciMLBase.solve(
                SciMLBase.DAEProblem(
                    noroot_residual!, [-1.0, 0.0], [1.0, 0.0], (0.0, 1.0);
                    differential_vars = [true, false],
                ), PETScDiffEq.TSDAE(); initializealg = brown,
            )
            @test sol.retcode == SciMLBase.ReturnCode.InitialFailure
            integ = SciMLBase.init(
                prob, PETScDiffEq.TSImplicit("bdf"); initializealg = SciMLBase.NoInit(),
            )
            SciMLBase.initialize_dae!(integ, brown)
            @test integ.sol.retcode == SciMLBase.ReturnCode.InitialFailure
            @test integ.u == [1.0, 0.0]
            SciMLBase.step!(integ)
            @test integ.t == 0.0
            @test SciMLBase.solve!(integ).retcode == SciMLBase.ReturnCode.InitialFailure
        end

        @testset "reinit! initializes again unless told not to" begin
            integ = SciMLBase.init(
                mass(good), PETScDiffEq.TSImplicit("bdf"); initializealg = brown, tol...,
            )
            SciMLBase.step!(integ)
            SciMLBase.reinit!(integ, bad)
            @test integ.u[1:2] == bad[1:2]
            @test abs(sum(integ.u) - 1) <= 1.0e-10
            SciMLBase.reinit!(integ, bad; reinit_dae = false)
            @test integ.u == bad
            SciMLBase.terminate!(integ)
            integ = SciMLBase.init(mass(good), PETScDiffEq.TSImplicit("bdf"))
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.reinit!(integ, bad)
            SciMLBase.terminate!(integ)
        end

        Sys.WORD_SIZE == 64 && @testset "a consistent start is left as it is" begin
            for (prob, alg) in (
                    (mass(good), PETScDiffEq.TSRosW()),
                    (dae(good, [-0.04, 0.04, 0.0]), PETScDiffEq.TSDAE()),
                )
                ref = SciMLBase.solve(prob, alg; initializealg = SciMLBase.NoInit(), tol...)
                for init in (SciMLBase.CheckInit(), brown, shampine)
                    sol = SciMLBase.solve(prob, alg; initializealg = init, tol...)
                    @test sol.t == ref.t
                    @test sol.u == ref.u
                    @test sol.stats.nf == ref.stats.nf
                end
            end
        end

        @testset "what it refuses" begin
            @test_throws ArgumentError SciMLBase.solve(
                dae(bad, zeros(3); differential_vars = nothing), PETScDiffEq.TSDAE();
                initializealg = brown,
            )
            @test_throws ArgumentError SciMLBase.solve(
                mass(good), PETScDiffEq.TSImplicit("bdf");
                initializealg = DiffEqBase.BrownFullBasicInit(; nlsolve = :newton),
            )
            @test_throws ArgumentError SciMLBase.solve(
                mass(good), PETScDiffEq.TSImplicit("bdf"); initializealg = :brown,
            )
            @test checks(
                mass(good), PETScDiffEq.TSImplicit("bdf"); initializealg = SciMLBase.OverrideInit(),
            )
            skew = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    (du, u, p, t) -> (du .= u .- 1; nothing);
                    mass_matrix = [1.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],
                ), zeros(3), (0.0, 1.0),
            )
            @test_throws "2 zero rows and 1 zero columns" SciMLBase.solve(
                skew, PETScDiffEq.TSImplicit("bdf"); initializealg = brown,
            )
        end

        Sys.WORD_SIZE == 64 && @testset "both solve on a communicator of one rank" begin
            comm = MPI.COMM_WORLD
            fd = PETScDiffEq.AutoFiniteDiff()
            proto = sparse(ones(3, 3))
            rober_dae_jac!(J, du, u, p, gamma, t) =
                (rober_jac!(J, u, p, t); J[1, 1] -= gamma; J[2, 2] -= gamma; nothing)
            function rober_dae(jac)
                kw = jac ? (; jac = rober_dae_jac!, jac_prototype = proto) :
                    (; jac_prototype = proto)
                return SciMLBase.DAEProblem(
                    SciMLBase.DAEFunction(rober_residual!; kw...), zeros(3), bad, span;
                    differential_vars = [true, true, false],
                )
            end
            bdf, dae_alg = PETScDiffEq.TSImplicit("bdf"; comm), PETScDiffEq.TSDAE(; comm)
            serial_bdf = PETScDiffEq.TSImplicit("bdf"; autodiff = fd)
            serial_dae = PETScDiffEq.TSDAE(; autodiff = fd)
            for jac in (true, false), init in (brown, shampine)
                kw = jac ? (; jac = rober_jac!, jac_prototype = proto) : (; jac_prototype = proto)
                for (prob, alg, serial) in (
                        (mass(bad; kw...), bdf, serial_bdf),
                        (rober_dae(jac), dae_alg, serial_dae),
                    )
                    sol = SciMLBase.solve(prob, alg; initializealg = init, tol...)
                    ref = SciMLBase.solve(prob, serial; initializealg = init, tol...)
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test sol.u[1] != bad
                    @test maximum(abs, sol.u[1] - ref.u[1]) <= 1.0e-14
                    @test isapprox(sol.u[end], ref.u[end]; rtol = 1.0e-12)
                end
            end
            hopeless = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    (du, u, p, t) -> (du[1] = -u[1]; du[2] = u[2]^2 + 1; nothing);
                    mass_matrix = Diagonal([1.0, 0.0]), jac_prototype = sparse(ones(2, 2)),
                ), [1.0, 0.0], (0.0, 1.0),
            )
            for init in (brown, shampine)
                sol = SciMLBase.solve(
                    hopeless, PETScDiffEq.TSImplicit("bdf"; comm); initializealg = init,
                )
                @test sol.retcode == SciMLBase.ReturnCode.InitialFailure
                @test sol.t == [0.0]
            end
        end

        @testset "initialize_dae! runs it on the integrator's state" begin
            calls = Ref(0)
            counted!(du, u, p, t) = (calls[] += 1; du[1] = -u[1]; nothing)
            integ = SciMLBase.init(
                SciMLBase.ODEProblem(counted!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp"),
            )
            made = calls[]
            SciMLBase.initialize_dae!(integ)
            SciMLBase.initialize_dae!(integ, brown)
            SciMLBase.initialize_dae!(integ, shampine)
            @test integ.u == [1.0]
            @test calls[] == made
            SciMLBase.terminate!(integ)
            integ = SciMLBase.init(
                SciMLBase.SecondOrderODEProblem(
                    (dv, v, u, p, t) -> (dv[1] = -u[1]; nothing), [0.0], [1.0], (0.0, 1.0),
                ), PETScDiffEq.TSRK("5dp"),
            )
            SciMLBase.initialize_dae!(integ)
            @test integ.u.x[1] == [0.0] && integ.u.x[2] == [1.0]
            SciMLBase.terminate!(integ)

            alg = PETScDiffEq.TSImplicit("bdf")
            integ = SciMLBase.init(mass(good), alg; tol...)
            SciMLBase.set_u!(integ, bad)
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.initialize_dae!(integ)
            @test integ.u == bad
            SciMLBase.initialize_dae!(integ, brown)
            @test integ.u[1:2] == bad[1:2]
            @test abs(sum(integ.u) - 1) <= 1.0e-10
            fixed = copy(integ.u)
            SciMLBase.initialize_dae!(integ)
            @test integ.u == fixed
            SciMLBase.set_u!(integ, bad)
            SciMLBase.initialize_dae!(integ, DiffEqBase.ShampineCollocationInit(1.0e-2))
            ref = SciMLBase.solve(
                mass(bad), alg; initializealg = DiffEqBase.ShampineCollocationInit(1.0e-2),
                maxiters = 1, tol...,
            )
            @test integ.u == ref.u[1]
            SciMLBase.set_u!(integ, bad)
            SciMLBase.initialize_dae!(integ, shampine)
            ref = SciMLBase.solve(
                mass(bad), alg; initializealg = DiffEqBase.ShampineCollocationInit(integ.dt / 5),
                maxiters = 1, tol...,
            )
            @test integ.u == ref.u[1]
            SciMLBase.terminate!(integ)
        end

        Sys.WORD_SIZE == 64 && @testset "the solve goes on from the initialized state" begin
            for (prob, from, alg) in (
                    (mass(good), mass(bad), PETScDiffEq.TSImplicit("bdf")),
                    (dae(good, [-0.04, 0.04, 0.0]), dae(bad, zeros(3)), PETScDiffEq.TSDAE()),
                )
                integ = SciMLBase.init(prob, alg; tol...)
                SciMLBase.set_u!(integ, bad)
                SciMLBase.initialize_dae!(integ, brown)
                ref = SciMLBase.solve(from, alg; initializealg = brown, tol...)
                @test integ.u == ref.u[1]
                sol = SciMLBase.solve!(integ)
                @test sol.t == ref.t
                @test sol.u[2:end] == ref.u[2:end]
            end
        end

        Sys.WORD_SIZE == 64 && @testset "a callback's change is initialized again" begin
            once() = (fired = Ref(false); (u, t, integ) -> t >= 1.0 && !fired[] && (fired[] = true))
            unbalance!(integ) = (integ.u[3] += 0.1; nothing)
            keep!(integ) = (unbalance!(integ); SciMLBase.derivative_discontinuity!(integ, false))
            crossing(affect; kw...) =
                SciMLBase.ContinuousCallback((u, t, integ) -> u[1] - 0.9, affect; kw...)
            for (prob, alg) in (
                    (mass(good), PETScDiffEq.TSRosW()),
                    (mass(good), PETScDiffEq.TSImplicit("bdf")),
                    (dae(good, [-0.04, 0.04, 0.0]), PETScDiffEq.TSDAE()),
                )
                @test_throws SciMLBase.CheckInitFailureError SciMLBase.solve(
                    prob, alg; callback = SciMLBase.DiscreteCallback(once(), unbalance!), tol...,
                )
                @test_throws SciMLBase.CheckInitFailureError SciMLBase.solve(
                    prob, alg; initializealg = brown, tol...,
                    callback = SciMLBase.DiscreteCallback(
                        once(), unbalance!; initializealg = SciMLBase.CheckInit(),
                    ),
                )
                @test_throws SciMLBase.CheckInitFailureError SciMLBase.solve(
                    prob, alg; callback = crossing(unbalance!), tol...,
                )
                left = SciMLBase.solve(
                    prob, alg; tol...,
                    callback = SciMLBase.DiscreteCallback(once(), integ -> nothing),
                )
                @test left.retcode == SciMLBase.ReturnCode.Success
                sol = SciMLBase.solve(
                    prob, alg; callback = SciMLBase.DiscreteCallback(once(), unbalance!),
                    initializealg = brown, tol...,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test isapprox(sol.u[end], left.u[end]; rtol = 1.0e-12)
                own = SciMLBase.solve(
                    prob, alg; tol...,
                    callback = SciMLBase.DiscreteCallback(
                        once(), unbalance!; initializealg = brown,
                    ),
                )
                @test own.u == sol.u
                kept = SciMLBase.solve(
                    prob, alg; callback = SciMLBase.DiscreteCallback(once(), keep!), tol...,
                )
                i = findfirst(>=(1.0), kept.t)
                @test kept.t[i + 1] == kept.t[i]
                @test kept.u[i + 1][3] - kept.u[i][3] ≈ 0.1
                left = SciMLBase.solve(prob, alg; callback = crossing(integ -> nothing), tol...)
                sol = SciMLBase.solve(
                    prob, alg; callback = crossing(unbalance!; initializealg = brown), tol...,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test isapprox(sol.u[end], left.u[end]; rtol = 1.0e-12)
            end
            sol = SciMLBase.solve(
                mass(good), PETScDiffEq.TSRosW(); initializealg = brown, tol...,
                callback = SciMLBase.DiscreteCallback(once(), unbalance!),
            )
            @test abs(sol.u[end][1] - 0.6172348797607147) < 1.0e-7
            sol = SciMLBase.solve(
                dae(good, [-0.04, 0.04, 0.0]), PETScDiffEq.TSDAE(); initializealg = brown, tol...,
                callback = SciMLBase.DiscreteCallback(once(), unbalance!),
            )
            @test abs(sol.u[end][1] - 0.6172348717109677) < 1.0e-5

            alg = PETScDiffEq.TSImplicit("bdf")
            nothing_later = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            flagged = SciMLBase.DiscreteCallback(
                nothing_later.condition, nothing_later.affect!;
                initialize = (cb, u, t, integ) -> (
                    unbalance!(integ); SciMLBase.derivative_discontinuity!(integ, true)
                ),
            )
            quiet = SciMLBase.DiscreteCallback(
                nothing_later.condition, nothing_later.affect!;
                initialize = (cb, u, t, integ) -> unbalance!(integ),
            )
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.init(
                mass(good), alg; callback = flagged, tol...,
            )
            integ = SciMLBase.init(
                mass(good), alg; callback = flagged, initializealg = brown, tol...,
            )
            @test integ.u[1:2] == good[1:2]
            @test abs(integ.u[3]) <= 1.0e-10
            SciMLBase.step!(integ)
            SciMLBase.reinit!(integ)
            @test abs(integ.u[3]) <= 1.0e-10
            SciMLBase.terminate!(integ)
            integ = SciMLBase.init(
                mass(good), alg; callback = quiet, initializealg = brown, tol...,
            )
            @test integ.u == [1.0, 0.0, 0.1]
            SciMLBase.terminate!(integ)
        end

        @testset "a callback's change of p is initialized against, and a failure ends there" begin
            rooted!(du, u, p, t) = (du[1] = -u[1]; du[2] = u[2]^2 - p[1]; nothing)
            prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(rooted!; mass_matrix = Diagonal([1.0, 0.0])), [1.0, 1.0],
                (0.0, 1.0), [1.0],
            )
            flip = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t == 0.5, integ -> (integ.p = [-1.0]; nothing),
            )
            alg = PETScDiffEq.TSImplicit("bdf")
            @test_throws SciMLBase.CheckInitFailureError SciMLBase.solve(
                prob, alg; callback = flip, tstops = [0.5],
            )
            sol = SciMLBase.solve(
                prob, alg; callback = flip, tstops = [0.5], initializealg = brown,
            )
            @test sol.retcode == SciMLBase.ReturnCode.InitialFailure
            @test sol.t[end] == 0.5
            @test abs(sol.u[end][1] - exp(-0.5)) < 1.0e-3
            @test sol.u[end][2] == 1.0

            residual!(r, du, u, p, t) =
                (r[1] = du[1] + p[1] * u[1]; r[2] = u[1] + p[1] * u[2] - 1; nothing)
            prob = SciMLBase.DAEProblem(
                residual!, [-1.0, 1.0], [1.0, 0.0], (0.0, 1.0), [1.0];
                differential_vars = [true, false],
            )
            flip = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t == 0.5, integ -> (integ.p = [2.0]; nothing),
            )
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSDAE("bdf"); callback = flip, tstops = [0.5],
                initializealg = brown,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            i = findfirst(==(0.5), sol.t)
            @test sol.t[i + 1] == 0.5
            @test sol.u[i + 1][1] == sol.u[i][1]
            @test abs(sol.u[i][2] - (1 - sol.u[i][1])) <= 1.0e-8
            @test abs(sol.u[i + 1][2] - (1 - sol.u[i + 1][1]) / 2) <= 1.0e-8
        end
    end

    struct CannedInit
        calls::Base.RefValue{Int}
    end
    function SciMLBase.solve(
            prob::Union{SciMLBase.NonlinearProblem, SciMLBase.NonlinearLeastSquaresProblem},
            alg::CannedInit; abstol, reltol,
        )
        alg.calls[] += 1
        return SciMLBase.build_solution(
            prob, alg, [1.0], [0.0]; retcode = SciMLBase.ReturnCode.Success,
        )
    end

    @testset "OverrideInit solves the problem's own initialization" begin
        # u[2]^3 + u[2] = u[1] holds the algebraic variable, and p[1] * u[1] = 3 the parameter.
        function rhs!(du, u, p, t)
            du[1] = -p[1] * u[1]
            du[2] = u[2]^3 + u[2] - u[1]
            return nothing
        end
        residual!(r, du, u, p, t) = (rhs!(r, u, p, t); r[1] -= du[1]; nothing)
        cubic!(r, z, q) = (r[1] = z[1]^3 + z[1] - q[1]; nothing)
        both!(r, z, q) = (cubic!(r, z, q); r[2] = z[2] * q[1] - 3; nothing)
        state(valp) = valp isa SciMLBase.DEIntegrator ? valp.u : valp.u0
        sync!(iprob, valp) = (iprob.p[1] = state(valp)[1]; nothing)
        data(iprob; update = sync!, map = sol -> [sol.prob.p[1], sol.u[1]], pmap = nothing) =
            SciMLBase.OverrideInitData(iprob, update, map, pmap)
        state_only = data(SciMLBase.NonlinearProblem(cubic!, [0.0], [0.0]))
        with_p = data(
            SciMLBase.NonlinearProblem(both!, [0.0, 0.0], [0.0]); pmap = (valp, sol) -> [sol.u[2]],
        )
        mass(d; u0 = [2.0, 0.0]) = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(
                rhs!; mass_matrix = Diagonal([1.0, 0.0]), initialization_data = d,
            ), u0, (0.0, 1.0), [1.0],
        )
        dae(d) = SciMLBase.DAEProblem(
            SciMLBase.DAEFunction(residual!; initialization_data = d), [-2.0, 0.0], [2.0, 0.0],
            (0.0, 1.0), [1.0]; differential_vars = [true, false],
        )
        plain(d) = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(
                (du, u, p, t) -> (du[1] = -p[1] * u[1]; du[2] = -u[2]; nothing);
                initialization_data = d,
            ), [2.0, 0.0], (0.0, 1.0), [1.0],
        )
        function root(a)
            z = 1.0
            for _ in 1:60
                z -= (z^3 + z - a) / (3z^2 + 1)
            end
            return z
        end
        tol = (abstol = 1.0e-10, reltol = 1.0e-10)
        own = SciMLBase.OverrideInit()
        rosw, bdf = PETScDiffEq.TSRosW(), PETScDiffEq.TSImplicit("bdf")
        at_half() = SciMLBase.DiscreteCallback(
            (u, t, integ) -> t == 0.5, integ -> (integ.u[1] = 1.0; nothing),
        )

        @testset "the default runs it and the solve takes its state and parameters" begin
            for (prob, alg) in ((mass(with_p), bdf), (dae(with_p), PETScDiffEq.TSDAE()))
                integ = SciMLBase.init(prob, alg; tol...)
                @test integ.u ≈ [2.0, 1.0] atol = 1.0e-12
                @test integ.p ≈ [1.5] atol = 1.0e-12
                @test integ.sol.prob.p == integ.p
                @test integ.sol.prob.u0 == [2.0, 0.0]
                @test prob.p == [1.0]
                SciMLBase.terminate!(integ)
                @test SciMLBase.solve(prob, alg; initializealg = own, maxiters = 1).u[1] ≈
                    [2.0, 1.0] atol = 1.0e-8
                @test_throws SciMLBase.CheckInitFailureError SciMLBase.solve(
                    prob, alg; initializealg = SciMLBase.CheckInit(),
                )
                @test SciMLBase.solve(
                    prob, alg; initializealg = SciMLBase.NoInit(), maxiters = 1,
                ).u[1] == [2.0, 0.0]
            end
            sol = SciMLBase.solve(plain(with_p), PETScDiffEq.TSRK("5dp"); maxiters = 1)
            @test sol.u[1] ≈ [2.0, 1.0] atol = 1.0e-8
            @test sol.prob.p ≈ [1.5] atol = 1.0e-8
        end

        Sys.WORD_SIZE == 64 && @testset "the solve goes on from what it gives" begin
            for (prob, alg, err) in (
                    (mass(with_p), rosw, 1.0e-9), (mass(with_p), bdf, 1.0e-6),
                    (dae(with_p), PETScDiffEq.TSDAE(), 1.0e-6),
                )
                sol = SciMLBase.solve(prob, alg; tol...)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.prob.p ≈ [1.5] atol = 1.0e-12
                a = 2exp(-1.5)
                @test sol.u[end] ≈ [a, root(a)] atol = err
                @test SciMLBase.solve(prob, alg; initializealg = own, tol...).u == sol.u
            end
            sol = SciMLBase.solve(mass(state_only), rosw; tol...)
            @test sol.u[1] ≈ [2.0, 1.0] atol = 1.0e-12
            @test sol.prob.p == [1.0]
            a = 2exp(-1)
            @test sol.u[end] ≈ [a, root(a)] atol = 1.0e-9
        end

        Sys.WORD_SIZE == 64 && @testset "initialize_dae!, reinit! and callbacks run it again" begin
            for (prob, alg) in ((mass(with_p), rosw), (dae(with_p), PETScDiffEq.TSDAE()))
                integ = SciMLBase.init(prob, alg; tol...)
                SciMLBase.step!(integ)
                integ.u[1] = 1.0
                SciMLBase.initialize_dae!(integ)
                @test integ.u ≈ [1.0, root(1.0)] atol = 1.0e-12
                @test integ.p ≈ [3.0] atol = 1.0e-12
                @test integ.sol.prob.p == integ.p
                SciMLBase.reinit!(integ, [3.0, 0.0])
                @test integ.u ≈ [3.0, root(3.0)] atol = 1.0e-12
                @test integ.p ≈ [1.0] atol = 1.0e-12
                @test integ.sol.prob.p == integ.p
                SciMLBase.reinit!(integ, [3.0, 0.0]; reinit_dae = false)
                @test integ.u == [3.0, 0.0]
                SciMLBase.reinit!(integ, [3.0, 0.0])
                sol = SciMLBase.solve!(integ)
                @test sol.prob.p ≈ [1.0] atol = 1.0e-12
                a = 3exp(-1)
                @test sol.u[end] ≈ [a, root(a)] atol = 2.0e-6
                sol = SciMLBase.solve(prob, alg; callback = at_half(), tstops = [0.5], tol...)
                i = findlast(==(0.5), sol.t)
                @test sol.u[i] ≈ [1.0, root(1.0)] atol = 1.0e-12
                @test sol.prob.p ≈ [3.0] atol = 1.0e-12
                retune = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> t == 0.5, integ -> (integ.p = [9.0]; nothing),
                )
                sol = SciMLBase.solve(prob, alg; callback = retune, tstops = [0.5], tol...)
                @test sol.prob.p ≈ [3 / sol.u[findlast(==(0.5), sol.t)][1]] atol = 1.0e-10
                checked = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> t == 0.5, integ -> (integ.u[1] = 1.0; nothing);
                    initializealg = SciMLBase.CheckInit(),
                )
                @test_throws SciMLBase.CheckInitFailureError SciMLBase.solve(
                    prob, alg; callback = checked, tstops = [0.5], tol...,
                )
            end
            integ = SciMLBase.init(mass(state_only), rosw; tol...)
            integ.p = [7.0]
            SciMLBase.initialize_dae!(integ, own)
            @test integ.sol.prob.p == [7.0]
            SciMLBase.terminate!(integ)
            sol = SciMLBase.solve(
                plain(with_p), PETScDiffEq.TSRK("5dp"); callback = at_half(), tstops = [0.5],
                tol...,
            )
            i = findlast(==(0.5), sol.t)
            @test sol.u[i][2] == sol.u[i - 1][2]
            @test sol.prob.p ≈ [1.5] atol = 1.0e-12
        end

        @testset "the systems SNES takes" begin
            started(d; kw...) = SciMLBase.solve(mass(d), bdf; maxiters = 1, kw...)
            oop = SciMLBase.NonlinearProblem((z, q) -> [z[1]^3 + z[1] - q[1]], [0.0], [0.0])
            @test started(data(oop)).u[1] ≈ [2.0, 1.0] atol = 1.0e-8
            squares(m, n) = SciMLBase.NonlinearLeastSquaresProblem(
                SciMLBase.NonlinearFunction(
                    (r, z, q) -> (fill!(r, 0); cubic!(r, z, q)); resid_prototype = zeros(m),
                ), zeros(n), [0.0],
            )
            @test started(data(squares(1, 1))).u[1] ≈ [2.0, 1.0] atol = 1.0e-8
            bare = SciMLBase.NonlinearLeastSquaresProblem(cubic!, [0.0], [0.0])
            @test started(data(bare)).u[1] ≈ [2.0, 1.0] atol = 1.0e-8
            @test started(data(oop); abstol = [1.0e-9, 1.0e-8]).u[1] ≈ [2.0, 1.0] atol = 1.0e-8
            @test started(
                data(oop); initializealg = SciMLBase.OverrideInit(; abstol = 1.0e-12, reltol = 1.0e-12),
            ).u[1] ≈ [2.0, 1.0] atol = 1.0e-12
            none = SciMLBase.NonlinearProblem((z, q) -> nothing, nothing, [0.0])
            sol = started(
                data(none; map = sol -> [sol.p[1], 1.0], pmap = (valp, sol) -> [4.0]);
                initializealg = SciMLBase.OverrideInit(),
            )
            @test sol.u[1] == [2.0, 1.0]
            @test sol.prob.p == [4.0]
            held = SciMLBase.NonlinearLeastSquaresProblem(
                SciMLBase.NonlinearFunction(
                    (r, z, q) -> (r[1] = q[1] - 2; nothing); resid_prototype = zeros(1),
                ), nothing, [0.0],
            )
            @test started(data(held; map = sol -> [2.0, 1.0])).u[1] == [2.0, 1.0]
            sol = SciMLBase.solve(mass(data(held; map = sol -> [3.0, 1.0]); u0 = [3.0, 0.0]), bdf)
            @test sol.retcode == SciMLBase.ReturnCode.InitialFailure
            single = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    rhs!; mass_matrix = Diagonal(Float32[1, 0]),
                    initialization_data = data(
                        SciMLBase.NonlinearProblem(cubic!, Float32[0], Float32[0]),
                    ),
                ), Float32[2, 0], (0.0f0, 1.0f0), Float32[1],
            )
            @test SciMLBase.solve(single, bdf; maxiters = 1).u[1] ≈ Float32[2, 1] atol = 1.0e-5
            narrow = SciMLBase.solve(
                SciMLBase.NonlinearProblem(cubic!, Float32[0], Float32[2]),
                PETScDiffEq.PETScSNES(PETScDiffEq._petsclib(Float64)); abstol = 1.0e-6,
            )
            @test narrow.retcode == SciMLBase.ReturnCode.Success && narrow.u isa Vector{Float32}
            @test narrow.u ≈ Float32[1] atol = 1.0e-5
        end

        @testset "a solve that fails is InitialFailure" begin
            noroot = data(
                SciMLBase.NonlinearProblem((r, z, q) -> (r[1] = z[1]^2 + 1; nothing), [0.0], [0.0]),
            )
            sol = SciMLBase.solve(mass(noroot), bdf)
            @test sol.retcode == SciMLBase.ReturnCode.InitialFailure
            @test sol.t == [0.0]
            integ = SciMLBase.init(mass(noroot), rosw)
            @test integ.sol.retcode == SciMLBase.ReturnCode.InitialFailure
            @test SciMLBase.solve!(integ).retcode == SciMLBase.ReturnCode.InitialFailure
        end

        @testset "a given nlsolve solves it instead" begin
            over = SciMLBase.NonlinearLeastSquaresProblem(
                SciMLBase.NonlinearFunction(
                    (r, z, q) -> (fill!(r, 0); cubic!(r, z, q)); resid_prototype = zeros(2),
                ), [0.0], [0.0],
            )
            for iprob in (over, SciMLBase.NonlinearProblem(cubic!, [0.0], [0.0]))
                canned = CannedInit(Ref(0))
                sol = SciMLBase.solve(
                    mass(data(iprob)), bdf; maxiters = 1,
                    initializealg = SciMLBase.OverrideInit(; nlsolve = canned),
                )
                @test canned.calls[] == 1
                @test sol.u[1] == [2.0, 1.0]
            end
        end

        @testset "what it refuses" begin
            lsq(m, n) = data(
                SciMLBase.NonlinearLeastSquaresProblem(
                    SciMLBase.NonlinearFunction(
                        (r, z, q) -> (fill!(r, 0); cubic!(r, z, q)); resid_prototype = zeros(m),
                    ), zeros(n), [0.0],
                ),
            )
            @test_throws "2 equations for 1 unknowns" SciMLBase.solve(mass(lsq(2, 1)), bdf)
            @test_throws "1 equations for 2 unknowns" SciMLBase.solve(mass(lsq(1, 2)), bdf)
            scalar = SciMLBase.OverrideInitData(
                SciMLBase.NonlinearProblem((z, q) -> z^3 + z - q[1], 0.0, [0.0]), nothing,
                nothing, nothing,
            )
            @test_throws "with a `Float64` state" SciMLBase.solve(mass(scalar), bdf)
            blocks = SciMLBase.SCCNonlinearProblem(
                [SciMLBase.NonlinearProblem(cubic!, [0.0], [0.0])], [Returns(nothing)],
            )
            haskey(Base.loaded_modules, PETScDiffEq._SCC_SOLVER) ||
                @test_throws "through SCCNonlinearSolve" SciMLBase.solve(
                mass(data(blocks; update = nothing, map = nothing)), bdf,
            )
            everywhere = PETScDiffEq.TSRK("5dp"; comm = MPI.COMM_WORLD)
            for init in (DiffEqBase.DefaultInit(), own)
                @test_throws "cannot run `OverrideInit` on a communicator" SciMLBase.solve(
                    plain(state_only), everywhere; initializealg = init,
                )
            end
            @test SciMLBase.solve(
                plain(state_only), everywhere; initializealg = SciMLBase.CheckInit(),
            ).retcode == SciMLBase.ReturnCode.Success
            adjoint(; kw...) = PETScDiffEq._discrete_adjoint(
                plain(state_only), PETScDiffEq.TSRK("4"), PETScAdjoint(); t = [1.0],
                dgdu_discrete = (out, u, p, t, i) -> (out .= u; nothing), dt = 0.01,
                adaptive = false, kw...,
            )
            for kw in ((;), (; initializealg = own))
                @test_throws "does not differentiate a problem's own initialization" adjoint(;
                    kw...,
                )
            end
            @test adjoint(; initializealg = SciMLBase.CheckInit())[1] ≈ [2exp(-2), 0.0]
        end
    end

    @testset "EnsembleProblem" begin
        function decay_p!(du, u, p, t)
            return du[1] = -p[1] * u[1]
        end
        prob = SciMLBase.ODEProblem(decay_p!, [1.0], (0.0, 1.0), [1.0])
        vary(p, ctx) = SciMLBase.remake(p; p = [Float64(ctx.sim_id)])
        eprob = SciMLBase.EnsembleProblem(prob; prob_func = vary)

        @testset "$name" for (name, alg, ens, n, tol) in (
                (
                    "explicit, serial", PETScDiffEq.TSRK("5dp"),
                    SciMLBase.EnsembleSerial(), 4, 1.0e-8,
                ),
                (
                    "explicit, threaded", PETScDiffEq.TSRK("5dp"),
                    SciMLBase.EnsembleThreads(), 4, 1.0e-8,
                ),
                (
                    "implicit, serial", PETScDiffEq.TSImplicit("bdf"),
                    SciMLBase.EnsembleSerial(), 3, 1.0e-5,
                ),
            )
            sim = SciMLBase.solve(
                eprob, alg, ens; trajectories = n, dt = 0.01, reltol = 1.0e-10,
                abstol = 1.0e-12,
            )
            @test length(sim.u) == n
            @test all(s -> s.retcode == SciMLBase.ReturnCode.Success, sim.u)
            @test all(abs(sim.u[i].u[end][1] - exp(-i)) < tol for i in 1:n)
        end
    end

    @testset "nf2 counts the explicit part of a split problem" begin
        function stiff!(du, u, p, t)
            return du[1] = -1000.0 * (u[1] - cos(t))
        end
        function forcing!(du, u, p, t)
            return du[1] = -sin(t)
        end
        split = SciMLBase.solve(
            SciMLBase.SplitODEProblem(stiff!, forcing!, [1.0], (0.0, 0.1)),
            PETScDiffEq.TSARKIMEX("3"); dt = 1.0e-3,
        )
        @test split.stats.nf > 0
        @test split.stats.nf2 > 0
        plain = SciMLBase.solve(
            SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
            dt = 0.1,
        )
        @test plain.stats.nf > 0
        @test plain.stats.nf2 == 0
    end

    @testset "stats count the linear solves and the callback condition calls" begin
        chain_prob = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(chain!; jac = chain_jac!, jac_prototype = CHAIN_PROTOTYPE),
            [1.0, 0.5, 0.25], (0.0, 1.0),
        )
        fixed = (dt = 0.05, adaptive = false)
        linear = ["-snes_type", "ksponly"]
        beuler(opts = linear; kw...) = PETScDiffEq.TSImplicit("beuler", opts; kw...)
        counts(sol) = (sol.stats.naccept, sol.stats.nsolve, sol.stats.nw, sol.stats.ncondition)
        @test counts(SciMLBase.solve(chain_prob, beuler(); fixed...)) == (20, 20, -1, 0)
        krylov = [linear..., "-ksp_type", "gmres", "-pc_type", "none"]
        @test counts(SciMLBase.solve(chain_prob, beuler(krylov); fixed...)) == (20, 20, -1, 0)
        rosw = PETScDiffEq.TSRosW("ra34pw2")
        @test counts(SciMLBase.solve(chain_prob, rosw; fixed...)) == (20, 80, -1, 0)
        @test counts(SciMLBase.solve(chain_prob, PETScDiffEq.TSRK("4"); fixed...)) == (20, 0, -1, 0)
        newton = SciMLBase.solve(chain_prob, beuler(String[]); fixed...)
        @test newton.stats.nsolve == newton.stats.nnonliniter >= 20
        @test SciMLBase.solve(chain_prob, beuler(["-snes_ksp_ew"]); fixed...).stats.nsolve == -1
        if Sys.WORD_SIZE == 64
            on_world = beuler(; comm = MPI.COMM_WORLD)
            @test counts(SciMLBase.solve(chain_prob, on_world; fixed...)) == (20, 20, -1, 0)
        end

        integ = SciMLBase.init(chain_prob, beuler(); fixed...)
        @test counts(integ.sol) == (0, 0, -1, 0)
        SciMLBase.step!(integ)
        SciMLBase.step!(integ)
        @test counts(integ.sol) == (2, 2, -1, 0)
        SciMLBase.reinit!(integ)
        @test counts(integ.sol) == (0, 0, -1, 0)
        @test counts(SciMLBase.solve!(integ)) == (20, 20, -1, 0)

        asked = [0, 0, 0]
        each_step = SciMLBase.DiscreteCallback(
            (u, t, integ) -> (asked[1] += 1; false), integ -> nothing,
        )
        crossing = SciMLBase.ContinuousCallback(
            (u, t, integ) -> (asked[2] += 1; u[1] - 0.8), integ -> nothing,
        )
        crossings = SciMLBase.VectorContinuousCallback(
            (out, u, t, integ) -> (asked[3] += 1; out[1] = u[1] - 0.7; out[2] = u[2] - 9.0; nothing),
            (integ, i) -> nothing, 2,
        )
        watched = SciMLBase.CallbackSet(crossing, crossings, each_step)
        integ = SciMLBase.init(chain_prob, beuler(); callback = watched, fixed...)
        SciMLBase.step!(integ)
        @test integ.sol.stats.ncondition == sum(asked) > 2
        sol = SciMLBase.solve!(integ)
        @test asked[1] == sol.stats.naccept
        @test minimum(asked[2:3]) > sol.stats.naccept
        @test sol.stats.ncondition == sum(asked)
        @test sol.stats.nsolve == sol.stats.naccept
    end

    @testset "a solver failure is a retcode, a misuse is an error" begin
        blow!(du, u, p, t) = (du[1] = -1.0e6 * (exp(u[1]) - 1); nothing)
        stiff = SciMLBase.ODEProblem(blow!, [20.0], (0.0, 10.0))
        sol = SciMLBase.solve(
            stiff, PETScDiffEq.TSImplicit("beuler"); dt = 1.0, adaptive = false,
        )
        @test sol.retcode == SciMLBase.ReturnCode.ConvergenceFailure
        @test sol.t[end] > 0.0
        @test allunique(sol.t)

        integ = SciMLBase.init(
            stiff, PETScDiffEq.TSImplicit("beuler"); dt = 1.0, adaptive = false,
        )
        n = 0
        while !SciMLBase.done(integ) && n < 100
            SciMLBase.step!(integ)
            n += 1
        end
        @test SciMLBase.done(integ)
        @test integ.sol.retcode == SciMLBase.ReturnCode.ConvergenceFailure
        @test allunique(integ.sol.t)

        @testset "an overflow is Unstable, not a raised error" begin
            square!(du, u, p, t) = (du[1] = u[1]^2; nothing)
            huge = SciMLBase.ODEProblem(square!, [1.0e100], (0.0, 1.0))
            over = @test_logs (:warn, r"floating point exception") SciMLBase.solve(
                huge, PETScDiffEq.TSRK("5dp"); dt = 1.0e-102,
            )
            @test over.retcode == SciMLBase.ReturnCode.Unstable
            @test over.t[end] < 1.0
            @test all(isfinite, over.u[end])
            @test allunique(over.t)
        end

        @testset "a step that raises leaves the last accepted state at the end" begin
            breaks = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = t > 0.5 ? NaN : -u[1]; nothing), [1.0], (0.0, 1.0),
            )
            never = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            for kw in ((;), (; save_everystep = false))
                plain = @test_logs (:warn, r"floating point exception") SciMLBase.solve(
                    breaks, PETScDiffEq.TSRK("5dp"); kw...,
                )
                stepped = @test_logs (:warn, r"floating point exception") SciMLBase.solve(
                    breaks, PETScDiffEq.TSRK("5dp"); callback = never, kw...,
                )
                @test plain.retcode == SciMLBase.ReturnCode.Unstable
                @test all(isfinite, plain.u[end])
                @test plain.t == stepped.t
                @test plain.u == stepped.u
            end
        end

        @testset "a NaN trial step is taken again smaller, as OrdinaryDiffEq takes it" begin
            breaks = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = t > 0.5 ? NaN : -u[1]; nothing), [1.0], (0.0, 1.0),
            )
            never = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            for alg in (PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRK("3bs")),
                    kw in ((;), (; saveat = 0.05))
                plain = @test_logs (:warn, r"floating point exception") SciMLBase.solve(
                    breaks, alg; kw...,
                )
                stepped = @test_logs (:warn, r"floating point exception") SciMLBase.solve(
                    breaks, alg; callback = never, kw...,
                )
                @test plain.retcode == SciMLBase.ReturnCode.Unstable
                @test isempty(kw) ? 0.5 - plain.t[end] < 1.0e-12 :
                    all(in(0.0:0.05:0.5), plain.t) && plain.t[end] >= 0.45
                @test maximum(abs(u[1] - exp(-t)) for (t, u) in zip(plain.t, plain.u)) < 2.0e-4
                @test plain.stats.nreject > 10
                @test plain.t == stepped.t
                @test plain.u == stepped.u
                @test stepped.retcode == plain.retcode
                @test (stepped.stats.naccept, stepped.stats.nreject, stepped.stats.nf) ==
                    (plain.stats.naccept, plain.stats.nreject, plain.stats.nf)
            end
            for run in (
                    () -> SciMLBase.solve(breaks, PETScDiffEq.TSRK("5dp"); dtmin = 0.01),
                    () -> SciMLBase.solve(
                        breaks, PETScDiffEq.TSRK("5dp"); dtmin = 0.01, callback = never,
                    ),
                )
                floored = @test_logs (:warn, r"floating point exception") run()
                @test floored.retcode == SciMLBase.ReturnCode.DtLessThanMin
                @test 0.45 < floored.t[end] < 0.5
            end
        end

        @testset "a fixed-step solve stops at its first state that is not finite" begin
            breaks = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = t > 0.5 ? NaN : -u[1]; nothing), [1.0], (0.0, 1.0),
            )
            never = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            for (alg, kw) in (
                    (PETScDiffEq.TSRK("4"), (; dt = 0.01)),
                    (PETScDiffEq.TSRK("5dp"), (; dt = 0.01, adaptive = false)),
                    (PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"]), (; dt = 0.01)),
                )
                plain = SciMLBase.solve(breaks, alg; kw...)
                stepped = SciMLBase.solve(breaks, alg; callback = never, kw...)
                @test plain.retcode == SciMLBase.ReturnCode.Unstable
                @test plain.t[end] < 0.51
                @test !all(isfinite, plain.u[end])
                @test all(u -> all(isfinite, u), plain.u[1:(end - 1)])
                @test plain.t == stepped.t
                @test isequal(plain.u, stepped.u)
                @test plain.stats.naccept == stepped.stats.naccept
            end
        end

        @testset "a step too small to move t is Unstable through the integrator" begin
            square!(du, u, p, t) = (du[1] = u[1]^2; nothing)
            runaway = SciMLBase.ODEProblem(square!, [1.0], (0.0, 2.0))
            never = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            sol = @test_logs (:warn, r"floating point spacing") SciMLBase.solve(
                runaway, PETScDiffEq.TSRK("5dp"); callback = never,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Unstable
            @test all(isfinite, sol.u[end])
            integ = SciMLBase.init(runaway, PETScDiffEq.TSRK("5dp"))
            @test_logs (:warn, r"floating point spacing") SciMLBase.solve!(integ)
            @test integ.sol.retcode == SciMLBase.ReturnCode.Unstable
            @test integ.sol.t == sol.t
            plain = @test_logs (:warn, r"floating point spacing") SciMLBase.solve(
                runaway, PETScDiffEq.TSRK("5dp"),
            )
            @test plain.retcode == SciMLBase.ReturnCode.Unstable
            @test plain.t == sol.t
            @test plain.u == sol.u
            for (alg, tol) in (
                    (PETScDiffEq.TSRK("5dp"), 1.0e-6), (PETScDiffEq.TSRK("5dp"), 1.0e-10),
                    (PETScDiffEq.TSRK("3bs"), 1.0e-6), (PETScDiffEq.TSRosW(), 1.0e-6),
                )
                kw = (; abstol = tol, reltol = tol, verbose = false)
                plain = SciMLBase.solve(runaway, alg; kw...)
                stepped = SciMLBase.solve(runaway, alg; callback = never, kw...)
                @test plain.t == stepped.t
                @test (plain.stats.naccept, plain.stats.nreject, plain.stats.nf) ==
                    (stepped.stats.naccept, stepped.stats.nreject, stepped.stats.nf)
                @test plain.stats.naccept == length(plain.t) - 1
            end
        end

        @testset "a fixed-step solve stops at a step too small to move t" begin
            late = SciMLBase.ODEProblem(decay!, [1.0], (1.0, 2.0))
            never = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            for (alg, kw) in (
                    (PETScDiffEq.TSRK("4"), (; dt = 1.0e-17)),
                    (PETScDiffEq.TSRK("5dp"), (; dt = 1.0e-17, adaptive = false)),
                    (PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"]), (; dt = 1.0e-17)),
                )
                plain = @test_logs (:warn, r"floating point spacing") SciMLBase.solve(
                    late, alg; kw...,
                )
                stepped = @test_logs (:warn, r"floating point spacing") SciMLBase.solve(
                    late, alg; callback = never, kw...,
                )
                for sol in (plain, stepped)
                    @test sol.retcode == SciMLBase.ReturnCode.Unstable
                    @test sol.t == [1.0]
                    @test (sol.stats.naccept, sol.stats.nreject) == (0, 1)
                end
            end
        end

        @testset "the post-step check allocates nothing" begin
            integ = SciMLBase.init(
                SciMLBase.ODEProblem(decay!, [1.0], (0.0, 10.0)), PETScDiffEq.TSRK("5dp"),
            )
            SciMLBase.step!(integ)
            h = integ.h
            h.ctx.halt_stalled = true
            PETScDiffEq._set_post_step!(h.petsclib, h.ts, h.ctx)
            ptr = h.ts.ptr
            function post(n)
                for _ in 1:n
                    PETScDiffEq._post_step!(ptr)
                end
                return nothing
            end
            post(2)
            @test (@allocated post(100)) == 0
            @test !h.ctx.stalled
            @test SciMLBase.solve!(integ).retcode == SciMLBase.ReturnCode.Success
            @test !haskey(PETScDiffEq.POST_STEP_CTX, ptr)
        end

        @testset "verbose silences the warning for a solve that ends early" begin
            breaks = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = t > 0.5 ? NaN : -u[1]; nothing), [1.0], (0.0, 1.0),
            )
            runaway = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = u[1]^2; nothing), [1.0], (0.0, 2.0),
            )
            never = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            logging = PETScDiffEq.DiffEqBase.SciMLLogging
            quiet = PETScDiffEq.DiffEqBase.DEVerbosity(logging.None())
            for pr in (breaks, runaway), kw in ((;), (; callback = never)),
                    verbose in (
                        false, logging.None(), quiet,
                        PETScDiffEq.DiffEqBase.DEVerbosity(instability = logging.Silent()),
                    )
                sol = @test_logs min_level = Logging.Warn SciMLBase.solve(
                    pr, PETScDiffEq.TSRK("5dp"); verbose, kw...,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Unstable
            end
            for verbose in (false, logging.None(), quiet)
                integ = SciMLBase.init(breaks, PETScDiffEq.TSRK("5dp"))
                integ.opts.verbose = verbose
                @test_logs min_level = Logging.Warn SciMLBase.solve!(integ)
            end
            # An implicit solve gives up on its Newton solve, which it reports at the end.
            bdf = PETScDiffEq.TSImplicit("bdf")
            for verbose in (false, quiet)
                sol = @test_logs min_level = Logging.Warn SciMLBase.solve(breaks, bdf; verbose)
                @test sol.retcode == SciMLBase.ReturnCode.Unstable
                integ = SciMLBase.init(breaks, bdf)
                integ.opts.verbose = verbose
                @test_logs min_level = Logging.Warn SciMLBase.solve!(integ)
            end
        end

        @testset "a solve that ends early keeps its last step as the one just taken" begin
            breaks = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = t > 0.5 ? NaN : -u[1]; nothing), [1.0], (0.0, 1.0),
            )
            runaway = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = u[1]^2; nothing), [1.0], (0.0, 2.0),
            )
            for (pr, alg) in (
                    (breaks, PETScDiffEq.TSRK("5dp")), (breaks, PETScDiffEq.TSRosW()),
                    (breaks, PETScDiffEq.TSImplicit("bdf")), (runaway, PETScDiffEq.TSRK("5dp")),
                )
                integ = SciMLBase.init(pr, alg; verbose = false)
                SciMLBase.solve!(integ)
                @test integ.sol.retcode != SciMLBase.ReturnCode.Success
                @test integ.sol.t[(end - 1):end] == [integ.tprev, integ.t]
                @test integ.sol.u[end - 1] == integ.uprev
                @test integ.t - integ.tprev == integ.dt
            end
        end

        @testset "a step below dtmin ends the solve where it still holds" begin
            fast!(du, u, p, t) = (du[1] = -50.0 * u[1]; nothing)
            quick = SciMLBase.ODEProblem(fast!, [1.0], (0.0, 1.0))
            kw = (; dt = 0.01, dtmin = 0.1, abstol = 1.0e-10, reltol = 1.0e-10)
            floored = SciMLBase.solve(quick, PETScDiffEq.TSRK("5dp"); kw...)
            @test floored.retcode == SciMLBase.ReturnCode.DtLessThanMin
            @test 0.0 < floored.t[end] < 1.0
            @test abs(floored.u[end][1] - exp(-50 * floored.t[end])) < 1.0e-6

            stepped = SciMLBase.init(quick, PETScDiffEq.TSRK("5dp"); kw...)
            n = 0
            while !SciMLBase.done(stepped) && n < 100
                SciMLBase.step!(stepped)
                n += 1
            end
            @test SciMLBase.done(stepped)
            @test stepped.sol.retcode == SciMLBase.ReturnCode.DtLessThanMin
            @test stepped.sol.t[end] < 1.0
            @test stepped.sol.t[end] == floored.t[end]

            line = SciMLBase.ODEProblem((du, u, p, t) -> (du[1] = 1.0; nothing), [0.0], (0.0, 1.0))
            for kw in ((; dt = 0.95), (; dt = 0.45, tstops = [0.5]))
                landed = SciMLBase.solve(line, PETScDiffEq.TSRK("5dp"); kw..., dtmin = 0.1)
                @test landed.retcode == SciMLBase.ReturnCode.Success
                @test landed.t[end] == 1.0
                integ = SciMLBase.init(line, PETScDiffEq.TSRK("5dp"); kw..., dtmin = 0.1)
                @test SciMLBase.solve!(integ).retcode == SciMLBase.ReturnCode.Success
            end
            fixed = SciMLBase.solve(
                quick, PETScDiffEq.TSRK("5dp"); dt = 0.3, adaptive = false, dtmin = 0.25,
            )
            @test fixed.retcode == SciMLBase.ReturnCode.Success
            @test fixed.t[end] == 1.0

            breaks = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = t > 0.7 ? NaN : -1.0; nothing), [1.0], (0.0, 1.0),
            )
            bdf = PETScDiffEq.TSImplicit("bdf", ["-ts_max_snes_failures", "2"])
            @test SciMLBase.solve(breaks, bdf; dt = 0.1, dtmin = 0.1).retcode ==
                SciMLBase.ReturnCode.DtLessThanMin
            @test SciMLBase.solve!(SciMLBase.init(breaks, bdf; dt = 0.1, dtmin = 0.1)).retcode ==
                SciMLBase.ReturnCode.DtLessThanMin

            forced = SciMLBase.solve(
                quick, PETScDiffEq.TSRK("5dp"); kw..., force_dtmin = true,
            )
            @test forced.retcode == SciMLBase.ReturnCode.Success
            @test forced.t[end] == 1.0
            @test minimum(diff(forced.t)[2:end]) > 0.9 * 0.1
            @test_logs min_level = Logging.Warn SciMLBase.solve(
                quick, PETScDiffEq.TSRK("5dp"); kw..., force_dtmin = true,
            )

            free = SciMLBase.solve(
                quick, PETScDiffEq.TSRK("5dp"); dt = 0.01, abstol = 1.0e-10, reltol = 1.0e-10,
            )
            @test free.retcode == SciMLBase.ReturnCode.Success
            @test free.t[end] == 1.0
        end

        plain = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        @test_throws Exception SciMLBase.solve(
            plain, PETScDiffEq.TSGeneric("euler"); dt = 0.01, adaptive = false,
        )

        boom!(du, u, p, t) = error("boom")
        @test_throws "boom" SciMLBase.solve(
            SciMLBase.ODEProblem(boom!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
            dt = 0.1,
        )
    end

    @testset "a failed implicit step is taken again smaller" begin
        never = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
        counts(sol) = (sol.stats.naccept, sol.stats.nreject, sol.stats.nnonlinconvfail)
        same(a, b) = a.t == b.t && a.u == b.u && a.retcode == b.retcode && counts(a) == counts(b)
        breaks = SciMLBase.ODEProblem(
            (du, u, p, t) -> (du[1] = t > 0.5 ? NaN : -u[1]; nothing), [1.0], (0.0, 1.0),
        )
        dae_breaks = SciMLBase.DAEProblem(
            (r, du, u, p, t) -> (r[1] = t > 0.5 ? NaN : du[1] + u[1]; nothing), [-1.0], [1.0],
            (0.0, 1.0),
        )
        adaptive = (
            (breaks, PETScDiffEq.TSImplicit("bdf")), (breaks, PETScDiffEq.TSRosW()),
            (breaks, PETScDiffEq.TSARKIMEX()), (dae_breaks, PETScDiffEq.TSDAE("bdf")),
        )

        @testset "a right-hand side that turns NaN" begin
            failed = (:warn, r"nonlinear solve failed at every step size")
            for (prob, alg) in adaptive
                plain = @test_logs failed SciMLBase.solve(prob, alg)
                stepped = @test_logs failed SciMLBase.solve(prob, alg; callback = never)
                @test plain.retcode == SciMLBase.ReturnCode.Unstable
                @test 0.5 - plain.t[end] < 1.0e-12
                @test maximum(abs(u[1] - exp(-t)) for (t, u) in zip(plain.t, plain.u)) < 2.0e-3
                newton = !(alg isa PETScDiffEq.TSRosW)
                @test (newton ? plain.stats.nnonlinconvfail : plain.stats.nreject) > 10
                @test (newton ? plain.stats.nreject : plain.stats.nnonlinconvfail) == 0
                @test same(plain, stepped)
            end
            for (prob, alg) in adaptive[1:3]
                floored = @test_logs failed SciMLBase.solve(prob, alg; dtmin = 0.01)
                @test floored.retcode == SciMLBase.ReturnCode.DtLessThanMin
                @test 0.45 < floored.t[end] < 0.5
                @test same(
                    floored,
                    @test_logs failed SciMLBase.solve(prob, alg; dtmin = 0.01, callback = never)
                )
                forced = @test_logs failed SciMLBase.solve(
                    prob, alg; dtmin = 0.01, force_dtmin = true,
                )
                @test forced.retcode == SciMLBase.ReturnCode.Unstable
                @test 0.45 < forced.t[end] < 0.5
                @test minimum(diff(forced.t)) > 0.9 * 0.01
            end
            nan = (du, u, p, t) -> (du[1] = NaN; nothing)
            for (_, alg) in adaptive[1:3]
                fails = map(((0.0, 1.0), (1.0, 2.0))) do span
                    sol = @test_logs failed SciMLBase.solve(
                        SciMLBase.ODEProblem(nan, [1.0], span), alg,
                    )
                    return sol.stats.nnonlinconvfail
                end
                @test fails[1] == fails[2]
            end
        end

        @testset "a Rosenbrock method whose first stage is explicit" begin
            positive = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = u[1] < 0 ? NaN : -1000 * u[1]; nothing), [1.0],
                (0.0, 1.0),
            )
            sol = SciMLBase.solve(positive, PETScDiffEq.TSRosW("assp3p3s1c"); dt = 1.0)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.stats.nreject > 1
            @test sol.stats.nnonlinconvfail == 0
            @test maximum(abs(u[1] - exp(-1000t)) for (t, u) in zip(sol.t, sol.u)) < 1.0e-3
        end

        @testset "a Newton solve that diverges at the first step" begin
            square = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = u[1]^2; nothing), [1.0], (0.0, 0.9),
            )
            dae_square = SciMLBase.DAEProblem(
                (r, du, u, p, t) -> (r[1] = du[1] - u[1]^2; nothing), [1.0], [1.0], (0.0, 0.9),
            )
            kw = (; dt = 0.5, abstol = 1.0e-10, reltol = 1.0e-8)
            for (prob, alg) in (
                    (square, PETScDiffEq.TSImplicit("bdf")), (square, PETScDiffEq.TSARKIMEX()),
                    (dae_square, PETScDiffEq.TSDAE("bdf")),
                )
                sol = SciMLBase.solve(prob, alg; kw...)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.stats.nnonlinconvfail >= 1
                @test abs(sol.u[end][1] - 10) < 1.0e-3
                @test same(sol, SciMLBase.solve(prob, alg; callback = never, kw...))
            end
        end

        @testset "a Newton matrix that is singular at the first step" begin
            linear(λ) = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du[1] = λ * u[1]; nothing), [1.0], (0.0, 2.0),
            )
            for (λ, alg) in (
                    (1 / 0.435866521508459, PETScDiffEq.TSRosW()),
                    (2.0, PETScDiffEq.TSImplicit("bdf")),
                    (4055673282236 / 1767732205903, PETScDiffEq.TSARKIMEX()),
                )
                sol = SciMLBase.solve(linear(λ), alg; dt = 1.0)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                if alg isa PETScDiffEq.TSRosW
                    @test sol.stats.nreject >= 2
                    @test sol.stats.nnonlinconvfail == 0
                else
                    @test sol.stats.nnonlinconvfail >= 1
                end
                @test abs(sol.u[end][1] / exp(2λ) - 1) < 0.05
                @test same(sol, SciMLBase.solve(linear(λ), alg; dt = 1.0, callback = never))
            end
        end

        @testset "an algebraic equation whose Jacobian vanishes" begin
            cusp = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    (du, u, p, t) -> (du[1] = 1.0; du[2] = u[2]^3 - (u[1] - 0.5); nothing);
                    mass_matrix = Diagonal([1.0, 0.0]),
                ), [0.0, -cbrt(0.5)], (0.0, 1.0),
            )
            for alg in (PETScDiffEq.TSRosW(), PETScDiffEq.TSImplicit("bdf"))
                plain, stepped = Logging.with_logger(Logging.NullLogger()) do
                    SciMLBase.solve(cusp, alg; dt = 0.1),
                        SciMLBase.solve(cusp, alg; dt = 0.1, callback = never)
                end
                @test same(plain, stepped)
                @test maximum(u -> abs(u[2] - cbrt(u[1] - 0.5)), plain.u) < 1.0e-3
                if plain.retcode == SciMLBase.ReturnCode.Success
                    @test plain.t[end] == 1.0
                else
                    @test plain.retcode == SciMLBase.ReturnCode.Unstable
                    @test 0.5 - plain.t[end] < 1.0e-8
                    @test_logs (:warn, r"ends here") SciMLBase.solve(cusp, alg; dt = 0.1)
                end
            end
        end

        @testset "an error estimate that overflows" begin
            decay = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
            kw = (;
                dt = 1.0, abstol = 1.0e-160, reltol = 1.0e-160, dtmin = 0.01, force_dtmin = true,
            )
            for alg in (PETScDiffEq.TSRosW(), PETScDiffEq.TSARKIMEX())
                sol = @test_logs SciMLBase.solve(decay, alg; kw...)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.stats.nreject >= 2
                @test maximum(abs(u[1] - exp(-t)) for (t, u) in zip(sol.t, sol.u)) < 1.0e-7
                @test same(sol, SciMLBase.solve(decay, alg; callback = never, kw...))
            end
            bdf = @test_logs (:warn, r"floating point exception") SciMLBase.solve(
                SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0e-4)),
                PETScDiffEq.TSImplicit("bdf"); abstol = 1.0e-200, reltol = 1.0e-200,
            )
            @test bdf.retcode == SciMLBase.ReturnCode.Unstable
            @test bdf.t[end] > 0
        end

        @testset "a fixed-step solve ends at its first failed Newton or linear solve" begin
            kw = (; dt = 0.1, adaptive = false)
            for alg in (
                    PETScDiffEq.TSIRK(), PETScDiffEq.TSImplicit("beuler"),
                    PETScDiffEq.TSImplicit("theta", 0.7), PETScDiffEq.TSImplicit("bdf"),
                    PETScDiffEq.TSRosW(),
                )
                plain = SciMLBase.solve(breaks, alg; kw...)
                @test plain.retcode == SciMLBase.ReturnCode.ConvergenceFailure
                @test plain.t[end] == 0.5
                @test counts(plain) == (alg isa PETScDiffEq.TSRosW ? (5, 1, 0) : (5, 0, 1))
                @test all(u -> all(isfinite, u), plain.u)
                @test same(plain, SciMLBase.solve(breaks, alg; callback = never, kw...))
            end
        end

        # PETSc's step check fails the t = 300 span on 32-bit x86 (#79).
        Sys.WORD_SIZE == 64 && @testset "a successful solve makes no work vector at each stage" begin
            pl = PETScDiffEq.PETSc.getlib(; PetscScalar = Float64)
            PETScDiffEq.PETSc.initialize(pl)
            function last_id()
                v = PETScDiffEq._state_vec(pl, nothing, 1)
                id = Ref{Int64}(0)
                ccall(
                    PETScDiffEq._symbol(pl, :PetscObjectGetId), Cint, (Ptr{Cvoid}, Ptr{Int64}),
                    v.ptr, id,
                )
                PETScDiffEq.PETScCompat.destroy!(v)
                return id[]
            end
            vdp!(du, u, p, t) = (du[1] = u[2]; du[2] = 100 * (1 - u[1]^2) * u[2] - u[1]; nothing)
            function made(tf, alg)
                before = last_id()
                sol = SciMLBase.solve(SciMLBase.ODEProblem(vdp!, [2.0, 0.0], (0.0, tf)), alg)
                return last_id() - before, sol.stats.naccept + sol.stats.nreject
            end
            for (alg, per_step) in (
                    (PETScDiffEq.TSImplicit("bdf"), 1), (PETScDiffEq.TSRosW(), 2),
                    (PETScDiffEq.TSARKIMEX(), 2),
                )
                (a, n), (b, m) = made(1.0, alg), made(300.0, alg)
                @test m - n > 100
                @test b - a < per_step * (m - n)
            end
        end
    end

    @testset "the standard DiffEqCallbacks work" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        faster = SciMLBase.ODEProblem((du, u, p, t) -> (du[1] = -2u[1]; nothing), [1.0], (0.0, 1.0))
        tol = (dt = 0.05, reltol = 1.0e-8, abstol = 1.0e-10)

        @testset "$name" for (name, pr, cb) in (
                ("StepsizeLimiter", prob, DiffEqCallbacks.StepsizeLimiter((u, p, t) -> 0.05)),
                ("AutoAbstol", prob, DiffEqCallbacks.AutoAbstol()),
                ("PeriodicCallback", prob, DiffEqCallbacks.PeriodicCallback(i -> nothing, 0.2)),
                (
                    "IterativeCallback", prob,
                    DiffEqCallbacks.IterativeCallback(i -> i.t + 0.3, i -> nothing),
                ),
                (
                    "TerminateSteadyState", faster,
                    DiffEqCallbacks.TerminateSteadyState(1.0e-6, 1.0e-6),
                ),
                (
                    "PresetTimeCallback", prob,
                    DiffEqCallbacks.PresetTimeCallback([0.4], i -> nothing),
                ),
            )
            sol = SciMLBase.solve(pr, PETScDiffEq.TSRK("5dp"); tol..., callback = cb)
            @test sol.retcode in
                (SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.Terminated)
        end

        @testset "the pieces those callbacks reach for" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1)
            SciMLBase.step!(integ)
            @test SciMLBase.isadaptive(integ)
            @test length(SciMLBase.get_tmp_cache(integ)) >= 2
            @test integ.f isa SciMLBase.AbstractODEFunction
            @test SciMLBase.isinplace(integ.f)
            @test integ.opts.abstol isa Real
            @test SciMLBase.get_du(integ)[1] ≈ -integ.u[1]
            mid = (integ.tprev + integ.t) / 2
            @test abs(integ(mid)[1] - exp(-mid)) < 1.0e-6
            @test_throws ArgumentError integ(integ.t + 1.0)
            SciMLBase.add_saveat!(integ, 0.55)
            SciMLBase.set_proposed_dt!(integ, 0.05)
            sol = SciMLBase.solve!(integ)
            @test 0.55 in sol.t
        end

        @testset "LinearizingSavingCallback asks for the derivative" begin
            rot!(du, u, p, t) = (du[1] = -u[2]; du[2] = u[1]; nothing)
            spin = SciMLBase.ODEProblem(rot!, [1.0, 0.0], (0.0, 1.0))
            for alg in (PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRK("4"))
                ils = DiffEqCallbacks.IndependentlyLinearizedSolution(spin, 1)
                cb = DiffEqCallbacks.LinearizingSavingCallback(ils; abstol = 1.0e-8, reltol = 1.0e-8)
                sol = SciMLBase.solve(spin, alg; dt = 0.1, adaptive = false, callback = cb)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test length(ils) > length(sol.t)
                worst = zeros(2)
                for (t, vals) in ils
                    exact = ([cos(t), sin(t)], [-sin(t), cos(t)])
                    for k in 1:2
                        worst[k] = max(worst[k], maximum(abs.(vals[:, k] .- exact[k])))
                    end
                end
                @test worst[1] < 3.0e-6
                @test worst[2] < 3.0e-5
            end
        end
    end

    @testset "audit regressions" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        ramp!(du, u, p, t) = (du[1] = 1.0; nothing)
        ramp = SciMLBase.ODEProblem(ramp!, [0.0], (0.0, 1.0))

        @testset "a per-component tolerance survives collection" begin
            function noisy!(du, u, p, t)
                for _ in 1:200
                    v = fill(1.0e8, length(u))
                    v[1] += t
                end
                @inbounds for k in eachindex(u)
                    du[k] = -k * u[k]
                end
                return nothing
            end
            wide = SciMLBase.ODEProblem(noisy!, ones(8), (0.0, 1.0))
            sol = SciMLBase.solve(
                wide, PETScDiffEq.TSRK("5dp"); dt = 0.01, abstol = fill(1.0e-12, 8),
                reltol = 1.0e-12,
            )
            @test maximum(abs.(sol.u[end] .- [exp(-k) for k in 1:8])) < 1.0e-10
            @test sol.stats.naccept > 100
        end

        @testset "savevalues! keeps the interpolant consistent" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            SciMLBase.step!(integ)
            SciMLBase.savevalues!(integ)
            sol = SciMLBase.solve!(integ)
            @test length(sol.interp.du) == length(sol.u)
            @test abs(sol(0.15)[1] - exp(-0.15)) < 1.0e-6
            @test isfinite(sol(0.95)[1])
        end

        @testset "a discrete jump is saved even with saveat" begin
            fired = Ref(false)
            cb = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t >= 0.5 && !fired[],
                integ -> (integ.u[1] += 1.0; fired[] = true),
            )
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, saveat = 0.25,
                callback = cb,
            )
            @test count(==(0.5), sol.t) == 2
            @test abs(sol.u[findlast(==(0.5), sol.t)][1] - (exp(-0.5) + 1)) < 1.0e-6
        end

        @testset "a discrete callback still runs on an event step" begin
            stopped = Ref(false)
            set = SciMLBase.CallbackSet(
                SciMLBase.ContinuousCallback((u, t, integ) -> u[1] - 0.55, integ -> nothing),
                SciMLBase.DiscreteCallback(
                    (u, t, integ) -> u[1] >= 0.5499,
                    integ -> (stopped[] = true; SciMLBase.terminate!(integ)),
                ),
            )
            sol = SciMLBase.solve(
                ramp, PETScDiffEq.TSRK("3bs"); dt = 0.1, adaptive = false, callback = set,
            )
            @test stopped[]
            @test sol.t[end] < 0.56
        end

        @testset "terminate! keeps the state its affect! just set" begin
            cb = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t >= 0.5,
                integ -> (integ.u[1] = -999.0; SciMLBase.terminate!(integ)),
            )
            sol = SciMLBase.solve(
                ramp, PETScDiffEq.TSRK("3bs"); dt = 0.1, adaptive = false, callback = cb,
            )
            @test sol.u[end][1] == -999.0
        end

        @testset "a tstop rolled past by a root is not discarded" begin
            hit = Ref(false)
            set = SciMLBase.CallbackSet(
                SciMLBase.ContinuousCallback(
                    (u, t, integ) -> u[1] - exp(-0.45), integ -> nothing,
                ),
                SciMLBase.DiscreteCallback(
                    (u, t, integ) -> t == 0.5, integ -> (hit[] = true; integ.u[1] += 1.0),
                ),
            )
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.4, reltol = 1.0e-8, abstol = 1.0e-10,
                tstops = [0.5], callback = set,
            )
            @test hit[]
            @test abs(sol.u[end][1] - 0.9744101) < 1.0e-4
        end

        @testset "a failed reinit! leaves the integrator usable" begin
            integ = SciMLBase.init(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, abstol = [1.0e-8], reltol = 1.0e-6,
            )
            SciMLBase.step!(integ)
            @test_throws ArgumentError SciMLBase.reinit!(integ, [1.0, 2.0])
            SciMLBase.step!(integ)
            @test integ.t > 0.1
            SciMLBase.terminate!(integ)
        end

        @testset "reinit! onto a different size resizes the interpolation cache" begin
            wide = SciMLBase.ODEProblem(
                (du, u, p, t) -> (du .= -u; nothing), ones(4), (0.0, 2.0),
            )
            integ = SciMLBase.init(
                wide, PETScDiffEq.TSRK("5dp"); dt = 0.2,
                callback = SciMLBase.ContinuousCallback(
                    (u, t, i) -> sum(u) - 1.0, i -> nothing,
                ),
            )
            SciMLBase.reinit!(integ, ones(2))
            @test length(integ.ucache) == 2
            sol = SciMLBase.solve!(integ)
            @test length(sol.u[end]) == 2
            @test all(isfinite, sol.u[end])
        end

        @testset "dtmax = Inf means no cap" begin
            sol = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.01, dtmax = Inf)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.01, dtmin = 0.0,
            ).retcode == SciMLBase.ReturnCode.Success
            capped = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.01, dtmax = 0.05,
            )
            @test maximum(diff(capped.t)) <= 0.0501
        end
    end

    @testset "TSIRK" begin
        proto = sparse([1], [1], [1.0], 1, 1)
        prob = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = decay_jac!, jac_prototype = proto),
            [1.0], (0.0, 1.0),
        )

        @testset "order is twice the stage count" begin
            for nstages in (1, 2, 3)
                errs = [
                    abs(
                        SciMLBase.solve(
                            prob, PETScDiffEq.TSIRK(nstages); dt = dt, adaptive = false,
                        ).u[end][1] - exp(-1),
                    ) for dt in (0.2, 0.1, 0.05)
                ]
                orders = [log2(errs[i] / errs[i + 1]) for i in 1:2]
                @test all(o -> isapprox(o, 2 * nstages; atol = 0.2), orders)
            end
        end

        @testset "a dense jac works too" begin
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(decay!; jac = decay_jac!), [1.0], (0.0, 1.0),
                ), PETScDiffEq.TSIRK(); dt = 0.1, adaptive = false,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test abs(sol.u[end][1] - exp(-1)) < 1.0e-10
        end

        @testset "the analytic Jacobian is what it solves with" begin
            wrong = SciMLBase.solve(
                SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(decay!; jac = decay_wrong_jac!), [1.0], (0.0, 1.0),
                ), PETScDiffEq.TSIRK(); dt = 0.1, adaptive = false,
            )
            @test wrong.retcode == SciMLBase.ReturnCode.ConvergenceFailure
            @test abs(wrong.u[end][1] - exp(-1)) > 1.0e-3
            right = SciMLBase.solve(prob, PETScDiffEq.TSIRK(); dt = 0.1, adaptive = false)
            @test abs(right.u[end][1] - exp(-1)) < 1.0e-10
            @test right.stats.njacs > 0
        end

        @testset "it says what it needs" begin
            @test_throws ArgumentError SciMLBase.solve(
                SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)),
                PETScDiffEq.TSIRK(; autodiff = PETScDiffEq.AutoFiniteDiff()); dt = 0.1,
            )
            @test_throws ArgumentError SciMLBase.solve(
                SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(
                        decay!; jac = decay_jac!, mass_matrix = fill(2.0, 1, 1),
                    ), [1.0], (0.0, 1.0),
                ), PETScDiffEq.TSIRK(); dt = 0.05,
            )
        end

        @testset "fixed step, so a tolerance warns" begin
            @test_logs (:warn,) match_mode = :any SciMLBase.solve(
                prob, PETScDiffEq.TSIRK(); dt = 0.1, reltol = 1.0e-8,
            )
        end

        @testset "the default preconditioner can be overridden" begin
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSIRK(3, ["-pc_type", "none"]); dt = 0.1,
                adaptive = false,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test abs(sol.u[end][1] - exp(-1)) < 1.0e-10
        end

        @testset "irk picked by TSGeneric or an option gets the same preconditioner" begin
            for pr in (prob, SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)))
                irk = SciMLBase.solve(pr, PETScDiffEq.TSIRK(); dt = 0.1, adaptive = false)
                for alg in (
                        PETScDiffEq.TSGeneric("irk"),
                        PETScDiffEq.TSImplicit("beuler", ["-ts_type", "irk"]),
                    )
                    @test SciMLBase.solve(pr, alg; dt = 0.1, adaptive = false).u == irk.u
                end
            end
        end

        @testset "a stiff system through the integrator" begin
            function stiff!(du, u, p, t)
                du[1] = -1000 * (u[1] - cos(t))
                return du[2] = -u[2]
            end
            function stiff_jac!(J, u, p, t)
                J[1, 1] = -1000.0
                return J[2, 2] = -1.0
            end
            sol = SciMLBase.solve!(
                SciMLBase.init(
                    SciMLBase.ODEProblem(
                        SciMLBase.ODEFunction(
                            stiff!; jac = stiff_jac!,
                            jac_prototype = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2),
                        ), [1.0, 1.0], (0.0, 0.1),
                    ), PETScDiffEq.TSIRK(); dt = 1.0e-3, adaptive = false,
                ),
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test abs(sol.u[end][2] - exp(-0.1)) < 1.0e-10
        end
    end

    @testset "who is warned about a tolerance" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        for alg in (
                PETScDiffEq.TSImplicit("beuler"), PETScDiffEq.TSImplicit("cn"),
                PETScDiffEq.TSImplicit("theta"),
            )
            @test_logs (:warn,) match_mode = :any SciMLBase.solve(
                prob, alg; dt = 0.05, reltol = 1.0e-6,
            )
        end
        @test_logs (:warn, r"`rk 4` has no embedded error estimate") match_mode = :any SciMLBase.solve(
            prob, PETScDiffEq.TSRK("4"); dt = 0.05, reltol = 1.0e-6,
        )
        for alg in (
                PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSRK("5dp"),
                PETScDiffEq.TSRosW("ra34pw2"),
                PETScDiffEq.TSGeneric("alpha"),
            )
            @test_logs min_level = Logging.Warn SciMLBase.solve(
                prob, alg; dt = 0.05, reltol = 1.0e-6,
            )
        end
    end

    @testset "the exported names carry a docstring" begin
        # Read from source: `Base.doc` is not on every version, and `@doc` never reports none.
        lines = reduce(
            vcat,
            split(read(joinpath(@__DIR__, "..", "src", file), String), '\n')
                for file in ("PETScDiffEq.jl", "adjoint.jl")
        )
        exported = filter(!=(:PETScDiffEq), names(PETScDiffEq))
        @test length(exported) == 13
        for n in exported
            i = findfirst(
                l -> occursin(Regex("^((mutable )?struct|function) \\Q$(n)\\E(?!\\w)"), l), lines,
            )
            @test i !== nothing
            i === nothing && continue
            @test strip(lines[i - 1]) == "\"\"\""
        end
    end

    @testset "every subtype the docstrings name actually runs" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 0.1))
        runs(alg) = SciMLBase.solve(prob, alg; dt = 0.01, adaptive = false).retcode ==
            SciMLBase.ReturnCode.Success
        for st in ("1fe", "2b", "3", "4", "3bs", "5dp", "5f", "5bs")
            @test runs(PETScDiffEq.TSRK(st))
        end
        for st in (
                "theta1", "theta2", "2p", "2m", "ra34pw2", "ra3pw", "r34prw", "sandu3",
                "rodas3", "grk4t",
            )
            @test runs(PETScDiffEq.TSRosW(st))
        end
        for st in ("2e", "prssp2", "3", "ars443", "bpr3", "4", "5")
            @test runs(PETScDiffEq.TSARKIMEX(st))
        end
        for st in ("beuler", "cn", "theta", "bdf")
            @test runs(PETScDiffEq.TSImplicit(st))
        end
        @test runs(PETScDiffEq.TSGeneric("alpha"))
        for st in ("euler", "ssp")
            @test runs(PETScDiffEq.TSGeneric(st; explicit = true))
            @test_throws Exception SciMLBase.solve(
                prob, PETScDiffEq.TSGeneric(st); dt = 0.01, adaptive = false,
            )
        end
    end

    @testset "VectorContinuousCallback" begin
        function two!(du, u, p, t)
            du[1] = -u[1]
            return du[2] = -2u[2]
        end
        prob = SciMLBase.ODEProblem(two!, [1.0, 1.0], (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp")
        tols = (dt = 0.05, reltol = 1.0e-10, abstol = 1.0e-12)

        @testset "each component gets its own root" begin
            events = Tuple{Float64, Vector{Int8}}[]
            cb = SciMLBase.VectorContinuousCallback(
                (out, u, t, integ) -> (out[1] = u[1] - 0.5; out[2] = u[2] - 0.5; nothing),
                (integ, mask) -> push!(events, (integ.t, Vector{Int8}(mask))), 2,
            )
            sol = SciMLBase.solve(prob, alg; tols..., callback = cb)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test length(events) == 2
            @test abs(events[1][1] - log(2.0) / 2) < 1.0e-9
            @test events[1][2] == Int8[0, -1]
            @test abs(events[2][1] - log(2.0)) < 1.0e-9
            @test events[2][2] == Int8[-1, 0]
        end

        @testset "components crossing together share one event" begin
            decay = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
            events = Tuple{Float64, Vector{Int8}}[]
            cb = SciMLBase.VectorContinuousCallback(
                (out, u, t, integ) -> (out[1] = u[1] - 0.5; out[2] = u[1] - 0.5; nothing),
                (integ, mask) -> push!(events, (integ.t, Vector{Int8}(mask))), 2,
            )
            SciMLBase.solve(decay, alg; tols..., callback = cb)
            @test length(events) == 1
            @test abs(events[1][1] - log(2.0)) < 1.0e-9
            @test events[1][2] == Int8[-1, -1]
        end

        @testset "components crossing together share one event far from t = 0" begin
            slow!(du, u, p, t) = (du[1] = -u[1] / 50; du[2] = -u[2] / 50; nothing)
            events = Tuple{Float64, Vector{Int8}}[]
            cb = SciMLBase.VectorContinuousCallback(
                (out, u, t, integ) -> (out[1] = u[1] - 0.5; out[2] = u[2] - 1.5; nothing),
                (integ, mask) -> push!(events, (integ.t, Vector{Int8}(mask))), 2,
            )
            SciMLBase.solve(
                SciMLBase.ODEProblem(slow!, [1.0, 3.0], (0.0, 100.0)), alg;
                dt = 2.5, reltol = 1.0e-10, abstol = 1.0e-12, callback = cb,
            )
            @test length(events) == 1
            @test abs(events[1][1] - 50 * log(2.0)) < 1.0e-7
            @test events[1][2] == Int8[-1, -1]
        end

        @testset "components crossing apart fire apart whatever abstol is" begin
            events = Tuple{Float64, Vector{Int8}}[]
            cb = SciMLBase.VectorContinuousCallback(
                (out, u, t, integ) -> (out[1] = u[1] - 0.5; out[2] = u[1] - 0.4995; nothing),
                (integ, mask) -> push!(events, (integ.t, Vector{Int8}(mask))), 2;
                abstol = 1.0e-2,
            )
            SciMLBase.solve(
                SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), alg;
                dt = 0.1, adaptive = false, callback = cb,
            )
            @test length(events) == 2
            @test abs(events[1][1] - log(2.0)) < 1.0e-8
            @test events[1][2] == Int8[-1, 0]
            @test abs(events[2][1] + log(0.4995)) < 1.0e-8
            @test events[2][2] == Int8[0, -1]
        end

        @testset "the mask carries the crossing direction" begin
            function osc!(du, u, p, t)
                du[1] = u[2]
                return du[2] = -u[1]
            end
            events = Tuple{Float64, Vector{Int8}}[]
            cb = SciMLBase.VectorContinuousCallback(
                (out, u, t, integ) -> (out[1] = u[1]; nothing),
                (integ, mask) -> push!(events, (integ.t, Vector{Int8}(mask))), 1,
            )
            SciMLBase.solve(
                SciMLBase.ODEProblem(osc!, [0.0, 1.0], (0.0, 7.0)), alg; tols...,
                callback = cb,
            )
            @test length(events) == 2
            @test abs(events[1][1] - pi) < 1.0e-9
            @test events[1][2] == Int8[-1]
            @test abs(events[2][1] - 2pi) < 1.0e-9
            @test events[2][2] == Int8[1]
        end

        @testset "affect! may change the state and terminate" begin
            cb = SciMLBase.VectorContinuousCallback(
                (out, u, t, integ) -> (out[1] = u[1] - 0.5; out[2] = u[2] - 0.5; nothing),
                (integ, mask) -> (
                    mask[1] != 0 ? SciMLBase.terminate!(integ) :
                        (integ.u[2] += 1.0)
                ), 2,
            )
            sol = SciMLBase.solve(prob, alg; tols..., callback = cb)
            @test sol.retcode == SciMLBase.ReturnCode.Terminated
            @test abs(sol.t[end] - log(2.0)) < 1.0e-9
            @test abs(sol.u[end][2] - 0.75) < 1.0e-7
        end

        @testset "beside the other callback kinds in a CallbackSet" begin
            events, hits, ticks = Int[], Float64[], Float64[]
            vec = SciMLBase.VectorContinuousCallback(
                (out, u, t, integ) -> (out[1] = u[2] - 0.5; nothing),
                (integ, mask) -> push!(events, 1), 1,
            )
            scalar = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.5, integ -> push!(hits, integ.t),
            )
            discrete = SciMLBase.DiscreteCallback(
                (u, t, integ) -> true, integ -> push!(ticks, integ.t),
            )
            sol = SciMLBase.solve(
                prob, alg; tols...,
                callback = SciMLBase.CallbackSet(vec, scalar, discrete),
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test length(events) == 1
            @test length(hits) == 1 && abs(hits[1] - log(2.0)) < 1.0e-9
            @test !isempty(ticks)
        end

        @testset "save_positions brackets the event" begin
            cb = SciMLBase.VectorContinuousCallback(
                (out, u, t, integ) -> (out[1] = u[1] - 0.5; nothing),
                (integ, mask) -> (integ.u[1] += 1.0), 1,
            )
            sol = SciMLBase.solve(prob, alg; tols..., callback = cb)
            i = findfirst(t -> abs(t - log(2.0)) < 1.0e-9, sol.t)
            @test i !== nothing
            @test sol.t[i + 1] == sol.t[i]
            @test abs(sol.u[i][1] - 0.5) < 1.0e-8
            @test abs(sol.u[i + 1][1] - 1.5) < 1.0e-8
        end
    end

    @testset "vector tolerances" begin
        function two!(du, u, p, t)
            du[1] = -u[1]
            return du[2] = -1000 * u[2]
        end
        prob = SciMLBase.ODEProblem(two!, [1.0, 1.0], (0.0, 0.1))
        alg = PETScDiffEq.TSRK("5dp")
        steps(kw) = SciMLBase.solve(prob, alg; dt = 1.0e-4, reltol = 1.0e-3, kw...)

        @testset "a vector of equal entries matches the scalar" begin
            a = SciMLBase.solve(prob, alg; dt = 1.0e-4, reltol = 1.0e-8, abstol = 1.0e-10)
            b = SciMLBase.solve(
                prob, alg; dt = 1.0e-4, reltol = [1.0e-8, 1.0e-8],
                abstol = [1.0e-10, 1.0e-10],
            )
            @test a.t == b.t
            @test a.u == b.u
            @test a.stats.naccept == b.stats.naccept
        end

        @testset "the entries apply per component" begin
            slow = steps((abstol = [1.0e-12, 1.0],))
            stiff = steps((abstol = [1.0, 1.0e-12],))
            loose = steps((abstol = [1.0, 1.0],))
            @test slow.stats.naccept == loose.stats.naccept
            @test stiff.stats.naccept > loose.stats.naccept
            @test abs(stiff.u[end][2] - exp(-100)) < 1.0e-10
            @test abs(slow.u[end][2] - exp(-100)) > 1.0e-3
        end

        @testset "a bad tolerance vector is rejected" begin
            @test_throws ArgumentError SciMLBase.solve(
                prob, alg; dt = 1.0e-4, abstol = [1.0e-6],
            )
            @test_throws ArgumentError SciMLBase.solve(
                prob, alg; dt = 1.0e-4, reltol = [1.0e-6, 1.0e-6, 1.0e-6],
            )
            @test_throws ArgumentError SciMLBase.solve(
                prob, alg; dt = 1.0e-4, abstol = [1.0e-6, -1.0],
            )

            integ = SciMLBase.init(prob, alg; dt = 1.0e-4)
            held() = (
                t = PETScDiffEq.LibPETSc.TSGetTolerances(integ.h.petsclib, integ.h.ts);
                (t[1], t[2].ptr, t[3], t[4].ptr)
            )
            @test_throws ArgumentError integ.opts.abstol = [1.0e-8]
            @test_throws ArgumentError integ.opts.reltol = [1.0e-6, -1.0]
            @test (integ.opts.abstol, integ.opts.reltol) == (1.0e-6, 1.0e-3)
            @test held() == (1.0e-6, C_NULL, 1.0e-3, C_NULL)
            integ.opts.abstol = [1.0e-8, 1.0e-8]
            @test held()[2] != C_NULL
            SciMLBase.step!(integ)
            @test !integ.finished
            @test integ.t > 0
        end

        @testset "a non-adaptive method still warns" begin
            @test_logs (:warn,) match_mode = :any SciMLBase.solve(
                prob, PETScDiffEq.TSImplicit("beuler"); dt = 1.0e-3,
                abstol = [1.0e-10, 1.0e-10],
            )
        end
    end

    @testset "save_idxs" begin
        function pair!(du, u, p, t)
            du[1] = -u[1]
            return du[2] = -2u[2]
        end
        prob = SciMLBase.ODEProblem(pair!, [1.0, 1.0], (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp")
        tols = (reltol = 1.0e-9, abstol = 1.0e-11)

        @testset "keeps only the named components" begin
            full = SciMLBase.solve(prob, alg; dt = 0.1, tols...)
            one = @test_logs min_level = Logging.Warn SciMLBase.solve(
                prob, alg; dt = 0.1, save_idxs = [2], tols...,
            )
            @test length(one.u[1]) == 1
            @test one.t == full.t
            @test all(one.u[i][1] == full.u[i][2] for i in eachindex(one.t))
            @test one.stats.nf == full.stats.nf
        end

        @testset "dense output keeps the matching derivative" begin
            sol = SciMLBase.solve(prob, alg; dt = 0.1, save_idxs = [2], tols...)
            @test sol.dense
            @test length(sol.interp.du) == length(sol.u)
            @test sol.interp.du[3] ≈ -2 .* sol.u[3]
            @test abs(sol(0.55)[1] - exp(-2 * 0.55)) < 1.0e-7
        end

        @testset "a single index and saveat" begin
            one = SciMLBase.solve(prob, alg; dt = 0.1, save_idxs = 1, tols...)
            @test length(one.u[1]) == 1
            @test abs(one.u[end][1] - exp(-1)) < 1.0e-7
            at = SciMLBase.solve(prob, alg; dt = 0.1, saveat = 0.25, save_idxs = [2], tols...)
            @test at.t == [0.0, 0.25, 0.5, 0.75, 1.0]
            @test abs(at.u[3][1] - exp(-1.0)) < 1.0e-7
        end

        @testset "a blow-up in an unsaved component is still Unstable" begin
            function mixed!(du, u, p, t)
                du[1] = u[1]^2
                return du[2] = -u[2]
            end
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(mixed!, [1.0, 1.0], (0.0, 5.0)), alg;
                dt = 0.1, adaptive = false, save_idxs = [2],
            )
            @test all(isfinite, sol.u[end])
            @test sol.retcode == SciMLBase.ReturnCode.Unstable
        end

        @testset "an index outside the state is rejected" begin
            for bad in ([0], [3], Int[])
                @test_throws ArgumentError SciMLBase.solve(
                    prob, alg; dt = 0.1, save_idxs = bad,
                )
            end
        end
    end

    @testset "out-of-place problems" begin
        oop(u, p, t) = -u
        prob = SciMLBase.ODEProblem(oop, [1.0], (0.0, 1.0))
        inplace = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))

        @testset "matches the in-place formulation exactly" begin
            for alg in (
                    PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSImplicit("bdf"),
                    PETScDiffEq.TSRosW("ra34pw2"),
                )
                a = SciMLBase.solve(prob, alg; dt = 0.05, reltol = 1.0e-9, abstol = 1.0e-11)
                b = SciMLBase.solve(inplace, alg; dt = 0.05, reltol = 1.0e-9, abstol = 1.0e-11)
                @test a.retcode == SciMLBase.ReturnCode.Success
                @test a.t == b.t
                @test a.u == b.u
                @test a.stats.nf == b.stats.nf
                @test abs(a.u[end][1] - exp(-1)) < 1.0e-6
            end
        end

        @testset "an out-of-place jac is used" begin
            ojac(u, p, t) = fill(-1.0, 1, 1)
            owrong(u, p, t) = fill(100.0, 1, 1)
            good = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(oop; jac = ojac), [1.0], (0.0, 1.0),
            )
            sol = SciMLBase.solve(good, PETScDiffEq.TSImplicit("bdf"); dt = 0.05)
            @test sol.stats.njacs > 0
            @test abs(sol.u[end][1] - exp(-1)) < 1.0e-2
            bad = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(oop; jac = owrong), [1.0], (0.0, 1.0),
            )
            @test SciMLBase.solve(
                bad, PETScDiffEq.TSImplicit("cn", ["-ts_max_snes_failures", "1"]); dt = 0.05,
            ).retcode != SciMLBase.ReturnCode.Success
        end

        @testset "with a sparse jac_prototype" begin
            pair(u, p, t) = [-1000 * (u[1] - cos(t)), -u[2]]
            pjac(u, p, t) = sparse([1, 2], [1, 2], [-1000.0, -1.0], 2, 2)
            outside(u, p, t) = sparse([1, 1, 2], [1, 2, 2], [-1000.0, 7.0, -1.0], 2, 2)
            proto = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(pair; jac = pjac, jac_prototype = proto),
                    [1.0, 1.0], (0.0, 0.1),
                ), PETScDiffEq.TSImplicit("bdf"); dt = 1.0e-3,
            )
            @test sol.stats.njacs > 0
            @test abs(sol.u[end][2] - exp(-0.1)) < 1.0e-4
            @test_throws ArgumentError SciMLBase.solve(
                SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(pair; jac = outside, jac_prototype = proto),
                    [1.0, 1.0], (0.0, 0.1),
                ), PETScDiffEq.TSImplicit("bdf"); dt = 1.0e-3,
            )
        end

        @testset "copying into the sparse buffer" begin
            declared = sparse([1, 1, 2], [1, 2, 2], [1.0, 1.0, 1.0], 2, 2)
            J = copy(declared)
            PETScDiffEq._copy_jac!(J, sparse([1, 1, 2], [1, 2, 2], [1.0, 7.0, 2.0], 2, 2))
            @test J[1, 2] == 7.0
            PETScDiffEq._copy_jac!(J, sparse([1, 2], [1, 2], [3.0, 4.0], 2, 2))
            @test J[1, 1] == 3.0
            @test J[2, 2] == 4.0
            @test J[1, 2] == 0.0

            diagonly = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
            PETScDiffEq._copy_jac!(diagonly, [5.0 0.0; 0.0 6.0])
            @test diagonly[1, 1] == 5.0
            @test diagonly[2, 2] == 6.0
        end

        @testset "split, callbacks and the integrator" begin
            split = SciMLBase.SplitODEProblem(
                (u, p, t) -> [-1000 * (u[1] - cos(t))], (u, p, t) -> [-sin(t)],
                [1.0], (0.0, 0.1),
            )
            s = SciMLBase.solve(split, PETScDiffEq.TSARKIMEX("3"); dt = 1.0e-3)
            @test s.retcode == SciMLBase.ReturnCode.Success
            @test abs(s.u[end][1] - cos(0.1)) < 1.0e-3

            hits = Float64[]
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.5,
                integ -> (push!(hits, integ.t); integ.u[1] += 1.0),
            )
            long = SciMLBase.ODEProblem(oop, [1.0], (0.0, 2.0))
            SciMLBase.solve(
                long, PETScDiffEq.TSRK("5dp"); dt = 0.1, reltol = 1.0e-10,
                abstol = 1.0e-12, callback = cb,
            )
            @test length(hits) == 2
            @test abs(hits[1] - log(2.0)) < 1.0e-9

            integ = SciMLBase.init(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, abstol = 1.0e-8, reltol = 1.0e-8,
            )
            SciMLBase.step!(integ)
            SciMLBase.reinit!(integ)
            @test abs(SciMLBase.solve!(integ).u[end][1] - exp(-1)) < 1.0e-5
        end
    end

    @testset "ContinuousCallback" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 2.0))
        jump(hits) = SciMLBase.ContinuousCallback(
            (u, t, integ) -> u[1] - 0.5, integ -> (push!(hits, integ.t); integ.u[1] += 1.0),
        )
        after(t) = 1.5 * exp(-(t - log(6.0)))
        ramp!(du, u, p, t) = (du[1] = 1.0; nothing)

        @testset "roots are located on the interpolant" begin
            for (alg, tol) in (
                    (PETScDiffEq.TSRK("5dp"), 1.0e-9),
                    (PETScDiffEq.TSRosW("ra34pw2"), 1.0e-9),
                    (PETScDiffEq.TSImplicit("bdf"), 1.0e-5),
                )
                hits = Float64[]
                sol = SciMLBase.solve(
                    prob, alg; dt = 0.1, reltol = 1.0e-10, abstol = 1.0e-12,
                    callback = jump(hits),
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test length(hits) == 2
                @test abs(hits[1] - log(2.0)) < tol
                @test abs(hits[2] - log(6.0)) < tol
                @test abs(sol.u[end][1] - after(2.0)) < tol
            end
        end

        @testset "the root does not move with the callback's abstol" begin
            loose(hits, abstol) = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.5,
                integ -> (push!(hits, integ.t); integ.u[1] += 1.0);
                abstol = abstol,
            )
            for abstol in (1.0e-6, 1.0e-4, 1.0e-2)
                hits = Float64[]
                sol = SciMLBase.solve(
                    prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, reltol = 1.0e-10,
                    abstol = 1.0e-12, callback = loose(hits, abstol),
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test length(hits) == 2
                @test abs(hits[1] - log(2.0)) < 1.0e-9
                @test abs(hits[2] - log(6.0)) < 1.0e-9
            end
            fall!(du, u, p, t) = (du[1] = u[2]; du[2] = -9.81; nothing)
            drop = SciMLBase.ODEProblem(fall!, [1.0, 0.0], (0.0, 1.0))
            impact = sqrt(2 / 9.81)
            for rootfind in (SciMLBase.LeftRootFind, SciMLBase.RightRootFind),
                    kw in ((;), (abstol = 1.0e-6,), (abstol = 1.0e-4,), (abstol = 1.0e-2,))
                cb = SciMLBase.ContinuousCallback(
                    (u, t, integ) -> u[1], SciMLBase.terminate!; rootfind = rootfind, kw...,
                )
                sol = SciMLBase.solve(
                    drop, PETScDiffEq.TSRK("5dp"); dt = 0.05, reltol = 1.0e-10,
                    abstol = 1.0e-12, callback = cb,
                )
                @test abs(sol.t[end] - impact) <= 16 * eps(impact)
            end
        end

        @testset "a crossing skipped as a repeat hides nothing after it" begin
            hold!(du, u, p, t) = (du[1] = 0.0; nothing)
            function level(u, t, integ)
                u[1] == 0 && return 0.2345 - t
                t < 0.235 && return -1.0e-16
                t <= 0.2353 && return 0.0
                t < 0.2495 && return 1.0e-16
                return -1.0e-16
            end
            hits = Float64[]
            cb = SciMLBase.ContinuousCallback(
                level, integ -> (push!(hits, integ.t); integ.u[1] = 1.0),
            )
            SciMLBase.solve(
                SciMLBase.ODEProblem(hold!, [0.0], (0.0, 0.5)), PETScDiffEq.TSRK("5dp");
                dt = 0.1, adaptive = false, callback = cb,
            )
            @test length(hits) == 2
            @test abs(hits[1] - 0.2345) < 1.0e-12
            @test abs(hits[2] - 0.2495) < 1.0e-12
        end

        @testset "a condition the affect! moves off its root can fire again at once" begin
            function fires(; kw...)
                hits = Float64[]
                cb = SciMLBase.ContinuousCallback(
                    (u, t, integ) -> u[1] - 1.0,
                    integ -> (push!(hits, integ.t); length(hits) < 3 && (integ.u[1] -= 1.0e-3));
                    kw...,
                )
                SciMLBase.solve(
                    SciMLBase.ODEProblem(ramp!, [0.99], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
                    dt = 0.25, adaptive = false, callback = cb,
                )
                return hits
            end
            hits = fires()
            @test length(hits) == 3
            @test all(abs.(hits .- [0.01, 0.011, 0.012]) .< 1.0e-12)
            @test length(fires(abstol = 1.0e-2)) == 1
        end

        @testset "a crossing just after an event is found" begin
            fall!(du, u, p, t) = (du[1] = u[2]; du[2] = -9.81; nothing)
            toss = SciMLBase.ODEProblem(fall!, [0.0, 5.0], (0.0, 1.0))
            h = 1.273
            apex, half = 5.0 / 9.81, sqrt(5.0^2 - 2 * 9.81 * h) / 9.81
            band = SciMLBase.ODEProblem(ramp!, [0.0], (0.0, 1.0))
            for rootfind in (SciMLBase.LeftRootFind, SciMLBase.RightRootFind)
                ups, downs = Float64[], Float64[]
                cb = SciMLBase.ContinuousCallback(
                    (u, t, integ) -> u[1] - h, integ -> push!(ups, integ.t),
                    integ -> push!(downs, integ.t); rootfind = rootfind,
                )
                SciMLBase.solve(
                    toss, PETScDiffEq.TSRK("5dp"); callback = cb, abstol = 1.0e-4, reltol = 1.0e-4,
                )
                @test length(ups) == 1 && abs(ups[1] - (apex - half)) < 1.0e-9
                @test length(downs) == 1 && abs(downs[1] - (apex + half)) < 1.0e-9

                ups, downs = Float64[], Float64[]
                cb = SciMLBase.ContinuousCallback(
                    (u, t, integ) -> (u[1] - 0.25) * (0.32 - u[1]),
                    integ -> push!(ups, integ.t), integ -> push!(downs, integ.t);
                    rootfind = rootfind, interp_points = 0,
                )
                SciMLBase.solve(
                    band, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, callback = cb,
                )
                @test length(ups) == 1 && abs(ups[1] - 0.25) < 1.0e-12
                @test length(downs) == 1 && abs(downs[1] - 0.32) < 1.0e-12
            end
        end

        @testset "a crossing inside the nudge is not seen" begin
            ups, downs = Float64[], Float64[]
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> (u[1] - 0.2545) * (0.29 - u[1]),
                integ -> push!(ups, integ.t), integ -> push!(downs, integ.t);
                repeat_nudge = 1 // 2,
            )
            SciMLBase.solve(
                SciMLBase.ODEProblem(ramp!, [0.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
                dt = 0.1, adaptive = false, callback = cb,
            )
            @test length(ups) == 1 && abs(ups[1] - 0.2545) < 1.0e-12
            @test isempty(downs)
        end

        @testset "only the step right after an event starts past it" begin
            ups, downs = Float64[], Float64[]
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> (u[1] - 0.2545) * (0.255 - u[1]),
                integ -> push!(ups, integ.t), integ -> push!(downs, integ.t);
                abstol = 1.0e-6,
            )
            SciMLBase.solve(
                SciMLBase.ODEProblem(ramp!, [0.0], (0.0, 0.5)), PETScDiffEq.TSRK("5dp");
                dt = 0.1, adaptive = false, tstops = [0.25451], callback = cb,
            )
            @test length(ups) == 1 && abs(ups[1] - 0.2545) < 1.0e-12
            @test length(downs) == 1 && abs(downs[1] - 0.255) < 1.0e-12
        end

        @testset "locating a root takes few condition evaluations" begin
            n = Ref(0)
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> (n[] += 1; u[1] - 0.5), integ -> nothing; interp_points = 0,
            )
            SciMLBase.solve(
                SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
                dt = 1.0, adaptive = false, callback = cb,
            )
            @test n[] <= 30
        end

        @testset "a bouncing ball" begin
            function ball!(du, u, p, t)
                du[1] = u[2]
                return du[2] = -9.81
            end
            bounces = Float64[]
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1],
                integ -> (push!(bounces, integ.t); integ.u[2] = -0.9 * integ.u[2]),
            )
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(ball!, [1.0, 0.0], (0.0, 3.0)), PETScDiffEq.TSRK("5dp");
                dt = 0.05, reltol = 1.0e-10, abstol = 1.0e-12, callback = cb,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test length(bounces) == 4
            @test abs(bounces[1] - sqrt(2 / 9.81)) < 1.0e-12
            gaps = diff(bounces)
            @test all(isapprox(0.9; atol = 1.0e-6), gaps[2:end] ./ gaps[1:(end - 1)])
            @test minimum(u[1] for u in sol.u) > -1.0e-10
        end

        @testset "RightRootFind lands each bounce just past the floor" begin
            fall!(du, u, p, t) = (du[1] = u[2]; du[2] = -9.81; nothing)
            ups, downs, heights = Float64[], Float64[], Float64[]
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1], integ -> push!(ups, integ.t),
                integ -> (
                    push!(downs, integ.t); push!(heights, integ.u[1]);
                    integ.u[2] = -0.9 * integ.u[2]
                );
                rootfind = SciMLBase.RightRootFind,
            )
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(fall!, [1.0, 0.0], (0.0, 3.0)), PETScDiffEq.TSRK("5dp");
                dt = 0.05, reltol = 1.0e-10, abstol = 1.0e-12, callback = cb,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test isempty(ups)
            @test length(downs) == 4
            @test all(h -> -1.0e-14 < h <= 0, heights)
            gaps = diff(downs)
            @test all(isapprox(0.9; atol = 1.0e-6), gaps[2:end] ./ gaps[1:(end - 1)])
        end

        @testset "crossing direction picks the handler" begin
            function osc!(du, u, p, t)
                du[1] = u[2]
                return du[2] = -u[1]
            end
            ups, downs = Float64[], Float64[]
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1], integ -> push!(ups, integ.t),
                integ -> push!(downs, integ.t),
            )
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(osc!, [0.0, 1.0], (0.0, 7.0)), PETScDiffEq.TSRK("5dp");
                dt = 0.05, reltol = 1.0e-10, abstol = 1.0e-12, callback = cb,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test length(downs) == 1 && abs(downs[1] - pi) < 1.0e-9
            @test length(ups) == 1 && abs(ups[1] - 2pi) < 1.0e-9
        end

        @testset "solve matches solve! on the integrator" begin
            h1, h2 = Float64[], Float64[]
            a = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = jump(h1))
            b = SciMLBase.solve!(
                SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = jump(h2)),
            )
            @test h1 == h2
            @test a.t == b.t
            @test a.u == b.u
        end

        @testset "save_positions brackets the event" begin
            hits = Float64[]
            both = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = jump(hits))
            @test count(==(hits[1]), both.t) == 2
            i = findfirst(==(hits[1]), both.t)
            @test abs(both.u[i][1] - 0.5) < 1.0e-8
            @test abs(both.u[i + 1][1] - 1.5) < 1.0e-8

            quiet = Float64[]
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.5,
                integ -> (push!(quiet, integ.t); integ.u[1] += 1.0);
                save_positions = (false, false),
            )
            none = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = cb)
            @test quiet == hits
            @test !(hits[1] in none.t)
        end

        @testset "a saveat on the root is not saved twice" begin
            double(sp) = SciMLBase.ContinuousCallback(
                (u, t, integ) -> t - 0.5, integ -> (integ.u .*= 2; nothing);
                save_positions = sp,
            )
            kw = (saveat = [0.5], abstol = 1.0e-10, reltol = 1.0e-10)
            for alg in (PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRK("3bs"))
                both = SciMLBase.solve(prob, alg; kw..., callback = double((true, true)))
                @test both.t == [0.5, 0.5]
                @test both.u[2] == 2 .* both.u[1]
                pre = SciMLBase.solve(prob, alg; kw..., callback = double((true, false)))
                @test pre.t == [0.5]
                @test abs(pre.u[1][1] - exp(-0.5)) < 1.0e-8
            end
        end

        @testset "dense output across the event" begin
            hits = Float64[]
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, reltol = 1.0e-10, abstol = 1.0e-12,
                callback = jump(hits),
            )
            @test sol.dense
            @test length(sol.interp.du) == length(sol.u)
            @test abs(sol(hits[1])[1] - 0.5) < 1.0e-8
            @test abs(sol(hits[1]; continuity = :right)[1] - 1.5) < 1.0e-8
            @test abs(sol(1.9)[1] - after(1.9)) < 1.0e-6
        end

        @testset "terminate! from a continuous affect!" begin
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.5, integ -> SciMLBase.terminate!(integ),
            )
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, reltol = 1.0e-10, abstol = 1.0e-12,
                callback = cb,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Terminated
            @test abs(sol.t[end] - log(2.0)) < 1.0e-9
            @test abs(sol.u[end][1] - 0.5) < 1.0e-8
        end

        @testset "a terminating event saves as OrdinaryDiffEq does" begin
            decay = SciMLBase.ODEProblem((du, u, p, t) -> (du .= -u; nothing), [1.0], (0.0, 1.0))
            stop(sp) = SciMLBase.ContinuousCallback(
                (u, t, integ) -> t - 0.55, SciMLBase.terminate!; save_positions = sp,
            )
            disc(sp, at) = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t == at, SciMLBase.terminate!; save_positions = sp,
            )
            near(a, b) = length(a) == length(b) && all(abs.(a .- b) .< 1.0e-9)
            at = [0.0, 0.5, 1.0]
            for alg in (PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRK("3bs"))
                kw = (dt = 0.1, abstol = 1.0e-10, reltol = 1.0e-10)
                ended(cb; extra...) = SciMLBase.solve(decay, alg; kw..., callback = cb, extra...)
                sol = ended(stop((true, true)); saveat = at)
                @test sol.retcode == SciMLBase.ReturnCode.Terminated
                @test near(sol.t, [0.0, 0.5, 0.55, 0.55])
                @test sol.u[3] == sol.u[4]
                @test abs(sol.u[4][1] - exp(-0.55)) < 1.0e-8
                @test near(ended(stop((false, false)); saveat = at).t, [0.0, 0.5])
                @test near(ended(stop((false, false)); saveat = at, save_end = true).t, [0.0, 0.5, 0.55])
                @test ended(stop((true, true)); saveat = at, save_on = false).t == [0.0]
                @test near(
                    ended(stop((true, true)); saveat = at, save_on = false, save_end = true).t,
                    [0.0, 0.55],
                )
                @test near(
                    ended(stop((true, true)); saveat = [0.0, 0.55, 1.0], save_on = false).t,
                    [0.0, 0.55],
                )
                every = ended(stop((false, false)); save_end = false)
                @test length(every.t) > 3 && abs(every.t[end] - 0.55) < 1.0e-9
                post = ended(stop((false, true)))
                @test near(post.t[(end - 1):end], [0.55, 0.55]) && post.u[end - 1] == post.u[end]
                bump = SciMLBase.ContinuousCallback(
                    (u, t, integ) -> t - 0.55,
                    integ -> (integ.u[1] += 1.0; SciMLBase.terminate!(integ));
                    save_positions = (true, false),
                )
                @test abs(ended(bump).u[end][1] - exp(-0.55)) < 1.0e-8
                vec = SciMLBase.VectorContinuousCallback(
                    (out, u, t, integ) -> (out[1] = t - 0.55; nothing),
                    (integ, mask) -> SciMLBase.terminate!(integ), 1,
                )
                @test near(ended(vec; saveat = at).t, [0.0, 0.5, 0.55, 0.55])
                @test ended(disc((true, true), 0.55); saveat = at, tstops = [0.55]).t ==
                    [0.0, 0.5, 0.55, 0.55]
                @test ended(disc((false, false), 0.55); saveat = at, tstops = [0.55]).t == [0.0, 0.5]
                @test SciMLBase.solve(decay, alg; kw..., saveat = at, maxiters = 5).t == [0.0]
                @test SciMLBase.solve(
                    decay, alg; kw..., saveat = at, unstable_check = (dt, u, p, t) -> t > 0.55,
                ).t == [0.0, 0.5]
                off = (save_on = false, tstops = [0.5])
                @test ended(disc((true, true), 0.5); off..., saveat = 0.25).t == [0.0]
                @test ended(disc((true, true), 0.5); off..., saveat = 0.5).t == [0.0, 0.5]
            end
        end

        @testset "initialize and finalize run" begin
            n_init, n_final = Ref(0), Ref(0)
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.5, integ -> nothing;
                initialize = (c, u, t, integ) -> (n_init[] += 1; nothing),
                finalize = (c, u, t, integ) -> (n_final[] += 1; nothing),
            )
            SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = cb)
            @test n_init[] == 1
            @test n_final[] == 1
        end

        @testset "in a CallbackSet beside a discrete callback" begin
            hits, ticks = Float64[], Float64[]
            discrete = SciMLBase.DiscreteCallback(
                (u, t, integ) -> true, integ -> push!(ticks, integ.t),
            )
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, reltol = 1.0e-10, abstol = 1.0e-12,
                callback = SciMLBase.CallbackSet(jump(hits), discrete),
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test length(hits) == 2
            @test !isempty(ticks)
            @test abs(sol.u[end][1] - after(2.0)) < 1.0e-8
        end

        @testset "rootfind = NoRootFind fires at the step end" begin
            hits = Float64[]
            cb = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.5,
                integ -> (push!(hits, integ.t); integ.u[1] += 1.0);
                rootfind = SciMLBase.NoRootFind,
            )
            SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, callback = cb)
            @test !isempty(hits)
            @test hits[1] > log(2.0)
            @test abs(hits[1] - 0.7) < 1.0e-8
        end

        @testset "without root finding a repeat is judged against zero" begin
            function downs(; kw...)
                found = Float64[]
                cb = SciMLBase.ContinuousCallback(
                    (u, t, integ) -> (u[1] - 0.25) * (0.3005 - u[1]), integ -> nothing,
                    integ -> push!(found, integ.t); rootfind = SciMLBase.NoRootFind, kw...,
                )
                SciMLBase.solve(
                    SciMLBase.ODEProblem(ramp!, [0.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
                    dt = 0.1, adaptive = false, callback = cb,
                )
                return found
            end
            found = downs()
            @test length(found) == 1 && abs(found[1] - 0.4) < 1.0e-12
            @test isempty(downs(abstol = 1.0e-2))
        end

        @testset "an unsupported callback kind is rejected" begin
            struct OddCallback <: SciMLBase.AbstractContinuousCallback end
            @test_throws ArgumentError SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = OddCallback(),
            )
        end
    end

    @testset "tstops" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))

        @testset "solve lands on every stop exactly" begin
            for (alg, kw) in (
                    (PETScDiffEq.TSRK("5dp"), (reltol = 1.0e-8, abstol = 1.0e-10)),
                    (PETScDiffEq.TSRK("5dp"), (adaptive = false,)),
                    (PETScDiffEq.TSImplicit("bdf"), (reltol = 1.0e-8, abstol = 1.0e-10)),
                )
                sol = @test_logs min_level = Logging.Warn SciMLBase.solve(
                    prob, alg; dt = 0.1, tstops = [0.55, 0.25], kw...,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test 0.25 in sol.t
                @test 0.55 in sol.t
                @test issorted(sol.t)
                @test sol.t[end] == 1.0
                @test abs(sol.u[end][1] - exp(-1)) < 1.0e-5
            end
        end

        @testset "a fixed dt resumes after the stop" begin
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, tstops = [0.25],
            )
            i = findfirst(==(0.25), sol.t)
            @test i !== nothing
            @test sol.t[i + 1] - sol.t[i] ≈ 0.1
            @test sol.t[i - 1] < 0.25
        end

        @testset "stops closer together than dt" begin
            for kw in ((adaptive = false,), (reltol = 1.0e-8, abstol = 1.0e-10))
                sol = SciMLBase.solve(
                    prob, PETScDiffEq.TSRK("5dp"); dt = 0.4, tstops = [0.15, 0.17], kw...,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test 0.15 in sol.t
                @test 0.17 in sol.t
                @test sol.t[end] == 1.0
                @test maximum(
                    abs(sol.u[i][1] - exp(-sol.t[i])) for i in eachindex(sol.t)
                ) < 1.0e-3
            end
        end

        @testset "a stop nearer than dt from the start" begin
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.4, adaptive = false, tstops = [0.05],
            )
            @test sol.t[1:2] == [0.0, 0.05]
            @test sol.t[3] ≈ 0.45
        end

        @testset "stops outside the span are ignored" begin
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, tstops = [0.0, 1.0, 1.5],
            )
            @test sol.t[end] == 1.0
            @test sol.retcode == SciMLBase.ReturnCode.Success
        end

        @testset "integrator queue" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1)
            @test SciMLBase.has_tstop(integ) && SciMLBase.first_tstop(integ) == 1.0
            SciMLBase.add_tstop!(integ, 0.33)
            SciMLBase.add_tstop!(integ, 0.33)
            SciMLBase.add_tstop!(integ, 0.15)
            @test SciMLBase.has_tstop(integ)
            @test SciMLBase.first_tstop(integ) == 0.15
            @test integ.tstops == [0.15, 0.33, 1.0]
            @test SciMLBase.pop_tstop!(integ) == 0.15
            while integ.t < 0.33
                SciMLBase.step!(integ)
            end
            @test integ.t == 0.33
            @test integ.tstops == [1.0]
            @test_throws ArgumentError SciMLBase.add_tstop!(integ, 0.1)
            @test_throws ArgumentError SciMLBase.add_tstop!(integ, 1.5)
            SciMLBase.terminate!(integ)
        end

        @testset "the queue is keyed on the direction of integration" begin
            back = SciMLBase.ODEProblem(decay!, [1.0], (1.0, 0.0))
            integ = SciMLBase.init(back, PETScDiffEq.TSRK("5dp"); dt = 0.1)
            @test integ.tdir == -1
            SciMLBase.add_tstop!(integ, 0.5)
            @test SciMLBase.first_tstop(integ) == -0.5
            @test integ.tdir * SciMLBase.first_tstop(integ) == 0.5
            @test SciMLBase.pop_tstop!(integ) == -0.5
            @test SciMLBase.has_tstop(integ) && SciMLBase.first_tstop(integ) == -0.0
            SciMLBase.terminate!(integ)
        end

        @testset "every tstop accessor reads the one queue" begin
            DEB = PETScDiffEq.DiffEqBase
            # `queued` holds keys `tdir * t`, each stop once, as Tsit5 does.
            function check_queue(integ, queued)
                @test DEB.get_tstops(integ) === integ.tstops
                @test DEB.get_tstops_array(integ) === integ.tstops
                @test DEB.get_tstops(integ) == queued
                @test SciMLBase.has_tstop(integ) == !isempty(queued)
                if isempty(queued)
                    @test_throws BoundsError SciMLBase.first_tstop(integ)
                    @test_throws BoundsError DEB.get_tstops_max(integ)
                else
                    @test SciMLBase.has_tstop(integ) && SciMLBase.first_tstop(integ) == queued[1]
                    @test SciMLBase.has_tstop(integ) && DEB.get_tstops_max(integ) == queued[end]
                end
                return nothing
            end

            @testset "forward" begin
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1)
                check_queue(integ, [1.0])
                SciMLBase.add_tstop!(integ, 0.35)
                check_queue(integ, [0.35, 1.0])
                SciMLBase.add_tstop!(integ, 1.0)
                check_queue(integ, [0.35, 1.0])
                SciMLBase.add_tstop!(integ, 0.7)
                while integ.t < 0.35
                    SciMLBase.step!(integ)
                end
                @test integ.t == 0.35
                check_queue(integ, [0.7, 1.0])
                SciMLBase.step!(integ)
                @test 0.35 < integ.t < 0.7
                check_queue(integ, [0.7, 1.0])
                SciMLBase.solve!(integ)
                @test integ.t == 1.0
                check_queue(integ, Float64[])
            end

            @testset "reversed span" begin
                back = SciMLBase.ODEProblem(decay!, [1.0], (1.0, 0.0))
                integ = SciMLBase.init(back, PETScDiffEq.TSRK("5dp"); dt = 0.1)
                check_queue(integ, [-0.0])
                SciMLBase.add_tstop!(integ, 0.5)
                check_queue(integ, [-0.5, -0.0])
                SciMLBase.add_tstop!(integ, 0.0)
                check_queue(integ, [-0.5, -0.0])
                while integ.t > 0.5
                    SciMLBase.step!(integ)
                end
                @test integ.t == 0.5
                check_queue(integ, [-0.0])
                SciMLBase.solve!(integ)
                @test integ.t == 0.0
                check_queue(integ, Float64[])
            end

            @testset "pop_tstop! takes the final time last" begin
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, tstops = [0.35])
                @test SciMLBase.pop_tstop!(integ) == 0.35
                @test SciMLBase.has_tstop(integ) && SciMLBase.pop_tstop!(integ) == 1.0
                check_queue(integ, Float64[])
                SciMLBase.terminate!(integ)
            end

            @testset "terminate! leaves no stop queued" begin
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, tstops = [0.35])
                SciMLBase.step!(integ)
                SciMLBase.terminate!(integ)
                check_queue(integ, Float64[])
            end

            @testset "reinit! queues the final time of the new span once" begin
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, tstops = [0.25])
                check_queue(integ, [0.25, 1.0])
                SciMLBase.solve!(integ)
                SciMLBase.reinit!(integ)
                check_queue(integ, [0.25, 1.0])
                SciMLBase.reinit!(integ; tf = 2.0)
                check_queue(integ, [0.25, 2.0])
                SciMLBase.terminate!(integ)
            end

            @testset "reinit! queues the stops given to init that lie in the new span" begin
                integ = SciMLBase.init(
                    prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, tstops = [0.25, 0.75, 1.5],
                )
                check_queue(integ, [0.25, 0.75, 1.0])
                SciMLBase.reinit!(integ; tf = 0.5)
                check_queue(integ, [0.25, 0.5])
                SciMLBase.reinit!(integ)
                check_queue(integ, [0.25, 0.75, 1.0])
                SciMLBase.reinit!(integ; tf = 2.0)
                check_queue(integ, [0.25, 0.75, 1.5, 2.0])
                SciMLBase.reinit!(integ; tstops = [0.6])
                check_queue(integ, [0.6, 1.0])
                SciMLBase.reinit!(integ)
                check_queue(integ, [0.25, 0.75, 1.0])
                SciMLBase.terminate!(integ)

                back = SciMLBase.ODEProblem(decay!, [1.0], (1.0, 0.0))
                integ = SciMLBase.init(back, PETScDiffEq.TSRK("5dp"); dt = 0.1, tstops = [0.25, 0.75])
                SciMLBase.reinit!(integ; tf = 0.5)
                check_queue(integ, [-0.75, -0.5])
                SciMLBase.reinit!(integ)
                check_queue(integ, [-0.75, -0.25, -0.0])
                SciMLBase.terminate!(integ)
            end
        end

        @testset "a discrete callback at a stop sees it at the head of the queue" begin
            head(integ) = SciMLBase.has_tstop(integ) ?
                (SciMLBase.first_tstop(integ), PETScDiffEq.DiffEqBase.get_tstops_max(integ)) :
                (NaN, NaN)
            for (span, stops) in (((0.0, 1.0), [0.35, 0.5]), ((1.0, 0.0), [0.5, 0.35]))
                seen = Tuple{Float64, Float64, Float64}[]
                cb = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> t in (stops..., span[2]),
                    integ -> (
                        push!(seen, (integ.t, head(integ)...));
                        SciMLBase.derivative_discontinuity!(integ, false)
                    ),
                )
                sol = SciMLBase.solve(
                    SciMLBase.ODEProblem(decay!, [1.0], span), PETScDiffEq.TSRK("5dp");
                    dt = 0.1, tstops = stops, callback = cb,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                tdir = sign(span[2] - span[1])
                @test seen == [(t, tdir * t, tdir * span[2]) for t in (stops..., span[2])]
            end
        end

        @testset "a callback that schedules its own ticks reaches the end" begin
            for (name, make, stops) in (
                    (
                        "IterativeCallback",
                        fires -> DiffEqCallbacks.IterativeCallback(
                            integ -> integ.t + 0.3,
                            integ -> (push!(fires, integ.t); nothing),
                        ),
                        Float64[],
                    ),
                    (
                        "PeriodicCallback",
                        fires -> DiffEqCallbacks.PeriodicCallback(
                            integ -> (push!(fires, integ.t); nothing), 0.3,
                        ),
                        [0.35],
                    ),
                )
                @testset "$name" begin
                    fires = Float64[]
                    sol = SciMLBase.solve(
                        prob, PETScDiffEq.TSRK("5dp"); dt = 0.1,
                        callback = make(fires), tstops = stops,
                    )
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test length(fires) == 3 && isapprox(fires, [0.3, 0.6, 0.9]; atol = 1.0e-9)
                end
            end
        end

        @testset "a tick that lands on the final time fires there" begin
            for (span, tick, want) in (
                    ((0.0, 1.0), 0.25, [0.25, 0.5, 0.75, 1.0]),
                    ((1.0, 0.0), -0.25, [0.75, 0.5, 0.25, 0.0]),
                )
                @testset "IterativeCallback on $span" begin
                    fires = Float64[]
                    cb = DiffEqCallbacks.IterativeCallback(
                        integ -> integ.t + tick, integ -> (push!(fires, integ.t); nothing),
                    )
                    sol = SciMLBase.solve(
                        SciMLBase.ODEProblem(decay!, [1.0], span), PETScDiffEq.TSRK("5dp");
                        dt = 0.1, callback = cb,
                    )
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test fires == want
                end
                @testset "PeriodicCallback with final_affect on $span" begin
                    fires = Float64[]
                    cb = DiffEqCallbacks.PeriodicCallback(
                        integ -> push!(fires, integ.t), tick; final_affect = true,
                    )
                    sol = SciMLBase.solve(
                        SciMLBase.ODEProblem(decay!, [1.0], span), PETScDiffEq.TSRK("5dp");
                        dt = 0.1, callback = cb,
                    )
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test fires == want
                end
            end
        end

        @testset "PresetTimeCallback" begin
            expected(t) = t < 0.5 ? exp(-t) : (exp(-0.5) + 1) * exp(-(t - 0.5))
            hits = Float64[]
            cb = PresetTimeCallback([0.5], integ -> (push!(hits, integ.t); integ.u[1] += 1.0))
            for (alg, tol) in (
                    (PETScDiffEq.TSRK("5dp"), 1.0e-5), (PETScDiffEq.TSImplicit("bdf"), 5.0e-3),
                )
                empty!(hits)
                sol = SciMLBase.solve(
                    prob, alg; dt = 0.1, reltol = 1.0e-8, abstol = 1.0e-10, callback = cb,
                )
                @test hits == [0.5]
                @test 0.5 in sol.t
                @test abs(sol.u[end][1] - expected(1.0)) < tol
                @test abs(sol(0.75)[1] - expected(0.75)) < tol
            end
        end

        @testset "reinit! restores the stops given to init" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, tstops = [0.25])
            @test 0.25 in SciMLBase.solve!(integ).t
            SciMLBase.reinit!(integ)
            @test 0.25 in SciMLBase.solve!(integ).t
            SciMLBase.reinit!(integ; tstops = [0.4])
            sol = SciMLBase.solve!(integ)
            @test 0.4 in sol.t
            @test !(0.25 in sol.t)
        end
    end

    @testset "SplitODEProblem uses the analytic Jacobian of f1" begin
        stiff!(du, u, p, t) = (du[1] = -1000.0 * (u[1] - cos(t)); nothing)
        forcing!(du, u, p, t) = (du[1] = -sin(t); nothing)
        stiff_jac!(J, u, p, t) = (J[1, 1] = -1000.0; nothing)
        stiff_wrong_jac!(J, u, p, t) = (J[1, 1] = 5000.0; nothing)
        alg = PETScDiffEq.TSARKIMEX("3", ["-ts_adapt_type", "none"])
        tspan = (0.0, 0.1)
        split(f1) = SciMLBase.SplitODEProblem(f1, forcing!, [1.0], tspan)

        plain = SciMLBase.solve(
            split(stiff!),
            PETScDiffEq.TSARKIMEX("3", ["-ts_adapt_type", "none"]; autodiff = PETScDiffEq.AutoFiniteDiff());
            dt = 1.0e-3,
        )
        withjac = SciMLBase.solve(
            split(SciMLBase.ODEFunction(stiff!; jac = stiff_jac!)), alg; dt = 1.0e-3,
        )
        onsplit = SciMLBase.solve(
            SciMLBase.SplitODEProblem(
                SciMLBase.SplitFunction{true}(stiff!, forcing!; jac = stiff_jac!),
                [1.0], tspan,
            ),
            alg; dt = 1.0e-3,
        )
        @test plain.stats.njacs == 0
        @test withjac.stats.njacs > 0
        @test withjac.stats.nf < plain.stats.nf
        @test withjac.retcode == SciMLBase.ReturnCode.Success
        @test abs(withjac.u[end][1] - cos(0.1)) < 1.0e-6
        @test withjac.u[end] ≈ plain.u[end] atol = 1.0e-10
        @test onsplit.stats.njacs == withjac.stats.njacs
        @test onsplit.u[end] == withjac.u[end]

        @test SciMLBase.solve(
            split(SciMLBase.ODEFunction(stiff!; jac = stiff_wrong_jac!)),
            PETScDiffEq.TSARKIMEX("3", ["-ts_max_snes_failures", "1"]); dt = 1.0e-3,
        ).retcode != SciMLBase.ReturnCode.Success

        @testset "with a sparse jac_prototype" begin
            stiff2!(du, u, p, t) = (
                du[1] = -1000.0 * (u[1] - cos(t)); du[2] = -2000.0 * (u[2] - sin(t)); nothing
            )
            forcing2!(du, u, p, t) = (du[1] = -sin(t); du[2] = cos(t); nothing)
            jac2!(J, u, p, t) = (J[1, 1] = -1000.0; J[2, 2] = -2000.0; nothing)
            proto = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
            sol = SciMLBase.solve(
                SciMLBase.SplitODEProblem(
                    SciMLBase.ODEFunction(stiff2!; jac = jac2!, jac_prototype = proto),
                    forcing2!, [1.0, 0.0], tspan,
                ),
                alg; dt = 1.0e-3,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.stats.njacs > 0
            @test maximum(abs.(sol.u[end] .- [cos(0.1), sin(0.1)])) < 1.0e-6
        end
    end

    @testset "every algorithm but TSARKIMEX solves a SplitODEProblem as the sum of its parts" begin
        fast!(du, u, p, t) = (du[1] = -p * u[1]; du[2] = -2p * u[2]; nothing)
        slow!(du, u, p, t) = (du[1] = u[2] * cos(t); du[2] = -u[1] * u[2]; nothing)
        function summed!(du, u, p, t)
            rest = similar(du)
            fast!(du, u, p, t)
            slow!(rest, u, p, t)
            du .+= rest
            return nothing
        end
        fast_jac!(J, u, p, t) = (J[1, 1] = -p; J[2, 2] = -2p; nothing)
        slow_jac!(J, u, p, t) = (J[1, 2] = cos(t); J[2, 1] = -u[2]; J[2, 2] = -u[1]; nothing)
        summed_jac!(J, u, p, t) =
            (J[1, 1] = -p; J[1, 2] = cos(t); J[2, 1] = -u[2]; J[2, 2] = -2p - u[1]; nothing)
        fast(u, p, t) = [-p * u[1], -2p * u[2]]
        slow(u, p, t) = [u[2] * cos(t), -u[1] * u[2]]
        summed(u, p, t) = fast(u, p, t) + slow(u, p, t)
        fast_jac(u, p, t) = [-p 0.0; 0.0 -2p]
        slow_jac(u, p, t) = [0.0 cos(t); -u[2] -u[1]]
        summed_jac(u, p, t) = fast_jac(u, p, t) + slow_jac(u, p, t)
        fast_proto = sparse([1, 2], [1, 2], ones(2))
        slow_proto = sparse([1, 2, 2], [2, 1, 2], ones(3))
        full, mass = sparse(ones(2, 2)), Diagonal([2.0, 1.0])
        u0, span, p = [1.0, 0.5], (0.0, 1.0), 3.0
        of(f; kw...) = SciMLBase.ODEFunction(f; kw...)
        parts(f1, f2; kw...) =
            SciMLBase.SplitODEProblem(SciMLBase.SplitFunction(f1, f2; kw...), u0, span, p)
        whole(f; kw...) = SciMLBase.ODEProblem(of(f; kw...), u0, span, p)
        tol = (abstol = 1.0e-8, reltol = 1.0e-8)
        function same(split, plain, alg; kw...)
            sol = SciMLBase.solve(split, alg; kw...)
            ref = SciMLBase.solve(plain, alg; kw...)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.t == ref.t
            @test sol.u == ref.u
            @test sol(0.37) == ref(0.37)
            @test (sol.stats.nf, sol.stats.nf2) == (ref.stats.nf, 0)
            @test sol.stats.njacs == ref.stats.njacs
            return sol
        end
        bdf, fd = TSImplicit("bdf"), PETScDiffEq.AutoFiniteDiff()
        split, plain = parts(fast!, slow!), whole(summed!)
        for alg in (TSRK("5dp"), bdf, TSRosW(; autodiff = fd))
            same(split, plain, alg; tol...)
        end
        for alg in (
                TSGeneric("ssp"; explicit = true), TSIRK(2), TSMPRK([1], "2a22"),
            )
            same(split, plain, alg; dt = 0.01)
        end
        same(parts(fast, slow), whole(summed), bdf; tol...)
        both = parts(of(fast!; jac = fast_jac!), of(slow!; jac = slow_jac!))
        with_jac = same(both, whole(summed!; jac = summed_jac!), TSRosW(); tol...)
        @test with_jac.stats.nf < SciMLBase.solve(split, TSRosW(); tol...).stats.nf
        same(
            parts(of(fast; jac = fast_jac), of(slow; jac = slow_jac)),
            whole(summed; jac = summed_jac), TSRosW(); tol...,
        )
        same(parts(of(fast!; jac = fast_jac!), slow!), plain, bdf; tol...)
        same(
            parts(of(fast!; jac_prototype = fast_proto), of(slow!; jac_prototype = slow_proto)),
            whole(summed!; jac_prototype = full), TSRosW(); tol...,
        )
        sparse_both = parts(
            of(fast!; jac = fast_jac!, jac_prototype = fast_proto),
            of(slow!; jac = slow_jac!, jac_prototype = slow_proto),
        )
        sparse_whole = whole(summed!; jac = summed_jac!, jac_prototype = full)
        same(sparse_both, sparse_whole, bdf; tol...)
        same(parts(of(fast!; jac_prototype = fast_proto), slow!), plain, bdf; tol...)
        same(
            parts(fast!, slow!; mass_matrix = mass), whole(summed!; mass_matrix = mass), bdf;
            tol...,
        )
        bump = SciMLBase.ContinuousCallback(
            (u, t, integ) -> u[1] - 0.5, integ -> (integ.u[2] += 0.1; nothing),
        )
        same(split, plain, TSRK("5dp"); callback = bump, tol...)
        function stepped(prob)
            integ = SciMLBase.init(prob, bdf; tol...)
            SciMLBase.step!(integ)
            mid = (integ.t, copy(integ.u), SciMLBase.get_du(integ), integ.sol.stats.nf2)
            SciMLBase.reinit!(integ)
            sol = SciMLBase.solve!(integ)
            return mid, sol.t, sol.u, sol.stats.nf, sol.stats.nf2
        end
        @test stepped(sparse_both) == stepped(sparse_whole)
        @test SciMLBase.solve(split, TSARKIMEX("3"); tol...).stats.nf2 > 0
        if Sys.WORD_SIZE == 64
            world = MPI.COMM_WORLD
            same(split, plain, TSRK("5dp"; comm = world); tol...)
            same(sparse_both, sparse_whole, TSImplicit("bdf"; comm = world); tol...)
        end
        zero!(du, u, p, t) = (du .= 0.0; nothing)
        @test_throws "operator-valued" SciMLBase.solve(
            SciMLBase.SplitODEProblem(
                SciMLOperators.MatrixOperator([-1.0 0.0; 0.0 -2.0]), zero!, u0, span,
            ), TSRK("5dp"),
        )
        @test_throws "have to be one size" SciMLBase.solve(
            parts(of(fast!; jac_prototype = fast_proto), of(slow!; jac_prototype = sparse(ones(3, 3)))),
            bdf,
        )

        PETSc, LibPETSc = PETScDiffEq.PETSc, PETScDiffEq.LibPETSc
        pl = PETSc.getlib(; PetscScalar = Float64)
        da = PETSc.DMDA(pl, MPI.COMM_SELF, (LibPETSc.DM_BOUNDARY_GHOSTED,), (5,), 1, 1)
        function spread!(du, u, da, t)
            U, D = PETScDiffEq.reshape_local_array(u, da), PETScDiffEq.reshape_local_array(du, da)
            for i in axes(D, 2)
                D[1, i] = U[1, i - 1] - 2U[1, i] + U[1, i + 1]
            end
            return nothing
        end
        function react!(du, u, da, t)
            U, D = PETScDiffEq.reshape_local_array(u, da), PETScDiffEq.reshape_local_array(du, da)
            for i in axes(D, 2)
                D[1, i] = U[1, i] * (1 - U[1, i])
            end
            return nothing
        end
        function spread_react!(du, u, da, t)
            other = similar(du)
            spread!(du, u, da, t)
            react!(other, u, da, t)
            du .+= other
            return nothing
        end
        unused_jac!(J, u, da, t) = nothing
        x0 = [0.1, 0.4, 0.9, 0.4, 0.1]
        on_dm(f1, f2) = SciMLBase.SplitODEProblem(f1, f2, x0, span, da)
        dm_whole = SciMLBase.ODEProblem(spread_react!, x0, span, da)
        for alg in (TSRK("5dp"; dm = da), TSImplicit("bdf"; dm = da))
            same(on_dm(spread!, react!), dm_whole, alg; tol...)
        end
        dm_both = on_dm(of(spread!; jac = unused_jac!), of(react!; jac = unused_jac!))
        same(dm_both, dm_whole, TSRK("5dp"; dm = da); tol...)
        @test_throws "cannot add the `jac`s" SciMLBase.solve(
            dm_both, TSImplicit("bdf"; dm = da); tol...,
        )
        PETScDiffEq.PETScCompat.destroy!(da)
    end

    @testset "without a jac the Jacobian is differentiated, as OrdinaryDiffEq does" begin
        function rober!(du, u, p, t)
            du[1] = -0.04u[1] + 1.0e4 * u[2] * u[3]
            du[2] = 0.04u[1] - 1.0e4 * u[2] * u[3] - 3.0e7 * u[2]^2
            du[3] = 3.0e7 * u[2]^2
            return nothing
        end
        function rober_jac!(J, u, p, t)
            J[1, 1] = -0.04; J[1, 2] = 1.0e4 * u[3]; J[1, 3] = 1.0e4 * u[2]
            J[2, 1] = 0.04; J[2, 2] = -1.0e4 * u[3] - 6.0e7 * u[2]; J[2, 3] = -1.0e4 * u[2]
            J[3, 1] = 0.0; J[3, 2] = 6.0e7 * u[2]; J[3, 3] = 0.0
            return nothing
        end
        fd = PETScDiffEq.AutoFiniteDiff()

        @testset "Robertson solves as it does with the analytic one" begin
            ref = [0.2083340149701255e-7, 0.8333360770334713e-13, 0.999999979166505]
            span = (0.0, 1.0e11)
            # On 32-bit, PETSc's `bad hmax` check can stop BDF and ARKIMEX at t = 1e11.
            algs = Sys.WORD_SIZE == 64 ?
                (
                    PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSRosW(),
                    PETScDiffEq.TSARKIMEX("4"),
                ) : (PETScDiffEq.TSRosW(),)
            for alg in algs
                withjac = SciMLBase.solve(
                    SciMLBase.ODEProblem(
                        SciMLBase.ODEFunction(rober!; jac = rober_jac!), [1.0, 0.0, 0.0], span,
                    ), alg; abstol = 1.0e-10, reltol = 1.0e-6,
                )
                plain = SciMLBase.solve(
                    SciMLBase.ODEProblem(rober!, [1.0, 0.0, 0.0], span), alg;
                    abstol = 1.0e-10, reltol = 1.0e-6,
                )
                @test plain.retcode == SciMLBase.ReturnCode.Success
                @test plain.t == withjac.t
                @test plain.u[end] ≈ withjac.u[end] rtol = 1.0e-12
                @test maximum(abs.(plain.u[end] .- ref) ./ ref) < 0.2
                @test plain.stats.njacs == withjac.stats.njacs
            end
        end

        @testset "a sparse prototype is coloured" begin
            n = 100
            calls = Ref(0)
            function heat!(du, u, p, t)
                calls[] += 1
                h2 = (n + 1)^2
                for i in 1:n
                    l = i > 1 ? u[i - 1] : zero(eltype(u))
                    r = i < n ? u[i + 1] : zero(eltype(u))
                    du[i] = h2 * (l - 2u[i] + r) - u[i]^3
                end
                return nothing
            end
            function heat_jac!(J, u, p, t)
                h2 = (n + 1)^2
                fill!(nonzeros(J), 0.0)
                for i in 1:n
                    J[i, i] = -2h2 - 3u[i]^2
                    i > 1 && (J[i, i - 1] = h2)
                    i < n && (J[i, i + 1] = h2)
                end
                return nothing
            end
            proto = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
            u0 = [sin(pi * i / (n + 1)) for i in 1:n]
            function run(f, alg)
                calls[] = 0
                sol = SciMLBase.solve(
                    SciMLBase.ODEProblem(f, u0, (0.0, 0.1)), alg;
                    abstol = 1.0e-6, reltol = 1.0e-6,
                )
                return sol, calls[]
            end
            for make in (
                    ad -> PETScDiffEq.TSImplicit("bdf"; autodiff = ad),
                    ad -> PETScDiffEq.TSRosW(; autodiff = ad),
                )
                ad = PETScDiffEq.AutoForwardDiff()
                given, ncalls = run(
                    SciMLBase.ODEFunction(heat!; jac = heat_jac!, jac_prototype = proto),
                    make(ad),
                )
                coloured, ncoloured = run(
                    SciMLBase.ODEFunction(heat!; jac_prototype = proto), make(ad),
                )
                @test coloured.u[end] ≈ given.u[end] rtol = 1.0e-12
                @test ncoloured < ncalls + 3 * coloured.stats.njacs
                dense, ndense = run(SciMLBase.ODEFunction(heat!), make(fd))
                sparse_fd, nsparse = run(
                    SciMLBase.ODEFunction(heat!; jac_prototype = proto), make(fd),
                )
                @test sparse_fd.retcode == SciMLBase.ReturnCode.Success
                @test sparse_fd.u[end] ≈ given.u[end] rtol = 1.0e-8
                @test nsparse < ndense / 5
            end
        end

        @testset "an off-diagonal mass matrix with a sparse prototype" begin
            f!(du, u, p, t) = (du[1] = -u[1] + u[2]^2; du[2] = -10u[2]; nothing)
            f_jac!(J, u, p, t) = (J[1, 1] = -1.0; J[1, 2] = 2u[2]; J[2, 2] = -10.0; nothing)
            M = [1.0 0.5; 0.0 1.0]
            exact = [exp(-1) * (1 + (1 - exp(-19)) / 19 + 5 * (1 - exp(-9)) / 9), exp(-10)]
            proto = sparse([1.0 1.0; 0.0 1.0])
            kw = (; abstol = 1.0e-10, reltol = 1.0e-10)
            for make in (
                    ad -> PETScDiffEq.TSImplicit("bdf"; autodiff = ad),
                    ad -> PETScDiffEq.TSRosW(; autodiff = ad),
                )
                run(f, ad) = SciMLBase.solve(
                    SciMLBase.ODEProblem(f, [1.0, 1.0], (0.0, 1.0)), make(ad); kw...,
                )
                ad = PETScDiffEq.AutoForwardDiff()
                dense = run(SciMLBase.ODEFunction(f!; mass_matrix = M), ad)
                @test maximum(abs.(dense.u[end] .- exact)) < 1.0e-6
                for (f, backend) in (
                        (SciMLBase.ODEFunction(f!; jac = f_jac!, jac_prototype = proto, mass_matrix = M), ad),
                        (SciMLBase.ODEFunction(f!; jac_prototype = proto, mass_matrix = M), ad),
                        (SciMLBase.ODEFunction(f!; jac_prototype = proto, mass_matrix = M), fd),
                    )
                    sol = run(f, backend)
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test sol.u[end] ≈ dense.u[end] rtol = 1.0e-10
                end
            end
            # The shift lands on M's (1, 2) entry, which the prototype leaves out.
            g!(du, u, p, t) = (du[1] = -u[1]; du[2] = -10u[2]; nothing)
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(g!; jac_prototype = sparse(1.0I, 2, 2), mass_matrix = M),
                    [1.0, 1.0], (0.0, 1.0),
                ),
                PETScDiffEq.TSImplicit("bdf"); kw...,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test maximum(abs.(sol.u[end] .- [exp(-1) * (1 + 5 * (1 - exp(-9)) / 9), exp(-10)])) <
                1.0e-6
        end

        @testset "a DAEProblem differentiates its residual" begin
            function resid!(r, du, u, p, t)
                r[1] = -0.04u[1] + 1.0e4 * u[2] * u[3] - du[1]
                r[2] = 0.04u[1] - 1.0e4 * u[2] * u[3] - 3.0e7 * u[2]^2 - du[2]
                r[3] = u[1] + u[2] + u[3] - 1.0
                return nothing
            end
            function resid_jac!(J, du, u, p, gamma, t)
                J[1, 1] = -0.04 - gamma; J[1, 2] = 1.0e4 * u[3]; J[1, 3] = 1.0e4 * u[2]
                J[2, 1] = 0.04; J[2, 2] = -1.0e4 * u[3] - 6.0e7 * u[2] - gamma
                J[2, 3] = -1.0e4 * u[2]
                J[3, 1] = 1.0; J[3, 2] = 1.0; J[3, 3] = 1.0
                return nothing
            end
            dae(f) = SciMLBase.DAEProblem(
                f, [-0.04, 0.04, 0.0], [1.0, 0.0, 0.0], (0.0, 1.0);
                differential_vars = [true, true, false],
            )
            kw = (; abstol = 1.0e-10, reltol = 1.0e-8)
            given = SciMLBase.solve(
                dae(SciMLBase.DAEFunction(resid!; jac = resid_jac!)), PETScDiffEq.TSDAE(); kw...,
            )
            plain = SciMLBase.solve(dae(SciMLBase.DAEFunction(resid!)), PETScDiffEq.TSDAE(); kw...)
            @test plain.retcode == SciMLBase.ReturnCode.Success
            @test plain.t == given.t
            @test plain.u[end] ≈ given.u[end] rtol = 1.0e-12
        end

        @testset "f1 of a SplitODEProblem is what is differentiated" begin
            f1!(du, u, p, t) = (du .= -100 .* u .+ 100 .* u .^ 3 ./ 3; nothing)
            f1_jac!(J, u, p, t) = (J .= Diagonal(-100 .+ 100 .* u .^ 2); nothing)
            f2!(du, u, p, t) = (du .= cos(t); nothing)
            split(f) = SciMLBase.SplitODEProblem(f, [0.5, 0.2], (0.0, 1.0))
            kw = (; abstol = 1.0e-10, reltol = 1.0e-10)
            given = SciMLBase.solve(
                split(SciMLBase.SplitFunction(f1!, f2!; jac = f1_jac!)),
                PETScDiffEq.TSARKIMEX("3"); kw...,
            )
            plain = SciMLBase.solve(
                split(SciMLBase.SplitFunction(f1!, f2!)), PETScDiffEq.TSARKIMEX("3"); kw...,
            )
            @test plain.t == given.t
            @test plain.u[end] ≈ given.u[end] rtol = 1.0e-12
        end

        @testset "a reversed span differentiates the caller's f" begin
            g!(du, u, p, t) = (du .= -2 .* u .+ sin(t); nothing)
            g_jac!(J, u, p, t) = (J .= -2; nothing)
            kw = (; abstol = 1.0e-10, reltol = 1.0e-10)
            given = SciMLBase.solve(
                SciMLBase.ODEProblem(SciMLBase.ODEFunction(g!; jac = g_jac!), [1.0], (0.0, -1.0)),
                PETScDiffEq.TSImplicit("bdf"); kw...,
            )
            plain = SciMLBase.solve(
                SciMLBase.ODEProblem(g!, [1.0], (0.0, -1.0)), PETScDiffEq.TSImplicit("bdf"); kw...,
            )
            @test plain.t == given.t
            @test plain.u[end] ≈ given.u[end] rtol = 1.0e-12
        end

        @testset "types that need a Jacobian run without a jac" begin
            g!(du, u, p, t) = (du .= -u .^ 2 .+ cos(t); nothing)
            g_jac!(J, u, p, t) = (J[1, 1] = -2u[1]; nothing)
            for (alg, kw) in (
                    (PETScDiffEq.TSIRK(2), (; dt = 0.05, adaptive = false)),
                    (PETScDiffEq.TSRosW("assp3p3s1c"), (; abstol = 1.0e-8, reltol = 1.0e-8)),
                )
                given = SciMLBase.solve(
                    SciMLBase.ODEProblem(SciMLBase.ODEFunction(g!; jac = g_jac!), [1.0], (0.0, 1.0)),
                    alg; kw...,
                )
                plain = SciMLBase.solve(SciMLBase.ODEProblem(g!, [1.0], (0.0, 1.0)), alg; kw...)
                @test plain.retcode == SciMLBase.ReturnCode.Success
                @test plain.t == given.t
                @test plain.u[end] ≈ given.u[end] rtol = 1.0e-12
            end
            prob = SciMLBase.ODEProblem(g!, [1.0], (0.0, 1.0))
            @test_throws r"TSIRK needs a Jacobian" SciMLBase.solve(
                prob, PETScDiffEq.TSIRK(2; autodiff = fd); dt = 0.05,
            )
            @test_throws r"assp3p3s1c.*needs a Jacobian" SciMLBase.solve(
                prob, PETScDiffEq.TSRosW("assp3p3s1c"; autodiff = fd),
            )
        end

        @testset "an f written for Float64 alone is told how to go on" begin
            buf = zeros(1)
            only64!(du, u, p, t) = (buf[1] = u[1]; du[1] = -buf[1]; nothing)
            prob = SciMLBase.ODEProblem(only64!, [1.0], (0.0, 1.0))
            err = try
                SciMLBase.solve(prob, PETScDiffEq.TSImplicit("bdf"))
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("autodiff = PETScDiffEq.AutoFiniteDiff()", err.msg)
            sol = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("bdf"; autodiff = fd))
            @test sol.retcode == SciMLBase.ReturnCode.Success
            typed!(du::Vector{Float64}, u::Vector{Float64}, p, t) = (du .= -u; nothing)
            asserted!(du, u, p, t) = (du[1] = -(u[1]::Float64); nothing)
            for (f, alg) in (
                    (typed!, PETScDiffEq.TSImplicit("bdf")),
                    (asserted!, PETScDiffEq.TSRosW()),
                    (typed!, PETScDiffEq.TSIRK(2)),
                )
                err = try
                    SciMLBase.solve(
                        SciMLBase.ODEProblem(f, [1.0], (0.0, 1.0)), alg; dt = 0.1,
                        adaptive = !(alg isa PETScDiffEq.TSIRK),
                    )
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("PETScDiffEq.AutoFiniteDiff()", err.msg)
            end
        end

        @testset "a non-finite derivative of a finite f is an error, not a stop at t0" begin
            pushed!(du, u, p, t) = (du .= -LinearAlgebra.norm(u) .* u; du[1] += 1.0; nothing)
            prob = SciMLBase.ODEProblem(pushed!, [0.0, 0.0], (0.0, 1.0))
            for alg in (PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSRosW())
                @test_throws r"non-finite entry" SciMLBase.solve(prob, alg)
            end
            @test_throws r"non-finite entry" SciMLBase.solve(
                prob, PETScDiffEq.TSIRK(2); dt = 0.1,
            )
            sol = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("bdf"; autodiff = fd))
            @test sol.retcode == SciMLBase.ReturnCode.Success
        end

        @testset "nf counts the Jacobian's evaluations of f" begin
            duals = Ref(0)
            function lv!(du, u, p, t)
                eltype(u) <: Float64 || (duals[] += 1)
                du[1] = 1.5u[1] - u[1] * u[2]
                du[2] = -3u[2] + u[1] * u[2]
                return nothing
            end
            function lv_jac!(J, u, p, t)
                J[1, 1] = 1.5 - u[2]
                J[1, 2] = -u[1]
                J[2, 1] = u[2]
                J[2, 2] = -3 + u[1]
                return nothing
            end
            kw = (; abstol = 1.0e-8, reltol = 1.0e-8)
            given = SciMLBase.solve(
                SciMLBase.ODEProblem(SciMLBase.ODEFunction(lv!; jac = lv_jac!), [1.0, 1.0], (0.0, 5.0)),
                PETScDiffEq.TSImplicit("bdf"); kw...,
            )
            duals[] = 0
            plain = SciMLBase.solve(
                SciMLBase.ODEProblem(lv!, [1.0, 1.0], (0.0, 5.0)), PETScDiffEq.TSImplicit("bdf");
                kw...,
            )
            @test duals[] > 0
            @test plain.stats.nf == given.stats.nf + duals[]
        end

        @testset "a sparse backend the caller passes colours the prototype" begin
            n = 20
            function lap!(du, u, p, t)
                for i in 1:n
                    du[i] = (i > 1 ? u[i - 1] : 0.0) - 2u[i] + (i < n ? u[i + 1] : 0.0) - u[i]^3
                end
                return nothing
            end
            proto = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
            prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(lap!; jac_prototype = proto), ones(n), (0.0, 1.0),
            )
            default = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("bdf"))
            bare = SciMLBase.solve(
                prob, PETScDiffEq.TSImplicit(
                    "bdf"; autodiff = PETScDiffEq.ADTypes.AutoSparse(PETScDiffEq.AutoForwardDiff()),
                ),
            )
            @test bare.retcode == SciMLBase.ReturnCode.Success
            @test bare.u[end] ≈ default.u[end] rtol = 1.0e-12
        end

        @testset "options that make SNES matrix-free" begin
            f!(du, u, p, t) = (du[1] = -u[1] + u[2]^2; du[2] = -2u[2]; nothing)
            prob = SciMLBase.ODEProblem(f!, [1.0, 0.5], (0.0, 1.0))
            default = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("beuler"); dt = 0.01)
            for opts in (["-snes_mf"], ["-snes_mf_operator"], ["-snes_fd"])
                sol = @test_logs min_level = Logging.Warn SciMLBase.solve(
                    prob, PETScDiffEq.TSImplicit("beuler", opts); dt = 0.01,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.u[end] ≈ default.u[end] rtol = 1.0e-6
            end
        end

        @testset "a zero on the Newton matrix's diagonal" begin
            function resid!(r, du, u, p, t)
                r[1] = u[2] - 0.5 * sin(t) - 0.5
                r[2] = -du[1] - u[1] + u[2]
                return nothing
            end
            function resid_jac!(J, du, u, p, gamma, t)
                J[1, 1] = 0.0
                J[1, 2] = 1.0
                J[2, 1] = -gamma - 1.0
                J[2, 2] = 1.0
                return nothing
            end
            exact = 0.5 + 0.25 * (sin(1.0) - cos(1.0)) + 0.75 * exp(-1.0)
            full = sparse(ones(2, 2))
            for (f, ad) in (
                    (SciMLBase.DAEFunction(resid!), PETScDiffEq.AutoForwardDiff()),
                    (SciMLBase.DAEFunction(resid!; jac_prototype = full), PETScDiffEq.AutoForwardDiff()),
                    (SciMLBase.DAEFunction(resid!; jac = resid_jac!, jac_prototype = full), PETScDiffEq.AutoForwardDiff()),
                    (SciMLBase.DAEFunction(resid!; jac_prototype = full), fd),
                )
                sol = SciMLBase.solve(
                    SciMLBase.DAEProblem(
                        f, [-0.5, 0.0], [1.0, 0.5], (0.0, 1.0); differential_vars = [false, true],
                    ),
                    PETScDiffEq.TSDAE("bdf"; autodiff = ad); abstol = 1.0e-8, reltol = 1.0e-8,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test abs(sol.u[end][1] - exact) < 1.0e-5
            end
        end


        @testset "autodiff takes an ADTypes backend" begin
            @test_throws r"ADTypes backend" PETScDiffEq.TSImplicit("bdf"; autodiff = true)
            @test_throws r"ADTypes backend" PETScDiffEq.TSRosW(; autodiff = :forward)
            @test PETScDiffEq.TSDAE().autodiff isa PETScDiffEq.AutoForwardDiff
        end

        Sys.WORD_SIZE == 64 && @testset "on a communicator the whole pattern is coloured" begin
            n = 30
            h2 = (n + 1)^2
            function cubic_heat!(du, u, p, t)
                for i in 1:n
                    l = i > 1 ? u[i - 1] : zero(eltype(u))
                    r = i < n ? u[i + 1] : zero(eltype(u))
                    du[i] = h2 * (l - 2u[i] + r) - u[i]^3
                end
                return nothing
            end
            function cubic_heat_jac!(J, u, p, t)
                for i in 1:n
                    J[i, i] = -2h2 - 3u[i]^2
                    i > 1 && (J[i, i - 1] = h2)
                    i < n && (J[i, i + 1] = h2)
                end
                return nothing
            end
            proto = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
            problem(; kw...) = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(cubic_heat!; jac_prototype = proto, kw...),
                [sin(pi * i / (n + 1)) for i in 1:n], (0.0, 0.1),
            )
            ad = PETScDiffEq.AutoForwardDiff()
            world = MPI.COMM_WORLD
            tight = (; abstol = 1.0e-8, reltol = 1.0e-8)
            for make in (kw -> PETScDiffEq.TSImplicit("bdf"; kw...), kw -> PETScDiffEq.TSRosW(; kw...))
                sol = SciMLBase.solve(problem(), make((; comm = world, autodiff = ad)); tight...)
                given = SciMLBase.solve(
                    problem(; jac = cubic_heat_jac!), make((; comm = world)); tight...,
                )
                serial = SciMLBase.solve(problem(), make((; autodiff = ad)); tight...)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.stats.njacs == given.stats.njacs > 0
                @test sol.t == given.t
                @test sol.u == given.u
                @test sol.u[end] ≈ serial.u[end] rtol = 1.0e-9
            end
        end
    end

    @testset "TSGeneric convergence order" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        exact = exp(-1.0)
        cases = (
            (
                "euler", PETScDiffEq.TSGeneric(
                    "euler", ["-ts_adapt_type", "none"]; explicit = true,
                ), 1,
            ),
            ("alpha", PETScDiffEq.TSGeneric("alpha", ["-ts_adapt_type", "none"]), 2),
        )
        for (label, alg, expected_order) in cases
            errs = Float64[]
            for dt in (0.1, 0.05, 0.025, 0.0125)
                sol = SciMLBase.solve(prob, alg; dt = dt)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                push!(errs, abs(sol.u[end][1] - exact))
            end
            orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
            @test all(o -> isapprox(o, expected_order; atol = 0.15), orders)
        end
        @test PETScDiffEq.TSGeneric("alpha").explicit == false
        @test PETScDiffEq.TSGeneric("euler"; explicit = true).explicit == true
    end

    @testset "Analytic Jacobian" begin
        exact = exp(-1.0)
        prob = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = decay_jac!), [1.0], (0.0, 1.0),
        )
        wrong_prob = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = decay_wrong_jac!), [1.0], (0.0, 1.0),
        )

        @testset "matches the FD fallback" begin
            no_jac_prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
            for alg in (
                    PETScDiffEq.TSImplicit("cn"), PETScDiffEq.TSImplicit("bdf"),
                    PETScDiffEq.TSRosW("ra34pw2"),
                )
                sol_jac = SciMLBase.solve(prob, alg; dt = 0.05)
                sol_fd = SciMLBase.solve(no_jac_prob, alg; dt = 0.05)
                @test sol_jac.retcode == SciMLBase.ReturnCode.Success
                @test isapprox(sol_jac.u[end][1], sol_fd.u[end][1]; atol = 1.0e-8)
            end
        end

        @testset "convergence order is preserved" begin
            for (alg, expected_order) in (
                    (PETScDiffEq.TSImplicit("cn", ["-ts_adapt_type", "none"]), 2),
                    (PETScDiffEq.TSRosW("ra34pw2", ["-ts_adapt_type", "none"]), 3),
                )
                errs = Float64[]
                for dt in (0.1, 0.05, 0.025, 0.0125)
                    sol = SciMLBase.solve(prob, alg; dt = dt)
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    push!(errs, abs(sol.u[end][1] - exact))
                end
                orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
                @test all(o -> isapprox(o, expected_order; atol = 0.15), orders)
            end
        end

        @testset "a wrong Jacobian breaks Newton convergence" begin
            @test SciMLBase.solve(
                wrong_prob, PETScDiffEq.TSImplicit("cn", ["-ts_max_snes_failures", "1"]); dt = 0.05,
            ).retcode != SciMLBase.ReturnCode.Success
        end

        @testset "ignored by algorithms that don't use IFunction" begin
            sol = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.05)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test abs(sol.u[end][1] - exact) < 1.0e-5
        end

        @testset "an asymmetric Jacobian is not transposed" begin
            asym!(du, u, p, t) = (du[1] = -u[1] + 3.0 * u[2]; du[2] = -2.0 * u[2]; nothing)
            function asym_jac!(J, u, p, t)
                J[1, 1] = -1.0
                J[1, 2] = 3.0
                J[2, 1] = 0.0
                return J[2, 2] = -2.0
            end
            prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(asym!; jac = asym_jac!), [1.0, 1.0], (0.0, 1.0),
            )
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSImplicit("bdf");
                dt = 0.002, reltol = 1.0e-11, abstol = 1.0e-13,
            )
            exact1 = 3 * (exp(-1) - exp(-2)) + exp(-1)
            @test isapprox(sol.u[end][1], exact1; atol = 1.0e-6)
            @test isapprox(sol.u[end][2], exp(-2); atol = 1.0e-6)
        end

        @testset "system Jacobian" begin
            p = (1.5, 1.0, 3.0, 1.0)
            jac_prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(lotka_volterra!; jac = lotka_volterra_jac!),
                [1.0, 1.0], (0.0, 5.0), p,
            )
            no_jac_prob = SciMLBase.ODEProblem(lotka_volterra!, [1.0, 1.0], (0.0, 5.0), p)
            alg = PETScDiffEq.TSImplicit("bdf")
            sol_jac = SciMLBase.solve(
                jac_prob, alg; dt = 0.01, reltol = 1.0e-8, abstol = 1.0e-10,
            )
            sol_fd = SciMLBase.solve(
                no_jac_prob, alg; dt = 0.01, reltol = 1.0e-8, abstol = 1.0e-10,
            )
            @test sol_jac.retcode == SciMLBase.ReturnCode.Success
            @test maximum(abs.(sol_jac.u[end] .- sol_fd.u[end])) < 1.0e-6
        end

        @testset "sparse jac_prototype matches the dense path" begin
            sparse_prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    chain!; jac = chain_jac!, jac_prototype = CHAIN_PROTOTYPE,
                ),
                [1.0, 0.0, 0.0], (0.0, 2.0),
            )
            dense_prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(chain!; jac = chain_jac!),
                [1.0, 0.0, 0.0], (0.0, 2.0),
            )
            alg = PETScDiffEq.TSImplicit("bdf")
            sol_sparse = SciMLBase.solve(
                sparse_prob, alg; dt = 0.01, reltol = 1.0e-9, abstol = 1.0e-11,
            )
            sol_dense = SciMLBase.solve(
                dense_prob, alg; dt = 0.01, reltol = 1.0e-9, abstol = 1.0e-11,
            )
            @test sol_sparse.retcode == SciMLBase.ReturnCode.Success
            @test maximum(abs.(sol_sparse.u[end] .- sol_dense.u[end])) < 1.0e-10
        end

        @testset "sparse jac_prototype convergence order" begin
            exact = exp(-1.0)
            proto = sparse([1], [1], [1.0], 1, 1)
            prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(decay!; jac = decay_jac!, jac_prototype = proto),
                [1.0], (0.0, 1.0),
            )
            errs = Float64[]
            for dt in (0.1, 0.05, 0.025, 0.0125)
                sol = SciMLBase.solve(
                    prob, PETScDiffEq.TSImplicit("cn", ["-ts_adapt_type", "none"]);
                    dt = dt,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                push!(errs, abs(sol.u[end][1] - exact))
            end
            orders = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
            @test all(o -> isapprox(o, 2; atol = 0.15), orders)
        end

        @testset "sparse prototype handles a structural zero on the diagonal" begin
            proto_prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    damped_oscillator!; jac = damped_oscillator_jac!,
                    jac_prototype = OSCILLATOR_PROTOTYPE,
                ),
                [1.0, 0.0], (0.0, 3.0),
            )
            fd_prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(damped_oscillator!; jac = damped_oscillator_jac!),
                [1.0, 0.0], (0.0, 3.0),
            )
            alg = PETScDiffEq.TSImplicit("bdf")
            sol_sparse = SciMLBase.solve(
                proto_prob, alg; dt = 0.01, reltol = 1.0e-9, abstol = 1.0e-11,
            )
            sol_fd = SciMLBase.solve(
                fd_prob, alg; dt = 0.01, reltol = 1.0e-9, abstol = 1.0e-11,
            )
            @test sol_sparse.retcode == SciMLBase.ReturnCode.Success
            @test maximum(abs.(sol_sparse.u[end] .- sol_fd.u[end])) < 1.0e-10
        end

        @testset "a wrong sparse Jacobian breaks Newton convergence" begin
            proto = sparse([1], [1], [1.0], 1, 1)
            wrong_prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    decay!; jac = decay_wrong_jac!, jac_prototype = proto,
                ),
                [1.0], (0.0, 1.0),
            )
            @test SciMLBase.solve(
                wrong_prob, PETScDiffEq.TSImplicit("cn", ["-ts_max_snes_failures", "1"]); dt = 0.05,
            ).retcode != SciMLBase.ReturnCode.Success
        end
    end

    @testset "a jac with a DM" begin
        PETSc = PETScDiffEq.PETSc
        LibPETSc = PETScDiffEq.LibPETSc
        pl = PETSc.getlib(; PetscScalar = Float64)
        PETScDiffEq.PETScCompat.isinitialized(pl) || PETSc.initialize(pl)
        ghosted = LibPETSc.DM_BOUNDARY_GHOSTED
        # One rank numbers the grid naturally, dofs innermost.
        da2 = PETSc.DMDA(
            pl, MPI.COMM_SELF, (ghosted, ghosted), (5, 4), 2, 1, LibPETSc.DMDA_STENCIL_STAR,
        )
        J = LibPETSc.DMCreateMatrix(pl, da2)
        at(c, i, j) = c + 2 * ((i - 1) + 5 * (j - 1))
        set_stencil_values!(J, (2, 3, 2), [(1, 3, 2), (2, 2, 2), (2, 3, 3)], [10.0, 20.0, 30.0])
        set_stencil_values!(J, [(1, 1, 1), (2, 1, 1)], (1, 1, 1), [1.0, 2.0])
        set_stencil_values!(J, [(1, 4, 3), (2, 4, 3)], [(1, 4, 3), (2, 4, 3)], [1.0 2.0; 3.0 4.0])
        set_stencil_values!(J, CartesianIndex(1, 5, 4), CartesianIndex(1, 5, 4), 7.0)
        set_stencil_values!(J, (2, 2, 2), ((1, 2, 2), (2, 2, 3)), (40.0, 50.0))
        set_stencil_values!(J, (1, 5, 4), (1, 6, 4), 5.0)
        PETSc.assemble!(J)
        set_stencil_values!(J, (1, 5, 4), (1, 5, 4), 1.0; add = true)
        PETSc.assemble!(J)
        function entry(r, c)
            v = Ref(0.0)
            PETScDiffEq._check_code(
                ccall(
                    PETScDiffEq._symbol(pl, :MatGetValues), PETScDiffEq.LibPETSc.PetscErrorCode,
                    (
                        Ptr{Cvoid}, PETScDiffEq.LibPETSc.PetscInt,
                        Ptr{PETScDiffEq.LibPETSc.PetscInt}, PETScDiffEq.LibPETSc.PetscInt,
                        Ptr{PETScDiffEq.LibPETSc.PetscInt}, Ptr{Float64},
                    ),
                    J.ptr, 1, [Int64(r - 1)], 1, [Int64(c - 1)], v,
                ),
            )
            return v[]
        end
        @test entry(at(2, 3, 2), at(1, 3, 2)) == 10.0
        @test entry(at(2, 3, 2), at(2, 2, 2)) == 20.0
        @test entry(at(2, 3, 2), at(2, 3, 3)) == 30.0
        @test entry(at(1, 1, 1), at(1, 1, 1)) == 1.0
        @test entry(at(2, 1, 1), at(1, 1, 1)) == 2.0
        @test [entry(at(c, 4, 3), at(d, 4, 3)) for c in 1:2, d in 1:2] == [1.0 2.0; 3.0 4.0]
        @test entry(at(2, 2, 2), at(1, 2, 2)) == 40.0
        @test entry(at(2, 2, 2), at(2, 2, 3)) == 50.0
        @test entry(at(1, 5, 4), at(1, 5, 4)) == 8.0
        @test_throws "a grid index is" set_stencil_values!(J, (1,), (1,), 1.0)
        @test_throws "a grid index is" set_stencil_values!(J, 3, 3, 1.0)
        @test_throws DimensionMismatch set_stencil_values!(
            J, (1, 1, 1), [(1, 1, 1), (2, 1, 1)], [1.0 2.0; 3.0 4.0],
        )
        @test_throws DimensionMismatch set_stencil_values!(
            J, [(1, 1, 1), (2, 1, 1)], [(1, 1, 1), (2, 1, 1)], [1.0, 2.0, 3.0, 4.0],
        )
        @test_throws DimensionMismatch set_stencil_values!(J, (1, 1, 1), [(1, 1, 1), (2, 1, 1)], 1.0)
        PETScDiffEq.PETScCompat.destroy!(J)
        PETScDiffEq.PETScCompat.destroy!(da2)

        N = 15
        dx = 1 / (N + 1)
        da = PETSc.DMDA(pl, MPI.COMM_SELF, (ghosted,), (N,), 1, 1)
        function heat_dm!(du, u, da, t)
            U = PETScDiffEq.reshape_local_array(u, da)
            D = PETScDiffEq.reshape_local_array(du, da)
            for i in axes(D, 2)
                D[1, i] = (U[1, i - 1] - 2U[1, i] + U[1, i + 1]) / dx^2
            end
            return nothing
        end
        seen = Int[]
        function heat_jac_dm!(J, u, da, t)
            push!(seen, length(u))
            for i in 1:N
                set_stencil_values!(J, (1, i), ((1, i - 1), (1, i), (1, i + 1)), (1, -2, 1) ./ dx^2)
            end
            return nothing
        end
        function heat!(du, u, p, t)
            for i in 1:N
                du[i] = ((i == 1 ? 0.0 : u[i - 1]) - 2u[i] + (i == N ? 0.0 : u[i + 1])) / dx^2
            end
            return nothing
        end
        function heat_jac!(J, u, p, t)
            for i in 1:N
                J[i, i] = -2 / dx^2
                i > 1 && (J[i, i - 1] = 1 / dx^2)
                i < N && (J[i, i + 1] = 1 / dx^2)
            end
            return nothing
        end
        near(i) = max(1, i - 1):min(N, i + 1)
        proto = sparse(
            [i for i in 1:N for _ in near(i)], [j for i in 1:N for j in near(i)], ones(3N - 2), N, N,
        )
        u0 = sinpi.((1:N) .* dx) .+ 0.5 .* sinpi.(3 .* (1:N) .* dx)
        tol = (abstol = 1.0e-8, reltol = 1.0e-8)
        dm_prob(span; kw...) = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(heat_dm!; jac = heat_jac_dm!, kw...), u0, span, da,
        )
        ref_prob(span) = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(heat!; jac = heat_jac!, jac_prototype = proto), u0, span,
        )
        for (with_dm, alg) in (
                (PETScDiffEq.TSImplicit("bdf"; dm = da), PETScDiffEq.TSImplicit("bdf")),
                (PETScDiffEq.TSRosW(; dm = da), PETScDiffEq.TSRosW()),
            )
            empty!(seen)
            got = SciMLBase.solve(dm_prob((0.0, 0.1)), with_dm; tol...)
            ref = SciMLBase.solve(ref_prob((0.0, 0.1)), alg; tol...)
            @test got.retcode == SciMLBase.ReturnCode.Success
            @test got.t == ref.t
            @test got.u == ref.u
            @test got.stats.njacs == ref.stats.njacs == length(seen) > 0
            @test all(==(N + 2), seen)
            coloured = SciMLBase.solve(
                SciMLBase.ODEProblem(heat_dm!, u0, (0.0, 0.1), da), with_dm; tol...,
            )
            @test coloured.stats.njacs == 0
            @test got.stats.nf < coloured.stats.nf
            # Measured 1.2e-14 for bdf and 4.4e-16 for rosw.
            @test maximum(abs, got.u[end] - coloured.u[end]) <= 1.0e-13
        end
        back = SciMLBase.solve(dm_prob((0.1, 0.0)), PETScDiffEq.TSImplicit("bdf"; dm = da); tol...)
        ref = SciMLBase.solve(ref_prob((0.1, 0.0)), PETScDiffEq.TSImplicit("bdf"); tol...)
        @test back.retcode == SciMLBase.ReturnCode.Success
        @test back.t == ref.t
        @test back.u == ref.u
        plain = SciMLBase.solve(dm_prob((0.0, 0.1)), PETScDiffEq.TSImplicit("bdf"; dm = da); tol...)
        with_ad = SciMLBase.solve(
            dm_prob((0.0, 0.1)),
            PETScDiffEq.TSImplicit("bdf"; dm = da, autodiff = PETScDiffEq.AutoForwardDiff());
            tol...,
        )
        @test with_ad.t == plain.t
        @test with_ad.u == plain.u
        empty!(seen)
        @test SciMLBase.solve(dm_prob((0.0, 0.1)), PETScDiffEq.TSRK(; dm = da); dt = 1.0e-3).u ==
            SciMLBase.solve(
            SciMLBase.ODEProblem(heat_dm!, u0, (0.0, 0.1), da), PETScDiffEq.TSRK(; dm = da);
            dt = 1.0e-3,
        ).u
        @test isempty(seen)
        @test_throws "has to be in place" SciMLBase.solve(
            SciMLBase.ODEProblem(
                SciMLBase.ODEFunction{false}((u, p, t) -> -u; jac = (u, p, t) -> nothing), u0,
                (0.0, 0.1), da,
            ),
            PETScDiffEq.TSImplicit("bdf"; dm = da),
        )
        @test_throws "leave out `jac_prototype`" SciMLBase.solve(
            dm_prob((0.0, 0.1); jac_prototype = proto), PETScDiffEq.TSImplicit("bdf"; dm = da),
        )
        algebraic(i) = i % 4 == 0
        calls = Ref(0)
        function chain_dm!(du, u, da, t)
            calls[] += 1
            U = PETScDiffEq.reshape_local_array(u, da)
            D = PETScDiffEq.reshape_local_array(du, da)
            for i in axes(D, 2)
                l, c, r = U[1, i - 1], U[1, i], U[1, i + 1]
                D[1, i] = algebraic(i) ? c^3 + c - (l + r) / 2 - 0.1 : l - 2c + r
            end
            return nothing
        end
        function chain_jac_dm!(J, u, da, t)
            push!(seen, length(u))
            U = PETScDiffEq.reshape_local_array(u, da)
            for i in 1:N
                c = U[1, i]
                vals = algebraic(i) ? (-0.5, 3c^2 + 1, -0.5) : (1.0, -2.0, 1.0)
                set_stencil_values!(J, (1, i), ((1, i - 1), (1, i), (1, i + 1)), vals)
            end
            return nothing
        end
        chain(x0 = u0, grid = da; kw...) = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(
                chain_dm!; kw...,
                mass_matrix = Diagonal([algebraic(i) ? 0.0 : 1.0 for i in eachindex(x0)]),
            ), x0, (0.0, 0.1), grid,
        )
        bdf = PETScDiffEq.TSImplicit("bdf"; dm = da)
        wide = PETSc.DMDA(pl, MPI.COMM_SELF, (ghosted,), (400,), 1, 1)
        for init in (DiffEqBase.BrownFullBasicInit(), DiffEqBase.ShampineCollocationInit())
            empty!(seen)
            with_jac = SciMLBase.init(
                chain(; jac = chain_jac_dm!), bdf; initializealg = init, tol...,
            )
            @test !isempty(seen)
            plain = SciMLBase.init(chain(), bdf; initializealg = init, tol...)
            @test with_jac.u != u0
            @test maximum(abs, with_jac.u - plain.u) <= 1.0e-13
            for integ in (with_jac, plain)
                SciMLBase.set_u!(integ, integ.u .+ 0.05)
                SciMLBase.initialize_dae!(integ)
            end
            @test maximum(abs, with_jac.u - plain.u) <= 1.0e-13
            SciMLBase.terminate!(with_jac)
            SciMLBase.terminate!(plain)
            calls[] = 0
            coloured = SciMLBase.init(
                chain(fill(0.5, 400), wide), PETScDiffEq.TSImplicit("bdf"; dm = wide);
                initializealg = init, tol...,
            )
            @test calls[] < 200
            SciMLBase.terminate!(coloured)
        end
        PETScDiffEq.PETScCompat.destroy!(wide)
        PETScDiffEq.PETScCompat.destroy!(da)
    end

    Sys.WORD_SIZE == 64 && @testset "a DMStag as the dm" begin
        PETSc = PETScDiffEq.PETSc
        LibPETSc = PETScDiffEq.LibPETSc
        on_grid = PETScDiffEq.reshape_local_array
        pl = PETSc.getlib(; PetscScalar = Float64)
        PETScDiffEq.PETScCompat.isinitialized(pl) || PETSc.initialize(pl)
        LEFT, RIGHT, ELEM = LibPETSc.DMSTAG_LEFT, LibPETSc.DMSTAG_RIGHT, LibPETSc.DMSTAG_ELEMENT
        DOWN, UP = LibPETSc.DMSTAG_DOWN, LibPETSc.DMSTAG_UP
        open_edge, ghosted = LibPETSc.DM_BOUNDARY_NONE, LibPETSc.DM_BOUNDARY_GHOSTED

        # One rank orders a 2 x 2 grid of faces and cells element by element, each element's
        # bottom face, left face and cell, the column past the end holding only left faces
        # and the row past the end only bottom ones.
        plane = PETSc.DMStag(pl, MPI.COMM_SELF, (open_edge, open_edge), (2, 2), (0, 1, 1), 1)
        index = collect(1.0:16.0)
        a = on_grid(index, plane)
        @test [a[loc, 1, 1, 1] for loc in (DOWN, LEFT, ELEM)] == [1, 2, 3]
        @test a[RIGHT, 1, 1, 1] == a[LEFT, 1, 2, 1] == 5
        @test a[LEFT, 1, 3, 1] == 7
        @test a[ELEM, 1, CartesianIndex(2, 2)] == 13
        @test a[LEFT, 1, 3, 2] == 14
        @test a[UP, 1, 1, 2] == a[DOWN, 1, 1, 3] == 15
        @test a[DOWN, 1, 2, 3] == 16
        @test axes(a) == (1:3, 1:3)
        @test axes(on_grid(zeros(PETScDiffEq._dm_local_size(pl, plane)), plane), 2) == 1:3
        @test_throws "only the lower points" a[ELEM, 1, 3, 1]
        @test_throws "has 1 components" a[ELEM, 2, 1, 1]
        @test_throws "takes 2 element indices" a[ELEM, 1, 1]
        @test_throws "not a location on a 2-D DMStag" a[LibPETSc.DMSTAG_BACK, 1, 1, 1]
        @test_throws "elements 1:3 x 1:3" a[ELEM, 1, 4, 1]
        @test_throws DimensionMismatch on_grid(zeros(15), plane)
        z = zeros(16)
        b = on_grid(z, plane)
        b[ELEM, 1, 1, 2] = 2.5
        b[UP, 1, CartesianIndex(2, 1)] = 3.5
        @test findall(!iszero, z) == [10, 11]
        @test z[10:11] == [2.5, 3.5]

        J = LibPETSc.DMCreateMatrix(pl, plane)
        set_stencil_values!(
            J, (ELEM, 1, 2, 1), [(LEFT, 1, 2, 1), (RIGHT, 1, 2, 1), (UP, 1, 2, 1)], [1.0, 2.0, 3.0],
        )
        set_stencil_values!(J, [(LEFT, 1, 3, 2), (DOWN, 1, 2, 3)], (LEFT, 1, 3, 2), (4.0, 5.0))
        PETSc.assemble!(J)
        function entry(r, c)
            v = Ref(0.0)
            PETScDiffEq._check_code(
                ccall(
                    PETScDiffEq._symbol(pl, :MatGetValues), PETScDiffEq.LibPETSc.PetscErrorCode,
                    (
                        Ptr{Cvoid}, PETScDiffEq.LibPETSc.PetscInt,
                        Ptr{PETScDiffEq.LibPETSc.PetscInt}, PETScDiffEq.LibPETSc.PetscInt,
                        Ptr{PETScDiffEq.LibPETSc.PetscInt}, Ptr{Float64},
                    ),
                    J.ptr, 1, [Int64(r - 1)], 1, [Int64(c - 1)], v,
                ),
            )
            return v[]
        end
        @test [entry(6, c) for c in (5, 7, 11)] == [1.0, 2.0, 3.0]
        @test [entry(r, 14) for r in (14, 16)] == [4.0, 5.0]
        @test_throws "takes points `(loc, c, i)`" set_stencil_values!(J, (1, 1, 1), (1, 1, 1), 1.0)
        @test_throws "ghosted region" set_stencil_values!(J, (ELEM, 1, 1, 1), (ELEM, 1, 5, 1), 1.0)
        PETScDiffEq.PETScCompat.destroy!(J)
        da = PETSc.DMDA(pl, MPI.COMM_SELF, (ghosted,), (4,), 1, 1)
        J = LibPETSc.DMCreateMatrix(pl, da)
        @test_throws "is a point of a DMStag" set_stencil_values!(
            J, (ELEM, 1, 1), (ELEM, 1, 1), 1.0,
        )
        PETScDiffEq.PETScCompat.destroy!(J)
        PETScDiffEq.PETScCompat.destroy!(da)

        # A damped wave: fluxes on the vertices, held at zero on the ends, and pressures in
        # the cells. One rank keeps the natural order, vertex then cell.
        N = 12
        h = 1 / N
        line = PETSc.DMStag(pl, MPI.COMM_SELF, (ghosted,), (N,), (1, 1), 1)
        function wave_dm!(du, u, dm, t)
            U, D = on_grid(u, dm), on_grid(du, dm)
            for i in axes(D, 1)
                D[LEFT, 1, i] = i == 1 || i == N + 1 ? 0.0 :
                    -(U[ELEM, 1, i] - U[ELEM, 1, i - 1]) / h - U[LEFT, 1, i]
                i <= N && (D[ELEM, 1, i] = -(U[RIGHT, 1, i] - U[LEFT, 1, i]) / h - U[ELEM, 1, i]^3)
            end
            return nothing
        end
        function wave!(dx, x, p, t)
            for v in 1:(N + 1)
                dx[2v - 1] = v == 1 || v == N + 1 ? 0.0 : -(x[2v] - x[2v - 2]) / h - x[2v - 1]
            end
            for i in 1:N
                dx[2i] = -(x[2i + 1] - x[2i - 1]) / h - x[2i]^3
            end
            return nothing
        end
        njac = Ref(0)
        function wave_jac_dm!(J, u, dm, t)
            njac[] += 1
            U = on_grid(u, dm)
            for i in 1:N
                1 < i && set_stencil_values!(
                    J, (LEFT, 1, i), ((ELEM, 1, i - 1), (ELEM, 1, i), (LEFT, 1, i)),
                    (1 / h, -1 / h, -1.0),
                )
                set_stencil_values!(
                    J, (ELEM, 1, i), ((LEFT, 1, i), (RIGHT, 1, i), (ELEM, 1, i)),
                    (1 / h, -1 / h, -3U[ELEM, 1, i]^2),
                )
            end
            return nothing
        end
        function wave_jac!(J, x, p, t)
            for v in 2:N
                J[2v - 1, 2v - 2], J[2v - 1, 2v], J[2v - 1, 2v - 1] = 1 / h, -1 / h, -1.0
            end
            for i in 1:N
                J[2i, 2i - 1], J[2i, 2i + 1], J[2i, 2i] = 1 / h, -1 / h, -3x[2i]^2
            end
            return nothing
        end
        n = 2N + 1
        near(r) = max(1, r - 2):min(n, r + 2)
        proto = sparse(
            [r for r in 1:n for _ in near(r)], [c for r in 1:n for c in near(r)],
            ones(sum(length ∘ near, 1:n)), n, n,
        )
        x0 = [isodd(k) ? 0.0 : sinpi((k ÷ 2 - 0.5) * h) for k in 1:n]
        @test on_grid(x0, line)[ELEM, 1, 3] == x0[6]
        span = (0.0, 0.3)
        fixed = (dt = 1.0e-3, adaptive = false)
        tol = (abstol = 1.0e-9, reltol = 1.0e-9)
        direct = ["-ksp_type", "preonly", "-pc_type", "lu"]
        gap(a, b) = maximum(maximum(abs, x - y) for (x, y) in zip(a, b))
        halve = SciMLBase.DiscreteCallback((u, t, i) -> t == 0.1, i -> (i.u .*= 0.5))
        got = SciMLBase.solve(
            SciMLBase.ODEProblem(wave_dm!, x0, span, line), PETScDiffEq.TSRK("5dp"; dm = line);
            saveat = 0.05, callback = halve, tstops = [0.1], fixed...,
        )
        ref = SciMLBase.solve(
            SciMLBase.ODEProblem(wave!, x0, span), PETScDiffEq.TSRK("5dp");
            saveat = 0.05, callback = halve, tstops = [0.1], fixed...,
        )
        @test got.retcode == SciMLBase.ReturnCode.Success
        @test got.t == ref.t
        @test got.u == ref.u

        with_jac(f, jac; kw...) = SciMLBase.ODEFunction(f; jac, kw...)
        for make in (
                (; kw...) -> PETScDiffEq.TSImplicit("bdf", direct; kw...),
                (; kw...) -> PETScDiffEq.TSRosW("ra34pw2", direct; kw...),
            )
            njac[] = 0
            got = SciMLBase.solve(
                SciMLBase.ODEProblem(with_jac(wave_dm!, wave_jac_dm!), x0, span, line),
                make(; dm = line); saveat = 0.05, tol...,
            )
            ref = SciMLBase.solve(
                SciMLBase.ODEProblem(with_jac(wave!, wave_jac!; jac_prototype = proto), x0, span),
                make(); saveat = 0.05, tol...,
            )
            coloured = SciMLBase.solve(
                SciMLBase.ODEProblem(wave_dm!, x0, span, line), make(; dm = line);
                saveat = 0.05, tol...,
            )
            @test got.retcode == coloured.retcode == SciMLBase.ReturnCode.Success
            @test got.stats.njacs == njac[] > 0
            @test coloured.stats.njacs == 0
            @test got.stats.nf < coloured.stats.nf
            # Measured 5.1e-26 for bdf and 1.4e-15 for rosw against the serial solve, and 1.0e-14
            # and 1.1e-15 against colouring.
            @test gap(got.u, ref.u) <= 1.0e-13
            @test gap(coloured.u, got.u) <= 1.0e-13
        end

        weights = Diagonal(1 .+ (1:n) ./ n)
        bdf(; kw...) = PETScDiffEq.TSImplicit("bdf", direct; kw...)
        got = SciMLBase.solve(
            SciMLBase.ODEProblem(
                with_jac(wave_dm!, wave_jac_dm!; mass_matrix = weights), x0, span, line,
            ),
            bdf(; dm = line); saveat = 0.05, tol...,
        )
        ref = SciMLBase.solve(
            SciMLBase.ODEProblem(
                with_jac(wave!, wave_jac!; jac_prototype = proto, mass_matrix = weights), x0, span,
            ),
            bdf(); saveat = 0.05, tol...,
        )
        @test got.retcode == SciMLBase.ReturnCode.Success
        # Measured 9.4e-15.
        @test gap(got.u, ref.u) <= 1.0e-13

        function walk(prob, alg)
            integ = SciMLBase.init(prob, alg; fixed...)
            seen = map(1:10) do _
                SciMLBase.step!(integ)
                (copy(integ.u), integ((integ.tprev + integ.t) / 2), SciMLBase.get_du(integ))
            end
            SciMLBase.terminate!(integ)
            return reduce(vcat, (vcat(s...) for s in seen))
        end
        mine = walk(
            SciMLBase.ODEProblem(with_jac(wave_dm!, wave_jac_dm!), x0, span, line),
            bdf(; dm = line),
        )
        theirs = walk(
            SciMLBase.ODEProblem(with_jac(wave!, wave_jac!; jac_prototype = proto), x0, span),
            bdf(),
        )
        # Measured 7.1e-24.
        @test maximum(abs, mine - theirs) <= 1.0e-15

        made = Ref{Ptr{Cvoid}}()
        PETScDiffEq._check_code(
            ccall(
                PETScDiffEq._symbol(pl, :DMShellCreate), PETScDiffEq.LibPETSc.PetscErrorCode,
                (MPI.API.MPI_Comm, Ptr{Ptr{Cvoid}}), MPI.COMM_SELF, made,
            ),
        )
        shell = LibPETSc.PetscDM(made[], pl)
        @test_throws "not a DM of type `shell`" SciMLBase.solve(
            SciMLBase.ODEProblem(wave_dm!, x0, span, shell), PETScDiffEq.TSRK(; dm = shell);
            fixed...,
        )
        PETScDiffEq.PETScCompat.destroy!(shell)
        PETScDiffEq.PETScCompat.destroy!(line)
        PETScDiffEq.PETScCompat.destroy!(plane)
    end

    Sys.WORD_SIZE == 64 && @testset "a DMPlex as the dm" begin
        PETSc = PETScDiffEq.PETSc
        LibPETSc = PETScDiffEq.LibPETSc
        on_mesh = PETScDiffEq.reshape_local_array
        pl = PETSc.getlib(; PetscScalar = Float64)
        PETScDiffEq.PETScCompat.isinitialized(pl) || PETSc.initialize(pl)
        sym(name) = PETScDiffEq._symbol(pl, name)
        chk(code) = PETScDiffEq._check_code(code)
        # PETSc's 64-bit builds take Int64 indices.
        function stratum(dm, depth)
            lo, hi = Ref(0), Ref(0)
            chk(
                ccall(
                    sym(:DMPlexGetDepthStratum), Cint, (Ptr{Cvoid}, Int64, Ptr{Int64}, Ptr{Int64}),
                    dm.ptr, depth, lo, hi,
                ),
            )
            return lo[]:(hi[] - 1)
        end
        function adjacent(size_name, name, dm, p)
            n, q = Ref(0), Ref{Ptr{Int64}}()
            chk(ccall(sym(size_name), Cint, (Ptr{Cvoid}, Int64, Ptr{Int64}), dm.ptr, p, n))
            chk(ccall(sym(name), Cint, (Ptr{Cvoid}, Int64, Ptr{Ptr{Int64}}), dm.ptr, p, q))
            return copy(unsafe_wrap(Array, q[], n[]))
        end
        cone(dm, p) = adjacent(:DMPlexGetConeSize, :DMPlexGetCone, dm, p)
        support(dm, p) = adjacent(:DMPlexGetSupportSize, :DMPlexGetSupport, dm, p)
        mesh(faces, simplex) = PETSc.DMPlex(
            pl, MPI.COMM_SELF; dm_plex_dim = 2, dm_plex_simplex = simplex ? "1" : "0",
            dm_plex_box_faces = join(faces, ","),
        )
        setdof(name, s, p, k) =
            chk(ccall(sym(name), Cint, (Ptr{Cvoid}, Int64, Int64), s, p, k))
        # dofs[d + 1] degrees of freedom on each point of depth d.
        function with_dofs!(dm, dofs; setup = true, pinned = nothing)
            s, lo, hi = Ref{Ptr{Cvoid}}(), Ref(0), Ref(0)
            chk(
                ccall(
                    sym(:PetscSectionCreate), Cint, (MPI.API.MPI_Comm, Ptr{Ptr{Cvoid}}),
                    MPI.COMM_SELF, s,
                ),
            )
            chk(
                ccall(
                    sym(:DMPlexGetChart), Cint, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64}), dm.ptr, lo, hi,
                ),
            )
            setdof(:PetscSectionSetChart, s[], lo[], hi[])
            for (d, k) in enumerate(dofs), p in stratum(dm, d - 1)
                setdof(:PetscSectionSetDof, s[], p, k)
            end
            pinned === nothing || setdof(:PetscSectionSetConstraintDof, s[], pinned, 1)
            setup && chk(ccall(sym(:PetscSectionSetUp), Cint, (Ptr{Cvoid},), s[]))
            chk(ccall(sym(:DMSetLocalSection), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), dm.ptr, s[]))
            chk(ccall(sym(:PetscSectionDestroy), Cint, (Ptr{Ptr{Cvoid}},), s))
            return dm
        end
        function entry(J, r, c)
            v = Ref(0.0)
            chk(
                ccall(
                    sym(:MatGetValues), Cint,
                    (Ptr{Cvoid}, Int64, Ptr{Int64}, Int64, Ptr{Int64}, Ptr{Float64}),
                    J.ptr, 1, [r - 1], 1, [c - 1], v,
                ),
            )
            return v[]
        end

        # On a 2 x 2 grid of squares PETSc numbers the 4 cells, then the 9 vertices, then the
        # 12 edges, and the local section lays out the points in that order.
        squares = with_dofs!(mesh((2, 2), false), (1, 0, 2))
        @test stratum(squares, 2) == 0:3
        @test stratum(squares, 0) == 4:12
        a = on_mesh(collect(1.0:17.0), squares)
        @test (a[1, 0], a[2, 0], a[2, 3]) == (1, 2, 8)
        @test (a[1, 4], a[1, 12]) == (9, 17)
        @test checkbounds(Bool, a, 2, 3)
        @test !checkbounds(Bool, a, 3, 3)
        @test !checkbounds(Bool, a, 1, 13)
        @test !checkbounds(Bool, a, 1, 25)
        @test_throws "has 2 degrees of freedom, counted from 1, so no component 3" a[3, 0]
        @test_throws "has 0 degrees of freedom" a[1, 13]
        @test_throws "outside this rank's points 0:24" a[1, 25]
        @test_throws DimensionMismatch on_mesh(zeros(16), squares)
        z = zeros(17)
        b = on_mesh(z, squares)
        b[2, 1] = 2.5
        b[1, 5] = 3.5
        @test findall(!iszero, z) == [4, 10]
        @test z[[4, 10]] == [2.5, 3.5]
        @test PETScDiffEq._dm_local_size(pl, squares) == 17

        corner = first(cone(squares, first(cone(squares, 0))))
        J = LibPETSc.DMCreateMatrix(pl, squares)
        set_stencil_values!(J, (2, 0), [(1, 0), (1, corner)], [1.0, 2.0])
        set_stencil_values!(J, [(1, corner), (2, 0)], (2, 0), (3.0, 4.0); add = false)
        PETSc.assemble!(J)
        @test entry(J, 2, 1) == 1.0
        @test entry(J, 2, 9 + corner - 4) == 2.0
        @test entry(J, 9 + corner - 4, 2) == 3.0
        @test entry(J, 2, 2) == 4.0
        @test_throws "takes points `(c, p)`" set_stencil_values!(J, (1, 0, 0), (1, 0), 1.0)
        @test_throws "so no component 3" set_stencil_values!(J, (3, 0), (1, 0), 1.0)
        @test_throws "outside this rank's points" set_stencil_values!(J, (1, 0), (1, 99), 1.0)
        PETScDiffEq.PETScCompat.destroy!(J)

        span = (0.0, 0.1)
        fixed = (dt = 1.0e-3, adaptive = false)
        tol = (abstol = 1.0e-9, reltol = 1.0e-9)
        direct = ["-ksp_type", "preonly", "-pc_type", "lu"]
        gap(x, y) = maximum(maximum(abs, p - q) for (p, q) in zip(x, y))
        decay_dm!(du, u, p, t) = (du .= -u; nothing)
        bare = mesh((2, 2), false)
        @test_throws "has no degrees of freedom" SciMLBase.solve(
            SciMLBase.ODEProblem(decay_dm!, zeros(9), span, bare), TSRK(; dm = bare); fixed...,
        )
        loose = with_dofs!(mesh((2, 2), false), (1, 0, 0); setup = false)
        @test_throws "is not set up; call PetscSectionSetUp" SciMLBase.solve(
            SciMLBase.ODEProblem(decay_dm!, zeros(9), span, loose), TSRK(; dm = loose); fixed...,
        )
        pinned = with_dofs!(mesh((2, 2), false), (1, 0, 0); pinned = 4)
        @test_throws "no constrained degrees of freedom" SciMLBase.solve(
            SciMLBase.ODEProblem(decay_dm!, zeros(9), span, pinned), TSRK(; dm = pinned); fixed...,
        )
        @test_throws "no constrained degrees of freedom" on_mesh(zeros(9), pinned)
        foreach(PETScDiffEq.PETScCompat.destroy!, (bare, loose, pinned, squares))

        # Reaction-diffusion coupling each point to its neighbours, on the DMPlex and on the
        # same connectivity written without a DM, in the order the mesh numbers the points.
        κ = 4.0
        function on_plex(points, near)
            function f!(du, u, dm, t)
                U, D = on_mesh(u, dm), on_mesh(du, dm)
                for p in points
                    s = 0.0
                    for x in near[p]
                        s += U[1, x] - U[1, p]
                    end
                    D[1, p] = κ * s - U[1, p]^3
                end
                return nothing
            end
            function jac!(J, u, dm, t)
                U = on_mesh(u, dm)
                for p in points
                    k = length(near[p])
                    set_stencil_values!(
                        J, (1, p), [(1, p); [(1, x) for x in near[p]]],
                        [-κ * k - 3U[1, p]^2; fill(κ, k)],
                    )
                end
                return nothing
            end
            return f!, jac!
        end
        function by_hand(points, near)
            at = Dict(p => k for (k, p) in enumerate(points))
            nk = [[at[x] for x in near[p]] for p in points]
            function f!(du, u, p, t)
                for k in eachindex(u)
                    s = 0.0
                    for j in nk[k]
                        s += u[j] - u[k]
                    end
                    du[k] = κ * s - u[k]^3
                end
                return nothing
            end
            function jac!(J, u, p, t)
                for k in eachindex(u)
                    J[k, k] = -κ * length(nk[k]) - 3u[k]^2
                    for j in nk[k]
                        J[k, j] = κ
                    end
                end
                return nothing
            end
            n = length(points)
            ks = [[k; nk[k]] for k in 1:n]
            return f!, jac!, sparse([k for k in 1:n for _ in ks[k]], reduce(vcat, ks), 1.0, n, n)
        end
        function walk(integ)
            seen = map(1:10) do _
                SciMLBase.step!(integ)
                vcat(copy(integ.u), integ((integ.tprev + integ.t) / 2), SciMLBase.get_du(integ))
            end
            SciMLBase.terminate!(integ)
            return reduce(vcat, seen)
        end

        # The vertices of a triangulated box, coupled along its edges, and the cells of a box
        # of squares, coupled across them once the adjacency says so.
        tri = with_dofs!(mesh((4, 3), true), (1, 0, 0))
        verts = stratum(tri, 0)
        along = Dict(
            v => [only(filter(!=(v), cone(tri, e))) for e in support(tri, v)] for v in verts
        )
        quads = with_dofs!(mesh((4, 3), false), (0, 0, 1))
        chk(ccall(sym(:DMSetBasicAdjacency), Cint, (Ptr{Cvoid}, Cint, Cint), quads.ptr, 1, 0))
        cells = stratum(quads, 2)
        across = Dict(
            c => [x for e in cone(quads, c) for x in support(quads, e) if x != c] for c in cells
        )
        solvers = (
            (; kw...) -> TSImplicit("bdf", direct; kw...),
            (; kw...) -> TSRosW("ra34pw2", direct; kw...),
        )
        for (dm, points, near) in ((tri, verts, along), (quads, cells, across))
            f_dm!, jac_dm! = on_plex(points, near)
            f!, jac!, proto = by_hand(points, near)
            x0 = sinpi.((1:length(points)) ./ length(points)) .+ 0.5
            on_dm(; kw...) =
                SciMLBase.ODEProblem(SciMLBase.ODEFunction(f_dm!; kw...), x0, span, dm)
            plain(; kw...) = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(f!; jac_prototype = proto, kw...), x0, span,
            )
            got = SciMLBase.solve(on_dm(), TSRK("5dp"; dm); saveat = 0.02, fixed...)
            ref = SciMLBase.solve(plain(), TSRK("5dp"); saveat = 0.02, fixed...)
            @test got.retcode == SciMLBase.ReturnCode.Success
            @test got.t == ref.t
            @test got.u == ref.u
            for make in solvers
                got = SciMLBase.solve(on_dm(; jac = jac_dm!), make(; dm); saveat = 0.02, tol...)
                coloured = SciMLBase.solve(on_dm(), make(; dm); saveat = 0.02, tol...)
                ref = SciMLBase.solve(plain(; jac = jac!), make(); saveat = 0.02, tol...)
                @test got.retcode == coloured.retcode == SciMLBase.ReturnCode.Success
                @test got.stats.njacs > 0
                @test coloured.stats.njacs == 0
                @test got.stats.nf < coloured.stats.nf
                # Measured 0.0 against the serial solve for both and up to 1.5e-14 against
                # colouring.
                @test gap(got.u, ref.u) <= 1.0e-13
                @test gap(coloured.u, got.u) <= 1.0e-13
            end
            bdf(; kw...) = TSImplicit("bdf", direct; kw...)
            mine = walk(SciMLBase.init(on_dm(; jac = jac_dm!), bdf(; dm); fixed...))
            theirs = walk(SciMLBase.init(plain(; jac = jac!), bdf(); fixed...))
            # Measured 0.0.
            @test maximum(abs, mine - theirs) <= 1.0e-15
        end
        PETScDiffEq.PETScCompat.destroy!(tri)
        PETScDiffEq.PETScCompat.destroy!(quads)
    end

    @testset "Adaptive stepping" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        sol = SciMLBase.solve(
            prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, reltol = 1.0e-8, abstol = 1.0e-10,
        )
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-6
    end

    @testset "System of equations" begin
        p = (1.5, 1.0, 3.0, 1.0)
        prob = SciMLBase.ODEProblem(lotka_volterra!, [1.0, 1.0], (0.0, 5.0), p)
        for alg in (
                PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRosW("ra34pw2"),
                PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSARKIMEX("3"),
                PETScDiffEq.TSGeneric("alpha"),
            )
            sol = SciMLBase.solve(prob, alg; dt = 0.01, reltol = 1.0e-8, abstol = 1.0e-10)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test all(isfinite, sol.u[end])
            @test all(>(0), sol.u[end])
        end
    end

    @testset "Saving controls" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"])
        exact(t) = exp(-t)

        @testset "save_everystep" begin
            every = SciMLBase.solve(prob, alg; dt = 0.1)
            @test length(every.t) == 11
            ends = SciMLBase.solve(prob, alg; dt = 0.1, save_everystep = false)
            @test ends.t == [0.0, 1.0]
            @test abs(ends.u[end][1] - exact(1.0)) < 1.0e-7
        end

        @testset "saveat hits the requested times" begin
            scalar = SciMLBase.solve(prob, alg; dt = 0.1, saveat = 0.25)
            @test scalar.t ≈ [0.0, 0.25, 0.5, 0.75, 1.0]
            vec = SciMLBase.solve(prob, alg; dt = 0.1, saveat = [0.3, 0.7])
            @test vec.t ≈ [0.3, 0.7]
            ends = SciMLBase.solve(
                prob, alg; dt = 0.1, saveat = [0.3, 0.7], save_start = true, save_end = true,
            )
            @test ends.t ≈ [0.0, 0.3, 0.7, 1.0]
            for sol in (scalar, vec), i in eachindex(sol.t)
                @test abs(sol.u[i][1] - exact(sol.t[i])) < 1.0e-7
            end
        end

        @testset "the saving flags combine as in OrdinaryDiffEq" begin
            both = SciMLBase.solve(
                prob, alg; dt = 0.25, saveat = [0.3], save_everystep = true, tstops = [0.5],
            )
            @test both.t ≈ [0.0, 0.25, 0.3, 0.5, 0.75, 1.0]
            @test allunique(both.t)
            @test isempty(SciMLBase.solve(prob, alg; dt = 0.1, saveat = [5.0]).t)
            @test SciMLBase.solve(prob, alg; dt = 0.1, save_on = false).t == [0.0, 1.0]
            @test isempty(SciMLBase.solve(prob, alg; dt = 0.1, saveat = [0.5], save_on = false).t)
            integ = SciMLBase.init(prob, alg; dt = 0.25, adaptive = false)
            SciMLBase.step!(integ)
            SciMLBase.add_saveat!(integ, 0.6)
            @test SciMLBase.solve!(integ).t ≈ [0.0, 0.25, 0.5, 0.6, 0.75, 1.0]
        end

        @testset "save_on = false leaves callbacks and savevalues! nothing to save" begin
            doubled = SciMLBase.DiscreteCallback(
                (u, t, integ) -> false, integ -> nothing;
                initialize = (c, u, t, integ) -> (integ.u .*= 2; nothing),
            )
            for cb in (
                    SciMLBase.DiscreteCallback((u, t, integ) -> t > 0.4, integ -> nothing),
                    SciMLBase.ContinuousCallback((u, t, integ) -> t - 0.55, integ -> nothing),
                    PresetTimeCallback([0.3], integ -> nothing), doubled,
                )
                @test SciMLBase.solve(prob, alg; dt = 0.1, callback = cb, save_on = false).t ==
                    [0.0, 1.0]
            end
            integ = SciMLBase.init(prob, alg; dt = 0.1, save_on = false)
            SciMLBase.step!(integ)
            @test SciMLBase.savevalues!(integ, true) == (false, false)
            @test SciMLBase.solve!(integ).t == [0.0, 1.0]
        end

        @testset "integ.opts.save_on pauses saving part-way" begin
            pause = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t > 0.25, integ -> (integ.opts.save_on = false);
                save_positions = (false, false),
            )
            @test SciMLBase.solve(prob, alg; dt = 0.1, callback = pause).t ≈
                [0.0, 0.1, 0.2, 0.3, 1.0]
            integ = SciMLBase.init(prob, alg; dt = 0.1, saveat = [0.05, 0.15, 0.55, 0.65])
            SciMLBase.step!(integ)
            SciMLBase.step!(integ)
            integ.opts.save_on = false
            for _ in 1:4
                SciMLBase.step!(integ)
            end
            @test SciMLBase.savevalues!(integ, true) == (false, false)
            integ.opts.save_on = true
            @test SciMLBase.solve!(integ).t == [0.05, 0.15, 0.65]
        end

        @testset "save_start and save_end" begin
            nostart = SciMLBase.solve(
                prob, alg; dt = 0.1, saveat = [0.0, 0.3, 1.0], save_start = false,
            )
            @test nostart.t ≈ [0.3, 1.0]
            noend = SciMLBase.solve(
                prob, alg; dt = 0.1, saveat = [0.0, 0.3, 1.0], save_end = false,
            )
            @test noend.t ≈ [0.0, 0.3]
            @test SciMLBase.solve(prob, alg; dt = 0.25, save_end = false).t ≈
                [0.0, 0.25, 0.5, 0.75]
            everystep = SciMLBase.solve(prob, alg; dt = 0.25, save_start = false)
            @test everystep.t ≈ [0.25, 0.5, 0.75, 1.0]
            noend = SciMLBase.solve(
                prob, alg; dt = 0.1, save_everystep = false, save_end = false,
            )
            @test noend.t ≈ [0.0]
        end

        @testset "final state is the solution at tf, not a stale duplicate" begin
            sol = SciMLBase.solve(prob, alg; dt = 0.1, save_everystep = false)
            @test sol.t[end] ≈ 1.0
            @test abs(sol.u[end][1] - exact(1.0)) < 1.0e-7
        end
    end

    @testset "The state between step ends" begin
        three!(du, u, p, t) = (du[1] = -u[1]; du[2] = -2u[2]; du[3] = -3u[3]; nothing)
        function three_jac!(J, u, p, t)
            J .= 0.0
            J[1, 1] = -1.0
            J[2, 2] = -2.0
            return J[3, 3] = -3.0
        end
        u0 = [1.0, 1.0, 1.0]
        f = SciMLBase.ODEFunction(three!; jac = three_jac!)
        fixed = (dt = 0.1, adaptive = false)
        inner = (save_start = false, save_end = false)
        spans = (
            (SciMLBase.ODEProblem(f, u0, (0.0, 1.0)), [0.25, 0.55, 0.85], 0.5),
            (SciMLBase.ODEProblem(f, u0, (1.0, 0.0)), [0.85, 0.55, 0.25], 2.0),
        )
        function petsc_at(integ, t)
            h = integ.h
            PETScDiffEq.LibPETSc.TSInterpolate(h.petsclib, h.ts, integ.tdir * t, h.ctx.work)
            return PETScDiffEq._readvec!(similar(integ.u), h.petsclib, h.ctx.work)
        end
        function matches_petsc(prob, alg)
            integ = SciMLBase.init(prob, alg; fixed...)
            same = true
            while true
                SciMLBase.step!(integ)
                SciMLBase.done(integ) && return same
                mid = (integ.tprev + integ.t) / 2
                same &= integ(mid) == petsc_at(integ, mid)
            end
        end
        function dense_root(sol, level)
            g(t) = sol(t)[1] - level
            k = findfirst(i -> g(sol.t[i]) * g(sol.t[i + 1]) <= 0, 1:(length(sol.t) - 1))
            lo, hi = sol.t[k], sol.t[k + 1]
            while true
                mid = (lo + hi) / 2
                (mid == lo || mid == hi) && return lo
                g(lo) * g(mid) <= 0 ? (hi = mid) : (lo = mid)
            end
        end

        @testset "PETSc's own where it is at least cubic" begin
            cubic = (
                PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRosW("ra34pw2"),
                PETScDiffEq.TSARKIMEX("4"), PETScDiffEq.TSARKIMEX("5"),
                PETScDiffEq.TSImplicit("bdf"),
            )
            for alg in cubic, (prob, want, _) in spans
                @test matches_petsc(prob, alg)
                kw = (; fixed..., saveat = want, inner...)
                @test SciMLBase.solve(prob, alg; kw...).u ==
                    SciMLBase.solve!(SciMLBase.init(prob, alg; kw...)).u
            end
        end

        @testset "the cubic Hermite on the step's ends everywhere else" begin
            algs = Any[
                PETScDiffEq.TSImplicit("beuler"), PETScDiffEq.TSImplicit("cn"),
                PETScDiffEq.TSImplicit("theta", 0.7), PETScDiffEq.TSIRK(1),
                PETScDiffEq.TSIRK(2), PETScDiffEq.TSIRK(3),
            ]
            append!(algs, PETScDiffEq.TSMPRK([1], st) for st in ("2a22", "2a32", "p2", "p3"))
            append!(algs, PETScDiffEq.TSMPRK([1], [2], st) for st in ("2a23", "2a33"))
            append!(algs, PETScDiffEq.TSGeneric(st; explicit = true) for st in ("euler", "ssp", "glee", "rk"))
            append!(
                algs,
                PETScDiffEq.TSGeneric(st) for st in ("alpha", "theta", "beuler", "cn", "bdf", "arkimex", "rosw")
            )
            push!(
                algs, PETScDiffEq.TSRK("5dp", ["-ts_rk_type", "8vr"]),
                PETScDiffEq.TSRosW("ra34pw2", ["-ts_rosw_type", "rodas3"]),
            )
            # ark3 and the lassp pair are refused, and ars122 needs a split problem.
            families = (
                (PETScDiffEq.TSRK, PETScDiffEq._RK_ORDER, ("5dp",)),
                (
                    PETScDiffEq.TSRosW, PETScDiffEq._ROSW_ORDER,
                    ("ra34pw2", "lassp3p4s2c", "llssp3p4s2c", "ark3"),
                ),
                (PETScDiffEq.TSARKIMEX, PETScDiffEq._ARKIMEX_ORDER, ("4", "5", "ars122")),
            )
            for (family, table, skip) in families, st in sort(collect(keys(table)))
                st in skip || push!(algs, family(st))
            end
            for alg in algs, (prob, want, level) in spans
                dense = SciMLBase.solve(prob, alg; fixed...)
                expected = [dense(t) for t in want]
                kw = (; fixed..., saveat = want, inner...)
                @test SciMLBase.solve(prob, alg; kw...).u == expected
                @test SciMLBase.solve!(SciMLBase.init(prob, alg; kw...)).u == expected
                picked = SciMLBase.solve(prob, alg; fixed..., save_idxs = [3, 1])
                @test SciMLBase.solve(prob, alg; kw..., save_idxs = [3, 1]).u ==
                    [picked(t) for t in want]
                hit = Ref((NaN, Float64[]))
                cb = SciMLBase.ContinuousCallback(
                    (u, t, integ) -> u[1] - level,
                    integ -> (hit[] = (integ.t, copy(integ.u)); SciMLBase.terminate!(integ)),
                )
                SciMLBase.solve(prob, alg; fixed..., callback = cb)
                t_hit, u_hit = hit[]
                @test u_hit == dense(t_hit)
                @test abs(t_hit - dense_root(dense, level)) < 5.0e-15
            end
            glle = PETScDiffEq.TSGeneric("glle")
            for (prob, want, _) in spans
                dense = SciMLBase.solve(prob, glle; fixed...)
                @test SciMLBase.solve(prob, glle; fixed..., saveat = want, inner...).u ==
                    [dense(t) for t in want]
            end
            halves!(du, u, p, t) = (du[1] = -0.5u[1]; du[2] = -u[2]; du[3] = -1.5u[3]; nothing)
            for st in sort(collect(keys(PETScDiffEq._ARKIMEX_ORDER))), (_, want, _) in spans
                st in ("4", "5", "bpr3") && continue
                span = want[1] < want[end] ? (0.0, 1.0) : (1.0, 0.0)
                prob = SciMLBase.SplitODEProblem(halves!, halves!, u0, span)
                alg = PETScDiffEq.TSARKIMEX(st)
                dense = SciMLBase.solve(prob, alg; fixed...)
                kw = (; fixed..., saveat = want, inner...)
                @test SciMLBase.solve(prob, alg; kw...).u == [dense(t) for t in want]
                @test SciMLBase.solve!(SciMLBase.init(prob, alg; kw...)).u ==
                    [dense(t) for t in want]
            end
            driven!(du, u, p, t) = (du[1] = -u[1] + cos(3t); du[2] = sin(2t) * u[1]; nothing)
            function driven_jac!(J, u, p, t)
                J .= 0.0
                J[1, 1] = -1.0
                return J[2, 1] = sin(2t)
            end
            back = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(driven!; jac = driven_jac!), [1.0, 0.5], (1.0, 0.0),
            )
            back_want = [0.85, 0.55, 0.25]
            driven_algs = (
                PETScDiffEq.TSRK("4"), PETScDiffEq.TSRosW("2m"), PETScDiffEq.TSIRK(2),
                PETScDiffEq.TSARKIMEX("3"),
            )
            for alg in driven_algs
                dense = SciMLBase.solve(back, alg; fixed...)
                expected = [dense(t) for t in back_want]
                kw = (; fixed..., saveat = back_want, inner...)
                @test SciMLBase.solve(back, alg; kw...).u == expected
                @test SciMLBase.solve!(SciMLBase.init(back, alg; kw...)).u == expected
                level = (dense.u[4][1] + dense.u[5][1]) / 2
                hit = Ref(NaN)
                cb = SciMLBase.ContinuousCallback(
                    (u, t, integ) -> u[1] - level,
                    integ -> (hit[] = integ.t; SciMLBase.terminate!(integ)),
                )
                SciMLBase.solve(back, alg; fixed..., callback = cb)
                @test abs(hit[] - dense_root(dense, level)) < 5.0e-15
            end
        end

        @testset "an end's derivative is taken once, and only when asked for" begin
            prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
            alg = PETScDiffEq.TSRK("3bs")
            nf(; kw...) = SciMLBase.solve(prob, alg; fixed..., kw...).stats.nf
            nf_init(; kw...) =
                SciMLBase.solve!(SciMLBase.init(prob, alg; fixed..., kw...)).stats.nf
            plain = nf(save_everystep = false)
            idle = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            @test nf(save_everystep = false, callback = idle) == plain
            @test nf(saveat = [0.2, 0.5]) == plain
            @test nf_init(saveat = [0.2, 0.5]) == plain
            @test nf(saveat = [0.25, 0.28, 0.35]) == plain + 3
            @test nf_init(saveat = [0.25, 0.28, 0.35]) == plain + 3
            for count in (nf, nf_init)
                @test count(saveat = [0.25, 0.3], dense = true) ==
                    count(saveat = [0.25], dense = true)
                @test count(saveat = [0.3, 0.35], dense = true) ==
                    count(saveat = [0.3], dense = true) + 2
            end
            never = SciMLBase.ContinuousCallback((u, t, integ) -> u[1] + 1.0, integ -> nothing)
            @test nf(callback = never) == nf()
        end

        @testset "a derivative is not reused once its end has moved" begin
            prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
            alg = PETScDiffEq.TSRK("4")
            function hermite(t, t0, u0, f0, t1, u1, f1)
                dt = t1 - t0
                Θ = (t - t0) / dt
                return @. (1 - Θ) * u0 + Θ * u1 +
                    Θ * (Θ - 1) * ((1 - 2Θ) * (u1 - u0) + (Θ - 1) * dt * f0 + Θ * dt * f1)
            end
            expected(integ, t, rhs) = hermite(
                t, integ.tprev, integ.uprev, rhs(integ.uprev, integ.tprev),
                integ.t, integ.u, rhs(integ.u, integ.t),
            )
            function halfway(pr)
                integ = SciMLBase.init(pr, alg; fixed...)
                for _ in 1:5
                    SciMLBase.step!(integ)
                end
                integ(integ.t - 0.05)
                return integ
            end

            integ = halfway(prob)
            SciMLBase.set_u!(integ, [2.0])
            SciMLBase.step!(integ)
            @test integ(0.55) == expected(integ, 0.55, (u, t) -> -u)
            SciMLBase.terminate!(integ)

            integ = halfway(prob)
            SciMLBase.change_t_via_interpolation!(integ, 0.45)
            @test integ(0.42) == expected(integ, 0.42, (u, t) -> -u)
            SciMLBase.terminate!(integ)

            forced!(du, u, p, t) = (du[1] = -u[1] + cos(t); nothing)
            integ = halfway(SciMLBase.ODEProblem(forced!, [1.0], (0.0, 1.0)))
            SciMLBase.set_t!(integ, 0.7)
            @test integ(0.6) == expected(integ, 0.6, (u, t) -> -u .+ cos(t))
            SciMLBase.step!(integ)
            @test integ(0.75) == expected(integ, 0.75, (u, t) -> -u .+ cos(t))
            SciMLBase.terminate!(integ)

            scaled!(du, u, p, t) = (du[1] = -p[1] * u[1]; nothing)
            integ = halfway(SciMLBase.ODEProblem(scaled!, [1.0], (0.0, 1.0), [1.0]))
            integ.p[1] = 3.0
            SciMLBase.u_modified!(integ, true)
            SciMLBase.step!(integ)
            @test integ(0.55) == expected(integ, 0.55, (u, t) -> -3.0 .* u)
            SciMLBase.terminate!(integ)

            lift = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.5, integ -> (integ.u[1] += 1.0),
            )
            fired = Ref(false)
            jump = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t >= 0.5 && !fired[],
                integ -> (integ.u[1] += 1.0; fired[] = true),
            )
            for (cb, want) in ((lift, [0.35, 0.75, 0.95]), (jump, [0.45, 0.55]))
                fired[] = false
                at = SciMLBase.solve(prob, alg; fixed..., saveat = want, callback = cb)
                fired[] = false
                dense = SciMLBase.solve(prob, alg; fixed..., callback = cb)
                @test [at.u[findfirst(==(t), at.t)] for t in want] == [dense(t) for t in want]
            end

            midstep(integ) = integ((integ.tprev + integ.t) / 2)
            retuned = Ref(false)
            retune! = integ -> (midstep(integ); integ.p[1] = 3.0; retuned[] = true)
            retunes = (
                (
                    SciMLBase.ContinuousCallback((u, t, integ) -> u[1] - 0.7, retune!),
                    [0.45, 0.65, 0.95],
                ),
                (
                    SciMLBase.DiscreteCallback((u, t, integ) -> t >= 0.5 && !retuned[], retune!),
                    [0.55, 0.95],
                ),
            )
            tunable = SciMLBase.ODEProblem(scaled!, [1.0], (0.0, 1.0), [1.0])
            for (cb, want) in retunes
                tunable.p[1] = 1.0
                retuned[] = false
                at = SciMLBase.solve(tunable, alg; fixed..., saveat = want, callback = cb)
                tunable.p[1] = 1.0
                retuned[] = false
                dense = SciMLBase.solve(tunable, alg; fixed..., callback = cb)
                @test [at.u[findfirst(==(t), at.t)] for t in want] == [dense(t) for t in want]
            end

            tuned = Vector{Float64}[]
            tuned_yet = Ref(false)
            looked_after = Ref(false)
            retune_peeking! = integ -> begin
                push!(tuned, midstep(integ))
                integ.p[1] = 3.0
                push!(tuned, midstep(integ))
                tuned_yet[] = true
            end
            watcher = SciMLBase.DiscreteCallback(
                (u, t, integ) -> (
                    tuned_yet[] && !looked_after[] &&
                        (push!(tuned, midstep(integ)); looked_after[] = true); false
                ),
                integ -> nothing,
            )
            retunes_at = (
                affect! -> SciMLBase.DiscreteCallback(
                    (u, t, integ) -> t >= 0.5 && !tuned_yet[], affect!,
                ),
                affect! -> SciMLBase.ContinuousCallback((u, t, integ) -> u[1] - 0.7, affect!),
            )
            peeked = Vector{Float64}[]
            for at in retunes_at
                empty!(tuned)
                tuned_yet[] = false
                looked_after[] = false
                tunable.p[1] = 1.0
                SciMLBase.solve(
                    tunable, alg; fixed...,
                    callback = SciMLBase.CallbackSet(at(retune_peeking!), watcher),
                )
                @test tuned == fill(tuned[1], 3)
                push!(peeked, tuned[1])
            end

            retune_blind! = integ -> begin
                integ.p[1] = 3.0
                push!(tuned, midstep(integ))
                tuned_yet[] = true
            end
            for (at, want) in zip(retunes_at, peeked),
                    kw in ((;), (save_everystep = false,), (dense = false,))
                empty!(tuned)
                tuned_yet[] = false
                tunable.p[1] = 1.0
                SciMLBase.solve(tunable, alg; fixed..., kw..., callback = at(retune_blind!))
                @test tuned == [want]
            end
            tunable.p[1] = 1.0

            driven = SciMLBase.ODEProblem(forced!, [1.0], (0.0, 1.0))
            for a in (alg, PETScDiffEq.TSRosW("2m"))
                seen = Vector{Float64}[]
                peeked_at = Ref(NaN)
                lifted = Ref(false)
                lift_peeking! = integ -> begin
                    peeked_at[] = (integ.tprev + integ.t) / 2
                    push!(seen, midstep(integ))
                    integ.u[1] += 1.0
                    push!(seen, midstep(integ))
                    lifted[] = true
                end
                looked = Ref(false)
                later = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> (lifted[] && !looked[] && (push!(seen, midstep(integ)); looked[] = true); false),
                    integ -> nothing,
                )
                lifts = (
                    SciMLBase.DiscreteCallback((u, t, integ) -> t >= 0.5 && !lifted[], lift_peeking!),
                    SciMLBase.ContinuousCallback((u, t, integ) -> u[1] - 0.9, lift_peeking!),
                )
                for cb in lifts
                    empty!(seen)
                    lifted[] = false
                    looked[] = false
                    sol = SciMLBase.solve(
                        driven, a; fixed..., callback = SciMLBase.CallbackSet(cb, later),
                    )
                    @test seen == fill(seen[1], 3)
                    @test seen[1] == sol(peeked_at[])
                end

                nolift = SciMLBase.solve(driven, a; fixed..., saveat = [0.45])
                saved = SavedValues(Float64, Vector{Float64})
                lifted[] = false
                lift = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> t >= 0.5 - 1.0e-12 && !lifted[],
                    integ -> (integ.u[1] += 1.0; lifted[] = true),
                )
                saving = SavingCallback((u, t, integ) -> copy(u), saved; saveat = [0.45])
                SciMLBase.solve(driven, a; fixed..., callback = SciMLBase.CallbackSet(lift, saving))
                @test saved.t == [0.45]
                @test saved.saveval == [nolift.u[findfirst(==(0.45), nolift.t)]]

                integ = SciMLBase.init(driven, a; fixed...)
                for _ in 1:5
                    SciMLBase.step!(integ)
                end
                before = midstep(integ)
                SciMLBase.set_u!(integ, integ.u .+ 1.0)
                @test midstep(integ) == before
                SciMLBase.terminate!(integ)
            end
        end

        @testset "saved times outside the step at hand" begin
            prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
            back = SciMLBase.ODEProblem(decay!, [1.0], (1.0, 0.0))
            overshoot = ["-ts_exact_final_time", "interpolate"]
            algs = (PETScDiffEq.TSRK("3bs", overshoot), PETScDiffEq.TSRK("5dp", overshoot))
            cases = ((prob, [0.5, 0.95, 1.0]), (back, [0.5, 0.05, 0.0]))
            long = (dt = 0.3, adaptive = false)
            for alg in algs, (pr, want) in cases
                sol = SciMLBase.solve(pr, alg; long..., saveat = want)
                @test sol.t == want
                @test sol.u[end] ==
                    SciMLBase.solve(pr, alg; long..., save_everystep = false).u[end]
            end
            for alg in (PETScDiffEq.TSRK("4"), PETScDiffEq.TSRK("5dp"))
                integ = SciMLBase.init(prob, alg; fixed...)
                for _ in 1:4
                    SciMLBase.step!(integ)
                end
                @test_throws ArgumentError SciMLBase.add_saveat!(integ, 0.25)
                @test_throws ArgumentError SciMLBase.add_saveat!(integ, prevfloat(integ.t))
                here = integ.t
                SciMLBase.add_saveat!(integ, here)
                @test here in SciMLBase.solve!(integ).t
            end
        end

        @testset "with a mass matrix or a DAEProblem, only PETSc's" begin
            mass = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    three!; jac = three_jac!, mass_matrix = Diagonal([2.0, 1.0, 1.0]),
                ), u0, (0.0, 1.0),
            )
            # PETSc prints to the process's own stderr, so it is caught there.
            function quietly(run)
                path, io = mktemp()
                result = redirect_stderr(io) do
                    try
                        run()
                    catch err
                        err
                    end
                end
                close(io)
                return result, read(path, String)
            end
            crossing(; kw...) = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 0.8, integ -> nothing; kw...,
            )
            function refuses_between_steps(alg)
                for run in (
                        () -> SciMLBase.solve(mass, alg; fixed..., saveat = [0.25]),
                        () -> SciMLBase.solve(mass, alg; fixed..., callback = crossing()),
                    )
                    err, printed = quietly(run)
                    @test err isa ArgumentError && occursin("no interpolant", err.msg)
                    @test isempty(printed)
                end
                @test SciMLBase.solve(mass, alg; fixed..., saveat = [0.2, 0.5]).t ==
                    [0.2, 0.5]
                @test SciMLBase.solve(
                    mass, alg; fixed..., callback = crossing(rootfind = SciMLBase.NoRootFind),
                ).retcode == SciMLBase.ReturnCode.Success
            end

            none = (
                "r34prw", "r3prl2", "rodas3", "rodaspr", "rodaspr2", "grk4t", "shamp4",
                "veldd4", "4l", "prssp2", "ars443", "bpr3",
            )
            refused = ("lassp3p4s2c", "llssp3p4s2c", "ark3", "assp3p3s1c", "ars122")
            families = (
                (PETScDiffEq.TSRosW, PETScDiffEq._ROSW_ORDER),
                (PETScDiffEq.TSARKIMEX, PETScDiffEq._ARKIMEX_ORDER),
            )
            for (family, table) in families, st in sort(collect(keys(table)))
                st in refused && continue
                st in none ? refuses_between_steps(family(st)) :
                    @test matches_petsc(mass, family(st))
            end
            for st in ("beuler", "cn", "theta", "bdf")
                @test matches_petsc(mass, PETScDiffEq.TSImplicit(st))
            end
            refuses_between_steps(PETScDiffEq.TSGeneric("rosw", ["-ts_rosw_type", "rodas3"]))
            @test_throws "a mass matrix with TSIRK" SciMLBase.solve(
                mass, PETScDiffEq.TSGeneric("irk", ["-pc_type", "pbjacobi"]); fixed...,
            )
            @test matches_petsc(mass, PETScDiffEq.TSGeneric("rosw"))
            refuses_between_steps(PETScDiffEq.TSRosW("ra34pw2", ["-ts_rosw_type", "rodas3"]))
            @test matches_petsc(mass, PETScDiffEq.TSRosW("rodas3", ["-ts_rosw_type", "ra34pw2"]))
            calls = Ref(0)
            counting!(du, u, p, t) = (calls[] += 1; three!(du, u, p, t))
            counted = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    counting!; jac = three_jac!, mass_matrix = Diagonal([2.0, 1.0, 1.0]),
                ), u0, (0.0, 1.0),
            )
            for alg in (
                    PETScDiffEq.TSRosW("rodas3"),
                    PETScDiffEq.TSGeneric("rosw", ["-ts_rosw_type", "rodas3"]),
                )
                whole = SciMLBase.solve(counted, alg; fixed...).stats.nf
                calls[] = 0
                quietly(() -> SciMLBase.solve(counted, alg; fixed..., saveat = [0.25]))
                @test calls[] <= 5 * whole / 10
            end

            rates = [1.0, 2.0, 3.0]
            dae = SciMLBase.DAEProblem(
                (r, du, u, p, t) -> (r .= du .+ rates .* u; nothing), -rates, u0, (0.0, 1.0),
            )
            for st in ("beuler", "cn", "theta", "bdf")
                @test matches_petsc(dae, PETScDiffEq.TSDAE(st))
            end
        end
    end

    @testset "Solution statistics" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        sol = SciMLBase.solve(
            prob, PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"]); dt = 0.1,
        )
        @test sol.stats !== nothing
        @test sol.stats.nf > 0
        @test sol.stats.naccept == 10
        @test sol.stats.nreject == 0

        for span in ((0.0, 1.0), (1.0, 0.0))
            reads = Int[]
            cb = SciMLBase.DiscreteCallback(
                (u, t, integ) -> true, integ -> push!(reads, integ.sol.stats.naccept);
                save_positions = (false, false),
            )
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(decay!, [1.0], span), PETScDiffEq.TSRK("5dp");
                callback = cb, abstol = 1.0e-8, reltol = 1.0e-8,
            )
            @test reads == 1:sol.stats.naccept
        end
    end

    @testset "Unsupported keywords warn rather than being dropped" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"])
        @test_logs (:warn,) match_mode = :any SciMLBase.solve(
            prob, alg; dt = 0.1, calck = false,
        )
        @test_logs (:warn,) match_mode = :any SciMLBase.solve(
            prob, alg; dt = 0.1, internalnorm = (u, t) -> maximum(abs, u),
        )
        @test_logs min_level = Logging.Warn SciMLBase.solve(prob, alg; dt = 0.1)
        for kw in (
                (; progress = true), (; failfactor = 4.0),
                (; step_limiter = (u, integ, p, t) -> nothing),
                (; stage_limiter = (u, integ, p, t) -> nothing),
                (; advance_to_tstop = true), (; stop_at_next_tstop = true),
            )
            all(in(PETScDiffEq.DiffEqBase.allowedkeywords), keys(kw)) || continue
            @test_logs (:warn, r"does not support") SciMLBase.solve(prob, alg; dt = 0.1, kw...)
        end
        @test_logs min_level = Logging.Warn SciMLBase.solve(
            prob, alg; dt = 0.1, progress = false, progress_steps = 10,
            advance_to_tstop = false, stop_at_next_tstop = false,
        )
    end

    @testset "error norms follow OrdinaryDiffEq's defaults and switches" begin
        known = SciMLBase.ODEFunction(decay!; analytic = (u0, p, t) -> u0 .* exp(-t))
        prob = SciMLBase.ODEProblem(known, [1.0], (0.0, 1.0))
        for kw in ((;), (; tstops = [0.5]))
            norms(; more...) = sort(
                collect(
                    keys(SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); kw..., more...).errors),
                ),
            )
            @test norms() == [:final, :l2, :l∞]
            @test norms(timeseries_errors = false) == [:final]
            @test norms(dense_errors = true) == [:L2, :L∞, :final, :l2, :l∞]
        end
    end

    @testset "No Jacobian buffer is allocated when none is used" begin
        n = 500
        prob = SciMLBase.ODEProblem(decay!, ones(n), (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"])
        run() = SciMLBase.solve(prob, alg; dt = 0.1, save_everystep = false)
        run()
        @test (@allocated run()) < 800 * 1024
    end

    @testset "Mass matrices" begin
        scaled!(du, u, p, t) = (du[1] = -u[1]; du[2] = -u[2]; nothing)
        function scaled_jac!(J, u, p, t)
            J .= 0.0
            J[1, 1] = -1.0
            return J[2, 2] = -1.0
        end
        M = [2.0 0.0; 0.0 1.0]
        prob = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(scaled!; mass_matrix = M), [1.0, 1.0], (0.0, 1.0),
        )

        @testset "M u' = f is solved, not u' = f" begin
            for alg in (
                    PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSRosW("ra34pw2"),
                )
                sol = SciMLBase.solve(
                    prob, alg; dt = 0.005, reltol = 1.0e-11, abstol = 1.0e-13,
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test isapprox(sol.u[end][1], exp(-0.5); atol = 1.0e-6)
                @test isapprox(sol.u[end][2], exp(-1.0); atol = 1.0e-6)
            end
        end

        @testset "with an analytic Jacobian" begin
            jac_prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(scaled!; mass_matrix = M, jac = scaled_jac!),
                [1.0, 1.0], (0.0, 1.0),
            )
            sol = SciMLBase.solve(
                jac_prob, PETScDiffEq.TSImplicit("bdf");
                dt = 0.005, reltol = 1.0e-11, abstol = 1.0e-13,
            )
            @test isapprox(sol.u[end][1], exp(-0.5); atol = 1.0e-6)
        end

        @testset "a singular M gives the index-1 DAE, not an ODE" begin
            dae!(du, u, p, t) = (du[1] = -u[1]; du[2] = u[2] - u[1]; nothing)
            dae_prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(dae!; mass_matrix = [1.0 0.0; 0.0 0.0]),
                [1.0, 1.0], (0.0, 1.0),
            )
            sol = SciMLBase.solve(
                dae_prob, PETScDiffEq.TSImplicit("bdf");
                dt = 0.005, reltol = 1.0e-10, abstol = 1.0e-12,
            )
            @test isapprox(sol.u[end][1], exp(-1.0); atol = 1.0e-6)
            @test isapprox(sol.u[end][2], exp(-1.0); atol = 1.0e-6)
        end

        @testset "rejected where it cannot be applied" begin
            @test_throws ArgumentError SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.01,
            )
            @test_throws ArgumentError SciMLBase.solve(
                prob, PETScDiffEq.TSGeneric("euler"; explicit = true); dt = 0.01,
            )
        end

        @testset "a mass matrix that couples the components" begin
            coupled = [2.0 0.5; 0.0 1.0]
            exact = [1.5exp(-0.5) - 0.5exp(-1.0), exp(-1.0)]
            for f in (
                    SciMLBase.ODEFunction(scaled!; mass_matrix = coupled),
                    SciMLBase.ODEFunction(
                        scaled!; mass_matrix = coupled, jac = scaled_jac!,
                    ),
                )
                for alg in (
                        PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSRosW("ra34pw2"),
                    )
                    sol = SciMLBase.solve(
                        SciMLBase.ODEProblem(f, [1.0, 1.0], (0.0, 1.0)), alg;
                        dt = 0.005, reltol = 1.0e-11, abstol = 1.0e-13,
                    )
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test isapprox(sol.u[end], exact; atol = 1.0e-6)
                end
            end
        end

        @testset "the mass matrix reaches the Jacobian, not just the residual" begin
            strong = SciMLBase.ODEFunction(
                scaled!; mass_matrix = [1.0 5.0; 0.0 1.0], jac = scaled_jac!,
            )
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(strong, [1.0, 1.0], (0.0, 1.0)),
                PETScDiffEq.TSImplicit("bdf", ["-ts_max_snes_failures", "1"]);
                dt = 0.01, adaptive = false,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test isapprox(sol.u[end], [6exp(-1.0), exp(-1.0)]; atol = 1.0e-4)
        end

        @testset "with a sparse jac_prototype" begin
            proto = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
            sparse_prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    scaled!; mass_matrix = M, jac = scaled_jac!, jac_prototype = proto,
                ), [1.0, 1.0], (0.0, 1.0),
            )
            sol = SciMLBase.solve(
                sparse_prob, PETScDiffEq.TSImplicit("bdf");
                dt = 0.005, reltol = 1.0e-11, abstol = 1.0e-13,
            )
            @test isapprox(sol.u[end][1], exp(-0.5); atol = 1.0e-6)

            offdiag = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    scaled!; mass_matrix = [2.0 0.5; 0.0 1.0], jac = scaled_jac!,
                    jac_prototype = proto,
                ), [1.0, 1.0], (0.0, 1.0),
            )
            # The shift lands on M's (1, 2) entry, which the prototype leaves out.
            sol = SciMLBase.solve(
                offdiag, PETScDiffEq.TSImplicit("bdf");
                dt = 0.005, reltol = 1.0e-11, abstol = 1.0e-13,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test isapprox(sol.u[end][1], 1.5exp(-0.5) - 0.5exp(-1.0); atol = 1.0e-6)
            @test isapprox(sol.u[end][2], exp(-1.0); atol = 1.0e-6)
        end

        @testset "a Diagonal or sparse mass matrix is not made dense" begin
            band(n, l, d) = spdiagm(-1 => fill(l, n - 1), 0 => fill(d, n), 1 => fill(l, n - 1))
            function run(M)
                n = size(M, 1)
                A = band(n, 1.0, -2.0)
                f = SciMLBase.ODEFunction(
                    (du, u, p, t) -> (mul!(du, A, u); nothing);
                    jac = (J, u, p, t) -> (copyto!(nonzeros(J), nonzeros(A)); nothing),
                    jac_prototype = A, mass_matrix = M,
                )
                return SciMLBase.solve(
                    SciMLBase.ODEProblem(f, sinpi.((1:n) ./ (n + 1)), (0.0, 0.1)),
                    PETScDiffEq.TSImplicit("beuler"); dt = 0.01, adaptive = false,
                )
            end
            lumped(n) = Diagonal(1 .+ (1:n) ./ n)
            for M in (lumped(20), band(20, 1 / 6, 2 / 3))
                @test run(M).u ≈ run(Matrix(M)).u rtol = 1.0e-12
            end
            for M in (lumped(2000), band(2000, 1 / 6, 2 / 3))
                @test (@allocated run(M)) < 16_000_000
            end
        end
    end

    @testset "An operator-valued right-hand side is rejected" begin
        A = SciMLOperators.MatrixOperator([-1.0 0.0; 0.0 -2.0])
        zero!(du, u, p, t) = (du .= 0.0; nothing)
        @test_throws ArgumentError SciMLBase.solve(
            SciMLBase.ODEProblem(SciMLBase.ODEFunction(A), [1.0, 1.0], (0.0, 1.0)),
            PETScDiffEq.TSImplicit("bdf"); dt = 0.01,
        )
        @test_throws ArgumentError SciMLBase.solve(
            SciMLBase.SplitODEProblem(A, zero!, [1.0, 1.0], (0.0, 1.0)),
            PETScDiffEq.TSARKIMEX("3"); dt = 0.01,
        )
    end

    @testset "adaptive = false gives fixed steps" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp")
        fixed = SciMLBase.solve(prob, alg; dt = 0.05, adaptive = false)
        @test fixed.stats.naccept == 20
        @test abs(fixed.u[end][1] - exp(-1.0)) < 1.0e-8
        @test SciMLBase.solve(prob, alg; dt = 0.05).stats.naccept < 20
    end

    @testset "dtmin and dtmax reach the controller" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp")
        free = SciMLBase.solve(prob, alg; dt = 0.05)
        capped = SciMLBase.solve(prob, alg; dt = 0.05, dtmax = 0.1)
        @test maximum(diff(free.t)) > 0.1
        @test maximum(diff(capped.t)) <= 0.1 + 1.0e-10
        @test capped.stats.naccept > free.stats.naccept
    end

    @testset "Subtype setters take effect" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        exact = exp(-1.0)
        function finest_order(alg)
            errs = [
                abs(SciMLBase.solve(prob, alg; dt = dt, adaptive = false).u[end][1] - exact)
                    for dt in (0.05, 0.025, 0.0125)
            ]
            return log2(errs[end - 1] / errs[end])
        end
        @test isapprox(finest_order(PETScDiffEq.TSRosW("2m")), 2; atol = 0.15)
        @test isapprox(finest_order(PETScDiffEq.TSRosW("ra34pw2")), 3; atol = 0.15)
        @test isapprox(finest_order(PETScDiffEq.TSARKIMEX("2e")), 2; atol = 0.15)
        @test isapprox(finest_order(PETScDiffEq.TSARKIMEX("3")), 3; atol = 0.15)
        @test isapprox(finest_order(PETScDiffEq.TSImplicit("theta", 1.0)), 1; atol = 0.15)
        @test isapprox(finest_order(PETScDiffEq.TSImplicit("theta", 0.5)), 2; atol = 0.15)
    end

    @testset "PETSc options that alter stepping" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))

        @testset "interpolated final time keeps sol.t sorted and inside tspan" begin
            for dt in (0.3, 0.7)
                sol = SciMLBase.solve(
                    prob, PETScDiffEq.TSRK("5dp", ["-ts_exact_final_time", "interpolate"]);
                    dt = dt,
                )
                @test issorted(sol.t)
                @test sol.t[end] ≈ 1.0
                @test maximum(sol.t) <= 1.0 + 1.0e-12
                @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-4
            end
        end

        @testset "a cancelled monitor still yields the final state" begin
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp", ["-ts_monitor_cancel"]); dt = 0.1,
            )
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.t == [1.0]
            @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-4
        end

        @testset "an option PETSc reads inside the solve is in force there" begin
            n = 8
            A = spdiagm(-1 => ones(n - 1), 0 => fill(-2.0, n), 1 => ones(n - 1))
            lin = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction((du, u, p, t) -> mul!(du, A, u); jac_prototype = A),
                ones(n), (0.0, 1.0),
            )
            coloured(opts) = PETScDiffEq.TSImplicit(
                "beuler", opts; autodiff = PETScDiffEq.AutoFiniteDiff(),
            )
            sol = SciMLBase.solve(lin, coloured(String[]); dt = 0.1)
            # One colour per column, where PETSc's default finds three.
            natural = coloured(["-mat_coloring_type", "natural"])
            more = SciMLBase.solve(lin, natural; dt = 0.1)
            @test more.stats.nf - sol.stats.nf == (n - 3) * sol.stats.nnonliniter
            @test more.u[end] ≈ sol.u[end]
            stepped = SciMLBase.solve!(SciMLBase.init(lin, natural; dt = 0.1))
            @test stepped.stats.nf == more.stats.nf
        end

        @testset "the integrator sets up under its options" begin
            function stage_type(integ)
                pl = integ.h.petsclib
                stage, name = Ref{Ptr{Cvoid}}(C_NULL), Ref{Ptr{Cchar}}(C_NULL)
                ccall(
                    PETScDiffEq._symbol(pl, :SNESGetFunction), Cint,
                    (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cvoid}, Ptr{Cvoid}),
                    PETScDiffEq._snes(pl, integ.h.ts.ptr), stage, C_NULL, C_NULL,
                )
                ccall(
                    PETScDiffEq._symbol(pl, :VecGetType), Cint,
                    (Ptr{Cvoid}, Ptr{Ptr{Cchar}}), stage[], name,
                )
                return unsafe_string(name[])
            end
            # TSIRK creates its stage vector in TSSetUp, of the type `-vec_type` names.
            integ = SciMLBase.init(prob, PETScDiffEq.TSIRK(2, ["-vec_type", "mpi"]); dt = 0.1)
            @test stage_type(integ) == "mpi"
            SciMLBase.reinit!(integ)
            @test stage_type(integ) == "mpi"
            SciMLBase.terminate!(integ)
        end
    end

    @testset "Integrator interface" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))

        @testset "the integrator reports what OrdinaryDiffEq's does" begin
            held(integ) = (
                t = PETScDiffEq.LibPETSc.TSGetTolerances(integ.h.petsclib, integ.h.ts);
                (t[1], t[3])
            )
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1)
            @test (integ.opts.abstol, integ.opts.reltol) == (1.0e-6, 1.0e-3)
            @test held(integ) == (1.0e-6, 1.0e-3)
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, reltol = 1.0e-5)
            integ.opts.abstol = 1.0e-9
            @test held(integ) == (1.0e-9, 1.0e-5)

            SciMLBase.step!(integ)
            SciMLBase.step!(integ)
            @test integ.dt == integ.t - integ.tprev
            @test integ(integ.t - integ.dt) == integ.uprev
            @test integ.sol.stats.naccept == 2
            @test integ.sol.stats.nf > 0
            saved, exactly = SciMLBase.savevalues!(integ)
            @test (saved, exactly) == (false, false)
            SciMLBase.change_t_via_interpolation!(integ, (integ.tprev + integ.t) / 2)
            @test integ.dt == integ.t - integ.tprev
            @test integ(integ.t - integ.dt) == integ.uprev

            never = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            capped = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, maxiters = 0,
                callback = never,
            )
            @test capped.retcode == SciMLBase.ReturnCode.MaxIters
            @test capped.t == [0.0]

            fast!(du, u, p, t) = (du[1] = -50.0 * u[1]; nothing)
            quick = SciMLBase.ODEProblem(fast!, [1.0], (0.0, 1.0))
            integ = SciMLBase.init(
                quick, PETScDiffEq.TSRK("5dp"); dt = 0.01, abstol = 1.0e-10, reltol = 1.0e-10,
            )
            integ.opts.dtmin = 0.1
            @test SciMLBase.solve!(integ).retcode == SciMLBase.ReturnCode.DtLessThanMin
        end

        @testset "the last step after the integrator has finished" begin
            for alg in (
                    PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRosW("ra34pw2"),
                    PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSARKIMEX("4"),
                    PETScDiffEq.TSRK("4"),
                )
                integ = SciMLBase.init(prob, alg; dt = 0.1, adaptive = false)
                SciMLBase.solve!(integ)
                t = (integ.tprev + integ.t) / 2
                @test integ(t) == integ.sol(t)
                @test abs(integ(t)[1] - exp(-t)) < 1.0e-3
            end

            stop = SciMLBase.DiscreteCallback((u, t, integ) -> t >= 0.5, SciMLBase.terminate!)
            integ = SciMLBase.init(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, callback = stop,
            )
            SciMLBase.solve!(integ)
            @test integ.finished
            t = (integ.tprev + integ.t) / 2
            @test integ(t) == integ.sol(t)

            massive = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(decay!; mass_matrix = fill(2.0, 1, 1)), [1.0], (0.0, 1.0),
            )
            integ = SciMLBase.init(massive, PETScDiffEq.TSImplicit("bdf"); dt = 0.1)
            SciMLBase.solve!(integ)
            @test_throws "freed PETSc's interpolant" integ((integ.tprev + integ.t) / 2)
        end

        @testset "the call forms OrdinaryDiffEq's integrator takes" begin
            rot!(du, u, p, t) = (du[1] = -u[2]; du[2] = u[1]; nothing)
            spin = SciMLBase.ODEProblem(rot!, [1.0, 0.0], (0.0, 1.0))
            slope(t) = [-sin(t), cos(t)]
            rhs(u, t) = (du = similar(u); rot!(du, u, nothing, t); du)
            # Measured over every step at dt = 0.1, the Hermite slope is off by 8e-6 for 5dp
            # and rk 4 and by 1.5e-3 for cn, whose steps carry that error themselves.
            for (alg, tol) in (
                    (PETScDiffEq.TSRK("5dp"), 2.0e-5), (PETScDiffEq.TSRK("4"), 2.0e-5),
                    (PETScDiffEq.TSImplicit("cn"), 4.0e-3),
                )
                integ = SciMLBase.init(spin, alg; dt = 0.1, adaptive = false)
                SciMLBase.step!(integ)
                SciMLBase.step!(integ)
                tm = (integ.tprev + integ.t) / 2
                u = integ(tm)
                du = integ(tm, Val{1})
                @test integ(tm, Val{0}) == u
                @test du isa Vector{Float64} && length(du) == 2
                for q in (0.25, 0.5, 0.75)
                    t = integ.tprev + q * integ.dt
                    @test maximum(abs.(integ(t, Val{1}) .- slope(t))) < tol
                end
                @test integ(integ.t, Val{1}) == SciMLBase.get_du(integ)
                @test integ(integ.tprev, Val{1}) == rhs(integ.uprev, integ.tprev)
                @test integ(tm; idxs = 1) === u[1]
                @test integ(tm; idxs = [2, 1]) == u[[2, 1]]
                @test integ(tm; idxs = 1:2) == u
                @test integ(tm, Val{1}; idxs = 2) === du[2]
                ts = [integ.tprev, tm, integ.t]
                @test integ(ts) == [integ(t) for t in ts]
                @test integ(ts, Val{1}) == [integ(t, Val{1}) for t in ts]
                @test integ(ts; idxs = 1) == [integ(t)[1] for t in ts]
                @test integ((integ.tprev, tm)) isa Vector{Vector{Float64}}
                out = zeros(2)
                @test integ(out, tm, Val{1}) === out && out == du
                @test integ(out, tm, Val{0}) === out && out == u
                one = zeros(1)
                @test integ(one, tm; idxs = [2]) === one && one == [u[2]]
                @test integ(one, tm, Val{1}; idxs = 2) == [du[2]]
                @test integ(out, ts) == fill(integ(integ.t), 3) && out == integ.u
                @test_throws ArgumentError integ(tm, Val{2})
                @test_throws ArgumentError integ(integ.t + 1.0, Val{1})
                SciMLBase.solve!(integ)
                tm = (integ.tprev + integ.t) / 2
                u = integ(tm)
                du = integ(tm, Val{1})
                for q in (0.25, 0.5, 0.75)
                    t = integ.tprev + q * integ.dt
                    @test maximum(abs.(integ(t, Val{1}) .- slope(t))) < tol
                end
                @test integ(integ.t, Val{1}) == SciMLBase.get_du(integ)
                @test integ(tm; idxs = 2) === u[2]
                @test integ(tm, Val{1}; idxs = [1]) == [du[1]]
                @test integ([integ.tprev, tm]) == [integ(integ.tprev), u]
                @test integ(out, tm, Val{1}) === out && out == du
            end

            # Where the package interpolates itself, Val{1} is that interpolant's own slope.
            integ = SciMLBase.init(spin, PETScDiffEq.TSRK("4"); dt = 0.1, adaptive = false)
            SciMLBase.step!(integ)
            SciMLBase.step!(integ)
            tm, h = (integ.tprev + integ.t) / 2, 1.0e-6
            @test maximum(abs.(integ(tm, Val{1}) .- (integ(tm + h) .- integ(tm - h)) ./ 2h)) <
                1.0e-9
            SciMLBase.terminate!(integ)

            back = SciMLBase.ODEProblem(decay!, [1.0], (1.0, 0.0))
            for alg in (PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRK("4"))
                integ = SciMLBase.init(back, alg; dt = 0.1, adaptive = false)
                SciMLBase.step!(integ)
                SciMLBase.step!(integ)
                for q in (0.25, 0.5, 0.75)
                    t = integ.tprev + q * integ.dt
                    @test abs(integ(t, Val{1})[1] + exp(1 - t)) < 6.0e-5
                end
                @test integ(integ.t, Val{1}) == SciMLBase.get_du(integ)
                @test integ(integ.tprev, Val{1}) == -integ.uprev
                SciMLBase.terminate!(integ)
            end

            single = SciMLBase.ODEProblem(decay!, Float32[1.0], (0.0f0, 1.0f0))
            integ = SciMLBase.init(single, PETScDiffEq.TSRK("5dp"); dt = 0.1f0, adaptive = false)
            SciMLBase.step!(integ)
            SciMLBase.step!(integ)
            tq = integ.tprev + integ.dt / 4
            if Float32 in PETScDiffEq._loaded_builds()
                @test integ(tq, Val{1}) isa Vector{Float32}
                @test integ(tq; idxs = 1) isa Float32
            end
            @test abs(integ(tq, Val{1})[1] + exp(-tq)) < 3.0e-5
            SciMLBase.terminate!(integ)

            massive = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(decay!; mass_matrix = fill(2.0, 1, 1)), [1.0], (0.0, 1.0),
            )
            integ = SciMLBase.init(massive, PETScDiffEq.TSImplicit("bdf"); dt = 0.1)
            SciMLBase.step!(integ)
            tm = (integ.tprev + integ.t) / 2
            @test integ(tm) isa Vector{Float64}
            @test_throws "no slope" integ(tm, Val{1})
            @test_throws "no slope" integ(integ.t, Val{1})
            SciMLBase.terminate!(integ)
        end

        @testset "init, step! and solve! reproduce solve exactly" begin
            for alg in (
                    PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSImplicit("bdf"),
                    PETScDiffEq.TSRosW("ra34pw2"),
                )
                ref = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false)
                integ = SciMLBase.init(prob, alg; dt = 0.1, adaptive = false)
                @test integ isa PETScDiffEq.PETScIntegrator
                @test integ.t == 0.0
                @test integ.u == [1.0]
                @test integ.sol.retcode == SciMLBase.ReturnCode.Default
                @test !SciMLBase.done(integ)

                SciMLBase.step!(integ)
                @test integ.t ≈ ref.t[2]
                @test integ.u ≈ ref.u[2]
                @test integ.tprev == 0.0
                @test integ.uprev == [1.0]

                sol = SciMLBase.solve!(integ)
                @test SciMLBase.done(integ)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.t == ref.t
                @test all(a == b for (a, b) in zip(sol.u, ref.u))
                @test sol.stats.naccept == ref.stats.naccept
            end
        end

        @testset "step!(integ, dt) advances through the generic loop" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            SciMLBase.step!(integ, 0.3)
            @test integ.t ≈ 0.3
            @test !SciMLBase.done(integ)
            SciMLBase.solve!(integ)
        end

        @testset "terminate! stops early and releases the solver" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            SciMLBase.step!(integ)
            SciMLBase.step!(integ)
            SciMLBase.terminate!(integ)
            @test SciMLBase.done(integ)
            @test integ.sol.retcode == SciMLBase.ReturnCode.Terminated
            @test integ.sol.t[end] ≈ 0.2
            @test length(integ.sol.t) == 3
            @test integ.h.destroyed
            @test_throws ArgumentError SciMLBase.step!(integ)
            @test integ.t ≈ 0.2
        end

        @testset "stepping past the end is refused" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            SciMLBase.solve!(integ)
            t_end = integ.t
            @test_throws ArgumentError SciMLBase.step!(integ)
            @test integ.t == t_end
            @test SciMLBase.done(integ)
        end

        @testset "step!(integ, dt) terminates at the end of the span" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            for _ in 1:3
                SciMLBase.step!(integ, 0.3)
            end
            @test integ.t == 1.0
            SciMLBase.step!(integ, 0.3)
            @test integ.t == 1.0
            @test SciMLBase.done(integ)
        end

        @testset "save_everystep = false keeps only the endpoints" begin
            integ = SciMLBase.init(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false,
                save_everystep = false,
            )
            sol = SciMLBase.solve!(integ)
            @test sol.t == [0.0, 1.0]
            @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-7
        end

        @testset "a blow-up finishes as Unstable" begin
            blowup!(du, u, p, t) = (du[1] = u[1]^2; nothing)
            integ = SciMLBase.init(
                SciMLBase.ODEProblem(blowup!, [1.0], (0.0, 5.0)),
                PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false,
            )
            sol = SciMLBase.solve!(integ)
            @test SciMLBase.done(integ)
            @test sol.retcode == SciMLBase.ReturnCode.Unstable
        end

        @testset "a user exception surfaces from step! and finishes the integrator" begin
            boom!(du, u, p, t) = (t > 0.25 && error("user rhs failed"); du[1] = -u[1]; nothing)
            integ = SciMLBase.init(
                SciMLBase.ODEProblem(boom!, [1.0], (0.0, 1.0)),
                PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false,
            )
            err = try
                SciMLBase.solve!(integ)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("user rhs failed", err.msg)
            @test SciMLBase.done(integ)
        end

        @testset "saveat matches solve exactly" begin
            for (alg, kw, tol) in (
                    (PETScDiffEq.TSRK("5dp"), (; saveat = 0.25), 1.0e-7),
                    (PETScDiffEq.TSRK("5dp"), (; saveat = [0.3, 0.7]), 1.0e-7),
                    (PETScDiffEq.TSRK("5dp"), (; saveat = [0.3, 0.7], save_start = false), 1.0e-7),
                    (PETScDiffEq.TSImplicit("bdf"), (; saveat = [0.15, 0.85]), 5.0e-3),
                )
                ref = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false, kw...)
                integ = SciMLBase.init(prob, alg; dt = 0.1, adaptive = false, kw...)
                sol = SciMLBase.solve!(integ)
                @test sol.t == ref.t
                @test all(a == b for (a, b) in zip(sol.u, ref.u))
                for i in eachindex(sol.t)
                    @test abs(sol.u[i][1] - exp(-sol.t[i])) < tol
                end
            end
        end

        @testset "DiscreteCallback" begin
            fired = Ref(false)
            jump = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t >= 0.5 && !fired[],
                integ -> (integ.u[1] += 1.0; fired[] = true; nothing),
            )
            expected(t) = t < 0.5 ? exp(-t) : (exp(-0.5) + 1.0) * exp(-(t - 0.5))

            @testset "affect! changes the state PETSc integrates from" begin
                for alg in (PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSImplicit("bdf"))
                    fired[] = false
                    sol = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false, callback = jump)
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test fired[]
                    @test abs(sol.u[end][1] - expected(1.0)) < 5.0e-3
                    i = findfirst(==(0.5), sol.t)
                    @test i !== nothing && sol.t[i + 1] == 0.5
                    @test sol.u[i + 1][1] ≈ sol.u[i][1] + 1.0
                end
            end

            @testset "solve with a callback equals solve! on init" begin
                fired[] = false
                a = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, callback = jump)
                fired[] = false
                b = SciMLBase.solve!(
                    SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, callback = jump),
                )
                @test a.t == b.t
                @test all(x == y for (x, y) in zip(a.u, b.u))
            end

            @testset "terminate! from affect!" begin
                stop = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> t >= 0.3, integ -> SciMLBase.terminate!(integ),
                )
                sol = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, callback = stop)
                @test sol.retcode == SciMLBase.ReturnCode.Terminated
                @test sol.t[end] ≈ 0.3
            end

            @testset "initialize and finalize hooks run once" begin
                n_init = Ref(0)
                n_fin = Ref(0)
                hooked = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> false, integ -> nothing;
                    initialize = (cb, u, t, integ) -> (n_init[] += 1; nothing),
                    finalize = (cb, u, t, integ) -> (n_fin[] += 1; nothing),
                )
                SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, callback = hooked)
                @test n_init[] == 1
                @test n_fin[] == 1
            end

            @testset "a CallbackSet of discrete callbacks works" begin
                fired[] = false
                cbs = SciMLBase.CallbackSet(jump)
                sol = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, callback = cbs)
                @test fired[]
                @test abs(sol.u[end][1] - expected(1.0)) < 1.0e-6
            end

            @testset "save_positions[1] saves the state before affect!" begin
                double(sp) = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> t == 0.5, integ -> (integ.u .*= 2; nothing);
                    save_positions = sp,
                )
                alg = PETScDiffEq.TSRK("5dp")
                kw = (tstops = [0.5], abstol = 1.0e-10, reltol = 1.0e-10)
                pre = (callback = double((true, false)), save_everystep = false)
                sol = SciMLBase.solve(prob, alg; kw..., pre...)
                @test sol.t == [0.0, 0.5, 1.0]
                @test abs(sol.u[2][1] - exp(-0.5)) < 1.0e-9
                @test SciMLBase.solve!(SciMLBase.init(prob, alg; kw..., pre...)).t == sol.t

                both = double((true, true))
                at = SciMLBase.solve(prob, alg; kw..., callback = both, saveat = [0.0, 0.3, 0.6, 1.0])
                @test at.t == [0.0, 0.3, 0.5, 0.5, 0.6, 1.0]
                @test at.u[4] == 2 .* at.u[3]
                @test count(==(0.5), SciMLBase.solve(prob, alg; kw..., callback = both).t) == 2
                twice = SciMLBase.solve(
                    prob, alg; kw..., callback = SciMLBase.CallbackSet(both, both),
                    save_everystep = false,
                )
                @test first.(twice.u[2:5]) == [1, 2, 2, 4] .* twice.u[2][1]

                ticks = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> true, integ -> nothing; save_positions = (true, false),
                )
                every = SciMLBase.solve(
                    prob, alg; callback = ticks, save_everystep = false, dt = 0.25,
                    adaptive = false,
                )
                @test every.t == [0.0, 0.25, 0.5, 0.75, 1.0]
            end

            @testset "an affect! that declares no change is not written back" begin
                touched = Ref(0)
                quiet = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> true,
                    integ -> (touched[] += 1; SciMLBase.derivative_discontinuity!(integ, false)),
                )
                ref = SciMLBase.solve(prob, PETScDiffEq.TSImplicit("bdf"); dt = 0.1, adaptive = false)
                sol = SciMLBase.solve(
                    prob, PETScDiffEq.TSImplicit("bdf"); dt = 0.1, adaptive = false,
                    callback = quiet, save_everystep = false,
                )
                @test touched[] == 10
                @test sol.u[end] == ref.u[end]
            end

            @testset "proposed dt and savevalues! hooks" begin
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
                @test SciMLBase.get_proposed_dt(integ) ≈ 0.1
                SciMLBase.set_proposed_dt!(integ, 0.05)
                SciMLBase.step!(integ)
                @test integ.t ≈ 0.05
                @test SciMLBase.get_dt(integ) ≈ 0.05
                n = length(integ.sol.t)
                @test SciMLBase.savevalues!(integ) == (false, false)
                @test length(integ.sol.t) == n
                @test SciMLBase.savevalues!(integ, true) == (true, true)
                @test length(integ.sol.t) == n + 1
                @test integ.sol.t[end] ≈ 0.05
                SciMLBase.solve!(integ)
            end

        end

        @testset "repeated init/solve! cycles do not crash" begin
            for _ in 1:30
                integ = SciMLBase.init(prob, PETScDiffEq.TSImplicit("bdf"); dt = 0.1)
                @test SciMLBase.solve!(integ).retcode == SciMLBase.ReturnCode.Success
            end
        end

        @testset "the rest of SciMLBase's integrator interface" begin
            RC = SciMLBase.ReturnCode
            lv = SciMLBase.ODEProblem(
                lotka_volterra!, [1.0, 1.0], (0.0, 2.0), [1.5, 1.0, 3.0, 1.0],
            )
            first_dt(p, alg; kw...) = (
                i = SciMLBase.init(p, alg; kw...); dt = i.dt; SciMLBase.terminate!(i); dt
            )

            @testset "check_error, check_error! and postamble! as OrdinaryDiffEq has them" begin
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"))
                @test SciMLBase.check_error(integ) == RC.Success
                @test integ.sol.retcode == RC.Default
                SciMLBase.step!(integ)
                @test integ.sol.retcode == RC.Success
                @test SciMLBase.check_error!(integ) == RC.Success
                @test !integ.finished
                SciMLBase.postamble!(integ)
                @test SciMLBase.done(integ)
                @test integ.sol.retcode == RC.Success
                @test integ.sol.t == [0.0, integ.t]
                @test integ.sol.u[end] == integ.u
                SciMLBase.postamble!(integ)
                @test length(integ.sol.t) == 2

                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"))
                SciMLBase.postamble!(integ)
                @test integ.sol.retcode == RC.Default
                @test integ.sol.t == [0.0]

                finals = Ref(0)
                cb = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> false, integ -> nothing;
                    finalize = (c, u, t, integ) -> (finals[] += 1),
                )
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); callback = cb)
                SciMLBase.step!(integ)
                SciMLBase.postamble!(integ)
                SciMLBase.postamble!(integ)
                @test finals[] == 1

                integ = SciMLBase.init(lv, PETScDiffEq.TSRK("5dp"); maxiters = 3)
                SciMLBase.solve!(integ)
                @test SciMLBase.check_error(integ) == RC.MaxIters
                @test SciMLBase.check_error!(integ) == RC.MaxIters
                @test integ.sol.retcode == RC.MaxIters

                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"))
                SciMLBase.solve!(integ)
                @test SciMLBase.check_error(integ) == RC.Success
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"))
                SciMLBase.step!(integ)
                SciMLBase.terminate!(integ)
                @test SciMLBase.check_error(integ) == RC.Terminated

                seen = RC.T[]
                watch = SciMLBase.DiscreteCallback(
                    (u, t, integ) -> false, integ -> nothing;
                    finalize = (c, u, t, integ) -> push!(seen, SciMLBase.check_error(integ)),
                )
                stop = SciMLBase.DiscreteCallback((u, t, integ) -> t > 0.5, SciMLBase.terminate!)
                SciMLBase.solve(
                    lv, PETScDiffEq.TSRK("5dp"); callback = SciMLBase.CallbackSet(stop, watch),
                )
                integ = SciMLBase.init(lv, PETScDiffEq.TSRK("5dp"); callback = watch)
                SciMLBase.step!(integ)
                SciMLBase.terminate!(integ, RC.Failure)
                @test seen == [RC.Terminated, RC.Failure]
            end

            @testset "last_step_failed is a fixed step whose Newton solve failed" begin
                cubic!(du, u, p, t) = (du[1] = -1.0e4 * u[1]^3 + sin(t); nothing)
                stiff = SciMLBase.ODEProblem(cubic!, [10.0], (0.0, 1.0))
                one_newton = ["-snes_max_it", "1"]
                integ = SciMLBase.init(
                    stiff, PETScDiffEq.TSImplicit("beuler", one_newton); dt = 0.5,
                    adaptive = false,
                )
                @test !SciMLBase.last_step_failed(integ)
                SciMLBase.step!(integ)
                @test integ.sol.retcode == RC.ConvergenceFailure
                @test SciMLBase.check_error(integ) == RC.ConvergenceFailure
                @test SciMLBase.last_step_failed(integ)
                integ = SciMLBase.init(stiff, PETScDiffEq.TSImplicit("bdf", one_newton); dt = 0.5)
                SciMLBase.step!(integ)
                @test integ.sol.retcode == RC.Success
                @test integ.sol.stats.nnonlinconvfail > 0
                @test !SciMLBase.last_step_failed(integ)
            end

            @testset "the proposed step is signed, and can come from another integrator" begin
                lv_back = SciMLBase.ODEProblem(lotka_volterra!, [1.0, 1.0], (2.0, 0.0), lv.p)
                integ = SciMLBase.init(lv_back, PETScDiffEq.TSRK("5dp"))
                @test SciMLBase.get_proposed_dt(integ) == integ.dt < 0
                SciMLBase.step!(integ)
                proposed = SciMLBase.get_proposed_dt(integ)
                @test proposed < 0
                SciMLBase.set_proposed_dt!(integ, proposed / 2)
                @test SciMLBase.get_proposed_dt(integ) == proposed / 2
                SciMLBase.solve!(integ)
                @test SciMLBase.get_proposed_dt(integ) < 0

                ahead = SciMLBase.init(lv, PETScDiffEq.TSRK("5dp"))
                SciMLBase.step!(ahead)
                SciMLBase.step!(ahead)
                integ = SciMLBase.init(lv, PETScDiffEq.TSRK("5dp"))
                SciMLBase.set_proposed_dt!(integ, ahead)
                @test SciMLBase.get_proposed_dt(integ) == SciMLBase.get_proposed_dt(ahead)
                SciMLBase.solve!(integ)
                SciMLBase.solve!(ahead)
            end

            @testset "auto_dt_reset! takes the step init would take from here" begin
                integ = SciMLBase.init(lv, PETScDiffEq.TSRK("5dp"); dt = 0.5)
                SciMLBase.auto_dt_reset!(integ)
                @test integ.dt == SciMLBase.get_proposed_dt(integ) ==
                    first_dt(lv, PETScDiffEq.TSRK("5dp"))
                SciMLBase.terminate!(integ)

                integ = SciMLBase.init(
                    lv, PETScDiffEq.TSRK("5dp"); save_everystep = false, tstops = [0.1],
                )
                SciMLBase.step!(integ)
                nf = integ.sol.stats.nf
                SciMLBase.auto_dt_reset!(integ)
                @test integ.sol.stats.nf == nf + 2
                here = SciMLBase.remake(lv; u0 = copy(integ.u), tspan = (integ.t, 2.0))
                @test integ.dt == first_dt(here, PETScDiffEq.TSRK("5dp"); tstops = [0.1])
                @test integ.dt < first_dt(here, PETScDiffEq.TSRK("5dp"))
                SciMLBase.step!(integ)
                @test integ.t == 0.1
                SciMLBase.solve!(integ)
                @test integ.sol.retcode == RC.Success

                split = SciMLBase.SplitODEProblem(
                    (du, u, p, t) -> (du .= -u; nothing), (du, u, p, t) -> (du .= sin(t); nothing),
                    [1.0], (0.0, 1.0),
                )
                integ = SciMLBase.init(split, PETScDiffEq.TSARKIMEX("4"); dt = 0.1)
                SciMLBase.step!(integ)
                counts = (integ.sol.stats.nf, integ.sol.stats.nf2)
                SciMLBase.auto_dt_reset!(integ)
                @test (integ.sol.stats.nf, integ.sol.stats.nf2) == counts .+ 2
                SciMLBase.terminate!(integ)

                lv_back = SciMLBase.ODEProblem(lotka_volterra!, [1.0, 1.0], (2.0, 0.0), lv.p)
                integ = SciMLBase.init(lv_back, PETScDiffEq.TSRosW("ra34pw2"); dt = 0.5)
                SciMLBase.auto_dt_reset!(integ)
                @test integ.dt == first_dt(lv_back, PETScDiffEq.TSRosW("ra34pw2")) < 0
                SciMLBase.terminate!(integ)

                massive = SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(decay!; mass_matrix = fill(2.0, 1, 1)), [1.0],
                    (0.0, 1.0),
                )
                integ = SciMLBase.init(massive, PETScDiffEq.TSImplicit("bdf"); dt = 0.1)
                SciMLBase.auto_dt_reset!(integ)
                @test integ.dt == first_dt(massive, PETScDiffEq.TSImplicit("bdf"))
                SciMLBase.terminate!(integ)

                integ = SciMLBase.init(
                    prob, PETScDiffEq.TSGeneric("rk"; explicit = true); dt = 0.1,
                )
                SciMLBase.auto_dt_reset!(integ)
                @test integ.dt == first_dt(prob, PETScDiffEq.TSRK("3bs"))
                SciMLBase.terminate!(integ)

                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.02, adaptive = false)
                SciMLBase.step!(integ)
                SciMLBase.auto_dt_reset!(integ)
                here = SciMLBase.remake(prob; u0 = copy(integ.u), tspan = (integ.t, 1.0))
                @test integ.dt == first_dt(here, PETScDiffEq.TSRK("5dp"))
                @test SciMLBase.get_proposed_dt(integ) == 0.02
                SciMLBase.step!(integ)
                @test integ.dt ≈ 0.02
                SciMLBase.reinit!(integ; reset_dt = true)
                @test integ.dt == first_dt(prob, PETScDiffEq.TSRK("5dp"))
                @test SciMLBase.get_proposed_dt(integ) == 0.02
                SciMLBase.step!(integ)
                @test integ.t == 0.02
                SciMLBase.terminate!(integ)
            end

            @testset "reinit! takes reset_dt and reinit_cache" begin
                integ = SciMLBase.init(lv, PETScDiffEq.TSRK("5dp"); dt = 0.5)
                SciMLBase.step!(integ)
                SciMLBase.step!(integ)
                proposed = SciMLBase.get_proposed_dt(integ)
                SciMLBase.reinit!(integ; reset_dt = false)
                @test SciMLBase.get_proposed_dt(integ) == integ.dt == proposed
                SciMLBase.reinit!(integ; reset_dt = true, reinit_cache = false)
                @test integ.dt == first_dt(lv, PETScDiffEq.TSRK("5dp"))
                SciMLBase.reinit!(integ, [2.0, 0.5]; reset_dt = true)
                @test integ.dt ==
                    first_dt(SciMLBase.remake(lv; u0 = [2.0, 0.5]), PETScDiffEq.TSRK("5dp"))
                SciMLBase.reinit!(integ)
                @test integ.dt == 0.5
                SciMLBase.solve!(integ)
                ended = SciMLBase.get_proposed_dt(integ)
                SciMLBase.reinit!(integ; reset_dt = false)
                @test SciMLBase.get_proposed_dt(integ) == integ.dt == ended
                SciMLBase.step!(integ)
                @test integ.t == ended
                @test SciMLBase.solve!(integ).retcode == RC.Success

                integ = SciMLBase.init(
                    prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false, tstops = [0.33],
                )
                SciMLBase.step!(integ)
                SciMLBase.set_proposed_dt!(integ, 0.05)
                SciMLBase.reinit!(integ; reset_dt = false)
                steps = Float64[]
                while !SciMLBase.done(integ)
                    SciMLBase.step!(integ)
                    push!(steps, integ.dt)
                end
                @test 0.33 in integ.sol.t
                @test maximum(steps) ≈ 0.05

                integ = SciMLBase.init(
                    prob, PETScDiffEq.TSGeneric("rk"; explicit = true); dt = 0.1,
                )
                SciMLBase.step!(integ)
                SciMLBase.reinit!(integ; reset_dt = true)
                @test integ.t == 0.0
                @test integ.dt == first_dt(prob, PETScDiffEq.TSRK("3bs"))
                SciMLBase.solve!(integ)
                @test integ.sol.retcode == RC.Success
            end

            @testset "a reinit! whose f throws frees the TS it made" begin
                live() = count(h -> !h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
                refuses!(du, u, p, t) = (any(>(5.0), u) && error("f refuses"); du .= -u; nothing)
                refusing = SciMLBase.ODEProblem(refuses!, ones(3), (0.0, 1.0))
                for (reset_dt, dense) in ((nothing, true), (true, false))
                    integ = SciMLBase.init(refusing, PETScDiffEq.TSRK("5dp"); dt = 0.1, dense)
                    SciMLBase.step!(integ)
                    before = live()
                    @test_throws "f refuses" SciMLBase.reinit!(integ, fill(10.0, 3); reset_dt)
                    @test live() <= before
                    @test SciMLBase.solve!(integ).retcode == RC.Success
                end
            end

            @testset "set_abstol! and set_reltol! hold for the steps after" begin
                integ = SciMLBase.init(lv, PETScDiffEq.TSRK("5dp"); dt = 0.01)
                SciMLBase.set_abstol!(integ, 1.0e-10)
                SciMLBase.set_reltol!(integ, 1.0e-8)
                @test (integ.opts.abstol, integ.opts.reltol) == (1.0e-10, 1.0e-8)
                ref = SciMLBase.solve(
                    lv, PETScDiffEq.TSRK("5dp"); dt = 0.01, abstol = 1.0e-10, reltol = 1.0e-8,
                )
                sol = SciMLBase.solve!(integ)
                @test sol.t == ref.t
                @test sol.u == ref.u
            end

            @testset "the state cannot change length" begin
                integ = SciMLBase.init(lv, PETScDiffEq.TSRK("5dp"))
                SciMLBase.step!(integ)
                for call in (
                        () -> resize!(integ, 3), () -> deleteat!(integ, 1),
                        () -> SciMLBase.addat!(integ, 3:3),
                    )
                    @test_throws "cannot change the length of the state" call()
                end
                @test SciMLBase.solve!(integ).retcode == RC.Success
            end

            @testset "change_t_via_interpolation! rewrites what was saved past the new time" begin
                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
                SciMLBase.step!(integ)
                SciMLBase.step!(integ)
                SciMLBase.change_t_via_interpolation!(integ, 0.15, Val{false}, nothing)
                @test integ.t == 0.15
                @test abs(integ.u[1] - exp(-0.15)) < 1.0e-8
                SciMLBase.terminate!(integ)

                integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
                SciMLBase.step!(integ)
                SciMLBase.step!(integ)
                SciMLBase.change_t_via_interpolation!(integ, 0.15, Val{true})
                @test integ.sol.t == [0.0, 0.1, 0.15]
                @test integ.sol.u[end] == integ.u
                SciMLBase.solve!(integ)
                @test issorted(integ.sol.t)
                @test allunique(integ.sol.t)
                @test maximum(t -> abs(integ.sol(t)[1] - exp(-t)), 0.1:0.005:0.15) < 5.0e-8

                back = SciMLBase.ODEProblem(decay!, [1.0], (1.0, 0.0))
                integ = SciMLBase.init(back, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
                SciMLBase.step!(integ)
                SciMLBase.step!(integ)
                SciMLBase.change_t_via_interpolation!(integ, 0.85, Val{true})
                @test integ.sol.t == [1.0, 0.9, 0.85]
                SciMLBase.solve!(integ)
                @test issorted(integ.sol.t; rev = true)
                @test allunique(integ.sol.t)

                integ = SciMLBase.init(
                    prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false,
                    saveat = [0.15, 0.5],
                )
                SciMLBase.step!(integ)
                SciMLBase.step!(integ)
                @test integ.sol.t == [0.15]
                SciMLBase.change_t_via_interpolation!(integ, 0.12, Val{true})
                @test isempty(integ.sol.t)
                SciMLBase.set_u!(integ, 2 .* integ.u)
                sol = SciMLBase.solve!(integ)
                @test sol.t == [0.15, 0.5]
                @test abs(sol.u[1][1] - 2exp(-0.15)) < 1.0e-7
            end
        end

        @testset "iter and opts read as OrdinaryDiffEq's do" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            @test integ.iter == 0
            SciMLBase.step!(integ)
            SciMLBase.step!(integ)
            @test integ.iter == 2
            o = integ.opts
            @test o.maxiters == 1000000
            @test o.save_everystep && o.save_start && o.save_end && o.dense && o.calck
            @test o.save_idxs === nothing
            @test o.internalnorm === DiffEqBase.ODE_DEFAULT_NORM
            @test o.unstable_check === nothing && o.isoutofdomain === nothing
            @test o.callback isa SciMLBase.CallbackSet && isempty(o.callback.discrete_callbacks)
            @test o.tstops === integ.tstops && o.tstops == [1.0]
            @test o.saveat === integ.h.ctx.saveat && isempty(o.saveat)
            @test isempty(o.d_discontinuities)

            never = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            blow(dt, u, p, t) = false
            neg(u, p, t) = false
            integ = SciMLBase.init(
                prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, tstops = [0.5], saveat = [0.25],
                d_discontinuities = [0.35], save_idxs = 1, callback = never,
                unstable_check = blow, isoutofdomain = neg,
            )
            o = integ.opts
            @test o.tstops == [0.35, 0.5, 1.0]
            @test o.saveat == [0.25]
            @test o.d_discontinuities == [0.35]
            @test o.save_idxs == [1]
            @test !o.dense && !o.save_everystep
            @test collect(o.callback.discrete_callbacks) == [never]
            @test o.unstable_check === blow && o.isoutofdomain === neg
            SciMLBase.add_tstop!(integ, 0.7)
            SciMLBase.add_saveat!(integ, 0.6)
            @test o.tstops == [0.35, 0.5, 0.7, 1.0]
            @test o.saveat == [0.25, 0.6]
            @test_throws ArgumentError integ.opts.dense = true
            @test_throws ArgumentError integ.opts.tstops = [0.9]
            SciMLBase.reinit!(integ; tstops = [0.4])
            @test integ.iter == 0
            @test integ.opts.tstops === integ.tstops && integ.opts.tstops == [0.35, 0.4, 1.0]

            stiff!(du, u, p, t) = (du[1] = -1000u[1]; nothing)
            quick = SciMLBase.ODEProblem(stiff!, [1.0], (0.0, 0.01))
            integ = SciMLBase.init(quick, PETScDiffEq.TSRK("5dp"); dt = 0.5)
            for _ in 1:3
                SciMLBase.step!(integ)
                @test integ.iter == integ.sol.stats.naccept + integ.sol.stats.nreject
            end
            @test integ.sol.stats.nreject > 0
        end

        @testset "assigning an opts field takes effect" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            integ.opts.maxiters = 3
            sol = SciMLBase.solve!(integ)
            @test sol.retcode == SciMLBase.ReturnCode.MaxIters
            @test sol.stats.naccept == 3 && integ.iter == 3
            @test integ.t ≈ 0.3

            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            SciMLBase.step!(integ)
            integ.opts.save_everystep = false
            @test SciMLBase.solve!(integ).t ≈ [0.0, 0.1, 1.0]

            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            integ.opts.save_start = false
            @test SciMLBase.solve!(integ).t[1] ≈ 0.1
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            integ.opts.save_end = false
            @test SciMLBase.solve!(integ).t[end] ≈ 0.9

            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            SciMLBase.step!(integ)
            integ.opts.unstable_check = (dt, u, p, t) -> t > 0.45
            sol = SciMLBase.solve!(integ)
            @test sol.retcode == SciMLBase.ReturnCode.Unstable
            @test sol.t[end] ≈ 0.5

            asked = Ref(0)
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1)
            SciMLBase.step!(integ)
            integ.opts.isoutofdomain = (u, p, t) -> (asked[] += 1; false)
            SciMLBase.step!(integ)
            @test asked[] == 1
        end

        @testset "the integrator plot recipe and DiffEqCallbacks agree with Tsit5" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            series = RecipesBase.apply_recipe(Dict{Symbol, Any}(), integ)
            @test length(series) == 1 && series[1].plotattributes[:denseplot] == false
            @test length.(series[1].args) == (1, 1)
            @test (series[1].args[1][1], series[1].args[2][1]) == (0.0, 1.0)
            SciMLBase.step!(integ)
            series = RecipesBase.apply_recipe(Dict{Symbol, Any}(:denseplot => false), integ)
            @test (series[1].args[1][1], series[1].args[2][1]) == (integ.t, integ.u[1])

            function saved(alg; kw...)
                v = DiffEqCallbacks.SavedValues(Float64, Float64)
                cb = DiffEqCallbacks.SavingCallback((u, t, integ) -> u[1], v; saveat = 0.0:0.25:1.0)
                SciMLBase.solve(prob, alg; kw..., callback = cb)
                return v
            end
            ref, mine = saved(Tsit5()), saved(PETScDiffEq.TSRK("5dp"); dt = 0.1)
            @test mine.t == ref.t
            @test maximum(abs.(mine.saveval .- ref.saveval)) < 1.0e-3

            function fired(alg; kw...)
                ts = Float64[]
                cb = PresetTimeCallback([0.3, 0.6], integ -> push!(ts, integ.t))
                SciMLBase.solve(prob, alg; kw..., callback = cb)
                return ts
            end
            @test fired(PETScDiffEq.TSRK("5dp"); dt = 0.1) == fired(Tsit5()) == [0.3, 0.6]

            faster = SciMLBase.ODEProblem((du, u, p, t) -> (du[1] = -2u[1]; nothing), [1.0], (0.0, 100.0))
            steady = DiffEqCallbacks.TerminateSteadyState(1.0e-6, 1.0e-6)
            ref = SciMLBase.solve(faster, Tsit5(); callback = steady)
            sol = SciMLBase.solve(faster, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = steady)
            @test sol.retcode == ref.retcode == SciMLBase.ReturnCode.Terminated
            @test sol.t[end] < 100 && abs(sol.u[end][1]) < 1.0e-6 && abs(ref.u[end][1]) < 1.0e-6
        end
    end

    @testset "a replaced integ.p is the one solved with" begin
        scaled!(du, u, p, t) = (du[1] = -p * u[1]; nothing)
        prob = SciMLBase.ODEProblem(scaled!, [1.0], (0.0, 1.0), 1.0)
        tight = (abstol = 1.0e-10, reltol = 1.0e-10)
        for alg in (PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRosW("ra34pw2"))
            cb = SciMLBase.DiscreteCallback((u, t, integ) -> t == 0.5, integ -> (integ.p = 2.0))
            sol = SciMLBase.solve(prob, alg; tight..., callback = cb, tstops = [0.5])
            @test abs(sol.u[end][1] - exp(-1.5)) < 2.0e-10
            integ = SciMLBase.init(prob, alg; tight..., tstops = [0.5])
            SciMLBase.step!(integ, 0.5, true)
            integ.p = 2.0
            @test abs(SciMLBase.solve!(integ).u[end][1] - exp(-1.5)) < 2.0e-10
            SciMLBase.reinit!(integ)
            @test abs(SciMLBase.solve!(integ).u[end][1] - exp(-2.0)) < 2.0e-10
        end
        bdf = PETScDiffEq.TSImplicit("bdf")
        integ = SciMLBase.init(prob, bdf; dt = 0.01, adaptive = false)
        while integ.t < 1.0
            integ.p = 1.0
            SciMLBase.step!(integ)
        end
        @test integ.u == SciMLBase.solve(prob, bdf; dt = 0.01, adaptive = false).u[end]
    end

    @testset "Only methods with an error estimate adapt" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        steps(alg, rt) = SciMLBase.solve(
            prob, alg; dt = 1.0e-3, reltol = rt, abstol = rt * 1.0e-2,
        ).stats.naccept

        @testset "these respond to tolerance" begin
            for alg in (
                    PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRosW("ra34pw2"),
                    PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSARKIMEX("3"),
                )
                @test steps(alg, 1.0e-8) > steps(alg, 1.0e-3)
            end
        end

        @testset "these do not, and say so" begin
            for sub in ("beuler", "cn", "theta")
                alg = PETScDiffEq.TSImplicit(sub)
                @test_logs (:warn,) match_mode = :any SciMLBase.solve(
                    prob, alg; dt = 1.0e-3, reltol = 1.0e-8, abstol = 1.0e-10,
                )
                @test steps(alg, 1.0e-3) == 1000
                @test steps(alg, 1.0e-10) == 1000
            end
        end
    end

    @testset "Failure is reported as failure" begin
        blowup!(du, u, p, t) = (du[1] = u[1]^2; nothing)
        sol = SciMLBase.solve(
            SciMLBase.ODEProblem(blowup!, [1.0], (0.0, 5.0)),
            PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"]); dt = 0.1,
        )
        @test !SciMLBase.successful_retcode(sol)
        @test sol.retcode == SciMLBase.ReturnCode.Unstable
    end

    @testset "A user exception reaches the caller" begin
        boom!(du, u, p, t) = (t > 0.25 && error("user rhs failed"); du[1] = -u[1]; nothing)
        prob = SciMLBase.ODEProblem(boom!, [1.0], (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"])
        err = try
            SciMLBase.solve(prob, alg; dt = 0.1)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("user rhs failed", err.msg)
    end

    @testset "The trajectory is well formed under awkward PETSc options" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        @testset "exact_final_time interpolate does not overshoot tspan" begin
            sol = SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp", ["-ts_exact_final_time", "interpolate"]);
                dt = 0.3,
            )
            @test issorted(sol.t)
            @test maximum(sol.t) <= 1.0 + 1.0e-10
            @test sol.t[end] ≈ 1.0
        end
        @testset "a cancelled monitor still yields the final state" begin
            sol = SciMLBase.solve(
                prob,
                PETScDiffEq.TSRK(
                    "5dp", ["-ts_monitor_cancel", "-ts_adapt_type", "none"],
                ); dt = 0.1,
            )
            @test sol.t[end] ≈ 1.0
            @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-7
        end
    end

    @testset "TSRosW on a right-hand side that depends on t" begin
        forced!(du, u, p, t) = (du[1] = -u[1] + cos(t); nothing)
        autonomous!(du, u, p, t) = (du[1] = -u[1] + cos(u[2]); du[2] = 1.0; nothing)
        function autonomous_jac!(J, u, p, t)
            J .= 0.0
            J[1, 1] = -1.0
            return J[1, 2] = -sin(u[2])
        end
        exact = (cos(2.0) + sin(2.0) + exp(-2.0)) / 2
        function order(prob, st)
            alg = PETScDiffEq.TSRosW(st, ["-ts_adapt_type", "none"])
            errs = [
                abs(SciMLBase.solve(prob, alg; dt = dt).u[end][1] - exact)
                    for dt in (0.02, 0.01)
            ]
            return log2(errs[1] / errs[2])
        end
        forced = SciMLBase.ODEProblem(forced!, [1.0], (0.0, 2.0))
        for st in ("ra34pw2", "ra3pw", "r34prw")
            @test order(forced, st) > 2.8
        end
        @test order(forced, "2m") > 1.8
        with_time = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(autonomous!; jac = autonomous_jac!), [1.0, 0.0], (0.0, 2.0),
        )
        for st in ("sandu3", "rodas3", "grk4t")
            @test order(forced, st) < 1.2
        end
        for st in ("sandu3", "rodas3")
            @test order(with_time, st) > 2.8
        end
        @test order(with_time, "grk4t") > 3.5
    end

    @testset "ra3pw's estimate misses the error on a linear problem" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        function err(tol)
            sol = SciMLBase.solve(prob, PETScDiffEq.TSRosW("ra3pw"); dt = 0.01, reltol = tol)
            return abs(sol.u[end][1] - exp(-1.0))
        end
        errs = err.((1.0e-4, 1.0e-8))
        @test isapprox(errs[1], errs[2]; rtol = 1.0e-12)
        @test errs[2] > 1.0e-3
    end

    @testset "a subtype adapts exactly when PETSc gives it an error estimate" begin
        LibPETSc = PETScDiffEq.LibPETSc
        prob = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = decay_jac!), [1.0], (0.0, 1.0),
        )
        split = SciMLBase.SplitODEProblem(decay!, decay!, [1.0], (0.0, 1.0))
        refused = String[]
        for (family, table) in (
                (PETScDiffEq.TSRK, PETScDiffEq._RK_ORDER),
                (PETScDiffEq.TSRosW, PETScDiffEq._ROSW_ORDER),
                (PETScDiffEq.TSARKIMEX, PETScDiffEq._ARKIMEX_ORDER),
            )
            for st in sort(collect(keys(table)))
                alg = family(st)
                pr = family === PETScDiffEq.TSARKIMEX && st == "ars122" ? split : prob
                integ = try
                    SciMLBase.init(pr, alg; dt = 0.01)
                catch err
                    err isa ArgumentError || rethrow()
                    push!(refused, "$(nameof(family))(\"$st\")")
                    continue
                end
                pl = integ.h.petsclib
                adapt_type = LibPETSc.TSAdaptGetType(pl, LibPETSc.TSGetAdapt(pl, integ.h.ts))
                @test PETScDiffEq._adapts(integ.h.ctx.alg_name) == (adapt_type == "basic")
                SciMLBase.terminate!(integ)
            end
        end
        @test sort(refused) ==
            ["TSRosW(\"ark3\")", "TSRosW(\"lassp3p4s2c\")", "TSRosW(\"llssp3p4s2c\")"]
        @test !SciMLBase.isadaptive(SciMLBase.init(prob, PETScDiffEq.TSRK("4"); dt = 0.01))
    end

    @testset "Subtypes that need more than a plain problem" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        with_jac = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = decay_jac!), [1.0], (0.0, 1.0),
        )
        for st in ("lassp3p4s2c", "llssp3p4s2c", "ark3"), pr in (prob, with_jac)
            @test_throws ArgumentError SciMLBase.solve(pr, PETScDiffEq.TSRosW(st); dt = 0.1)
        end
        @test_throws ArgumentError SciMLBase.solve(
            prob, PETScDiffEq.TSRosW("assp3p3s1c"; autodiff = PETScDiffEq.AutoFiniteDiff()); dt = 0.1,
        )
        sol = SciMLBase.solve(
            with_jac, PETScDiffEq.TSRosW("assp3p3s1c"); dt = 0.01, reltol = 1.0e-8,
            abstol = 1.0e-10,
        )
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-6
        function pair_jac!(J, u, p, t)
            J .= 0.0
            J[1, 1] = -1.0
            return J[2, 2] = -1.0
        end
        with_mass = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = pair_jac!, mass_matrix = [2.0 0.0; 0.0 1.0]),
            [1.0, 1.0], (0.0, 1.0),
        )
        @test_throws ArgumentError SciMLBase.solve(
            with_mass, PETScDiffEq.TSRosW("assp3p3s1c"); dt = 0.01, adaptive = false,
        )
        @test_throws ArgumentError SciMLBase.solve(
            prob, PETScDiffEq.TSARKIMEX("ars122"); dt = 0.1,
        )
        split = SciMLBase.SplitODEProblem(decay!, decay!, [1.0], (0.0, 1.0))
        sol = SciMLBase.solve(
            split, PETScDiffEq.TSARKIMEX("ars122"); dt = 0.01, adaptive = false,
        )
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test abs(sol.u[end][1] - exp(-2.0)) < 1.0e-4
        @test_throws ArgumentError SciMLBase.solve(
            split, PETScDiffEq.TSARKIMEX("bpr3"); dt = 0.01, adaptive = false,
        )
        sol = SciMLBase.solve(prob, PETScDiffEq.TSARKIMEX("bpr3"); dt = 0.01)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-7
    end

    @testset "Requested subtypes actually take effect" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        exact = exp(-1.0)
        function measured_order(alg)
            errs = Float64[]
            for dt in (0.1, 0.05, 0.025)
                sol = SciMLBase.solve(prob, alg; dt = dt)
                push!(errs, abs(sol.u[end][1] - exact))
            end
            return log2(errs[end - 1] / errs[end])
        end
        @test measured_order(
            PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"]),
        ) > 4.5
        @test measured_order(
            PETScDiffEq.TSRK("3bs", ["-ts_adapt_type", "none"]),
        ) < 3.5
        @test measured_order(
            PETScDiffEq.TSImplicit("theta", 1.0, ["-ts_adapt_type", "none"]),
        ) < 1.5
        @test measured_order(
            PETScDiffEq.TSImplicit("theta", 0.5, ["-ts_adapt_type", "none"]),
        ) > 1.8
    end

    @testset "Reversed time span" begin
        grow!(du, u, p, t) = (du[1] = u[1]; nothing)
        back = SciMLBase.ODEProblem(decay!, [1.0], (1.0, 0.0))
        tight = (dt = 0.1, reltol = 1.0e-10, abstol = 1.0e-12)

        @testset "steps exactly as the forward mirror problem" begin
            # A reversed span runs as v' = v on s = -t, so this mirror matches bit for bit.
            fwd = SciMLBase.ODEProblem(grow!, [1.0], (-1.0, 0.0))
            for (alg, kw) in (
                    (PETScDiffEq.TSRK("5dp"), (reltol = 1.0e-8, abstol = 1.0e-10)),
                    (PETScDiffEq.TSRosW("ra34pw2"), (reltol = 1.0e-8, abstol = 1.0e-10)),
                    (PETScDiffEq.TSImplicit("bdf"), (reltol = 1.0e-8, abstol = 1.0e-10)),
                    (PETScDiffEq.TSARKIMEX("3"), (reltol = 1.0e-8, abstol = 1.0e-10)),
                    (PETScDiffEq.TSRK("3bs"), (adaptive = false,)),
                )
                b = SciMLBase.solve(back, alg; dt = 0.1, kw...)
                f = SciMLBase.solve(fwd, alg; dt = 0.1, kw...)
                @test b.retcode == SciMLBase.ReturnCode.Success
                @test b.t[1] == 1.0
                @test b.t[end] == 0.0
                @test b.t == -f.t
                @test b.u == f.u
            end
        end

        @testset "time zero is +0.0 and a stop there fires" begin
            for span in ((0.0, -1.0), (0.5, -0.5), (1.0, 0.0))
                prob = SciMLBase.ODEProblem(decay!, [1.0], span)
                hits = Float64[]
                cb = PresetTimeCallback([0.0], integ -> push!(hits, integ.t))
                sol = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = cb)
                @test hits == [0.0]
                @test !any(t -> t === -0.0, hits)
                @test !any(t -> t === -0.0, sol.t)
                sol = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, saveat = 0.25)
                @test 0.0 in sol.t
                @test !any(t -> t === -0.0, sol.t)
            end
        end

        @testset "saveat, dense output and tstops" begin
            alg = PETScDiffEq.TSRK("5dp")
            sol = SciMLBase.solve(back, alg; saveat = 0.25, tight...)
            @test sol.t == [1.0, 0.75, 0.5, 0.25, 0.0]
            @test maximum(abs(sol.u[i][1] - exp(1 - sol.t[i])) for i in eachindex(sol.t)) < 1.0e-7
            sol = SciMLBase.solve(back, alg; saveat = [0.2, 0.7], tight...)
            @test sol.t == [0.7, 0.2]
            sol = SciMLBase.solve(back, alg; tight...)
            @test issorted(sol.t; rev = true)
            @test abs(sol(0.5)[1] - exp(0.5)) < 1.0e-6
            sol = SciMLBase.solve(back, alg; tstops = [0.3, 0.6], tight...)
            @test 0.3 in sol.t
            @test 0.6 in sol.t
        end

        @testset "the integrator's clock runs backward" begin
            integ = SciMLBase.init(back, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            @test integ.tdir == -1
            SciMLBase.step!(integ)
            @test integ.t ≈ 0.9
            @test integ.tprev == 1.0
            @test integ.dt < 0
            @test SciMLBase.get_proposed_dt(integ) ≈ -0.1
            @test SciMLBase.get_du(integ) ≈ -integ.u
            @test_throws ArgumentError SciMLBase.add_tstop!(integ, 0.95)
            t1 = integ.t
            @test SciMLBase.savevalues!(integ, true) == (true, true)
            SciMLBase.add_tstop!(integ, 0.45)
            SciMLBase.solve!(integ)
            @test issorted(integ.sol.t; rev = true)
            @test count(==(t1), integ.sol.t) == 2
            @test 0.45 in integ.sol.t
            @test integ.t == 0.0
            @test abs(integ.u[1] - exp(1.0)) < 1.0e-5
        end

        @testset "the running solution keeps up with each step" begin
            fwd = SciMLBase.ODEProblem(grow!, [1.0], (-1.0, 0.0))
            for dense in (true, false)
                b = SciMLBase.init(back, PETScDiffEq.TSRK("5dp"); dense = dense, tight...)
                f = SciMLBase.init(fwd, PETScDiffEq.TSRK("5dp"); dense = dense, tight...)
                for _ in 1:3
                    SciMLBase.step!(b)
                    SciMLBase.step!(f)
                end
                @test length(b.sol.t) == length(b.sol.u) == 4
                @test b.sol.t[end] == b.t
                @test b.sol(b.t) == b.u
                @test b.sol.t == -f.sol.t
                tm = (b.tprev + b.t) / 2
                @test b.sol(tm) == f.sol(-tm)
            end
        end

        @testset "callbacks" begin
            hits = Float64[]
            preset = PresetTimeCallback([0.25, 0.75], integ -> push!(hits, integ.t))
            sol = SciMLBase.solve(back, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = preset)
            @test hits == [0.75, 0.25]
            @test issorted(sol.t; rev = true)
            @test count(==(0.75), sol.t) == 2
            empty!(hits)
            cross = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u[1] - 2.0, integ -> push!(hits, integ.t),
            )
            sol = SciMLBase.solve(back, PETScDiffEq.TSRK("5dp"); callback = cross, tight...)
            @test abs(only(hits) - (1 - log(2.0))) < 1.0e-6
            @test issorted(sol.t; rev = true)
            @test count(==(hits[1]), sol.t) == 2
        end

        @testset "a state changed by initialize is the one PETSc integrates" begin
            setu = SciMLBase.DiscreteCallback(
                (u, t, integ) -> false, integ -> nothing;
                initialize = (c, u, t, integ) -> (integ.u[1] = 2.0; nothing),
            )
            for (span, expected) in (((0.0, 1.0), 2 * exp(-1.0)), ((1.0, 0.0), 2 * exp(1.0)))
                prob = SciMLBase.ODEProblem(decay!, [1.0], span)
                sol = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); callback = setu, tight...)
                @test abs(sol.u[end][1] - expected) < 1.0e-8
                @test sol.t[1] == span[1]
                @test sol.t[2] == span[1]
                @test first.(sol.u[1:2]) == [1.0, 2.0]
            end
            integ = SciMLBase.init(
                SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), PETScDiffEq.TSRK("5dp");
                callback = setu, tight...,
            )
            SciMLBase.solve!(integ)
            first_run = integ.u[1]
            SciMLBase.reinit!(integ)
            SciMLBase.solve!(integ)
            @test integ.u[1] == first_run
            @test abs(first_run - 2 * exp(-1.0)) < 1.0e-8
        end

        @testset "every problem form integrates backward" begin
            kw = (dt = 0.01, reltol = 1.0e-8, abstol = 1.0e-10)
            mass = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(decay!; mass_matrix = fill(2.0, 1, 1)), [1.0], (1.0, 0.0),
            )
            sol = SciMLBase.solve(mass, PETScDiffEq.TSRosW("ra34pw2"); kw...)
            @test abs(sol.u[end][1] - exp(0.5)) < 1.0e-5
            split = SciMLBase.SplitODEProblem(decay!, decay!, [1.0], (1.0, 0.0))
            sol = SciMLBase.solve(split, PETScDiffEq.TSARKIMEX("3"); kw...)
            @test abs(sol.u[end][1] - exp(2.0)) < 1.0e-4
            dae = SciMLBase.DAEProblem(
                (r, du, u, p, t) -> (r .= du .+ u; nothing), [-1.0], [1.0], (1.0, 0.0),
            )
            sol = SciMLBase.solve(dae, PETScDiffEq.TSDAE("bdf"); kw...)
            @test abs(sol.u[end][1] - exp(1.0)) < 1.0e-4
            sine!(du, u, p, t) = (du[1] = sin(t); nothing)
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(sine!, [0.0], (1.0, 0.0)), PETScDiffEq.TSRK("5dp"); kw...,
            )
            @test abs(sol.u[end][1] - (cos(1.0) - 1)) < 1.0e-6
            sine_dae = SciMLBase.DAEProblem(
                (r, du, u, p, t) -> (r[1] = du[1] - sin(t); nothing), [sin(1.0)], [0.0], (1.0, 0.0),
            )
            sol = SciMLBase.solve(sine_dae, PETScDiffEq.TSDAE("bdf"); kw...)
            @test abs(sol.u[end][1] - (cos(1.0) - 1)) < 1.0e-4
            two_rate!(du, u, p, t) = (du[1] = -u[1]; du[2] = -3 * u[2]; nothing)
            mprk = SciMLBase.ODEProblem(two_rate!, [1.0, 1.0], (1.0, 0.0))
            sol = SciMLBase.solve(mprk, PETScDiffEq.TSMPRK([1], "p2"); dt = 0.001)
            @test isapprox(sol.u[end], [exp(1.0), exp(3.0)]; rtol = 1.0e-4)
        end
    end

    Sys.WORD_SIZE == 64 && @testset "times PETSc reads as PETSC_DETERMINE, PETSC_CURRENT or PETSC_UNLIMITED" begin
        # Those are -1, -2 and -3 on PETSc's clock, which runs on -t for a reversed span.
        Success, MaxIters = SciMLBase.ReturnCode.Success, SciMLBase.ReturnCode.MaxIters
        osc!(ddu, du, u, p, t) = (ddu .= .-u; nothing)
        algs(R) = (
            (PETScDiffEq.TSRK("4"), (dt = R(0.1), adaptive = false)),
            (PETScDiffEq.TSRK("3bs"), (;)),
            (PETScDiffEq.TSImplicit("bdf"), (dt = R(0.01),)),
        )
        decaying(R, span) = SciMLBase.ODEProblem(decay!, R[1], R.(span))
        swinging(R, span) = SciMLBase.SecondOrderODEProblem(osc!, R[0], R[1], R.(span))
        swung(span) = [-sin(span[2] - span[1]), cos(span[2] - span[1])]
        # An autonomous problem takes the same steps ten later, where no time is a sentinel.
        same_steps(sol, twin, shift) = length(sol.t) == length(twin.t) &&
            isapprox(sol.t .+ shift, twin.t; rtol = 1.0e-10) &&
            isapprox(first.(sol.u), first.(twin.u); rtol = 1.0e-9)

        @testset "a span ending on one: $R" for R in (Float64, Float32)
            for s in 1:3, span in ((-s - 2, -s), (s + 2, s))
                for (alg, kw) in algs(R)
                    sol = SciMLBase.solve(decaying(R, span), alg; kw...)
                    @test sol.retcode == Success
                    @test sol.t[end] === R(span[2])
                    @test isapprox(sol.u[end][1], exp(span[1] - span[2]); rtol = 0.05)
                end
                for alg in (PETScDiffEq.TSBasicSymplectic(), PETScDiffEq.TSAlpha2())
                    sol = SciMLBase.solve(swinging(R, span), alg; dt = R(0.01))
                    @test sol.retcode == Success
                    @test sol.t[end] === R(span[2])
                    @test isapprox(collect(sol.u[end]), swung(span); atol = 1.0e-3)
                end
            end
        end

        @testset "the steps are those of a span that ends elsewhere" begin
            for s in 1:3, dir in (1, -1), (alg, kw) in algs(Float64)
                span, shift = (-dir * (s + 2.0), -dir * float(s)), -10.0 * dir
                for stops in (Float64[], [(span[1] + span[2]) / 2])
                    sol = SciMLBase.solve(decaying(Float64, span), alg; tstops = stops, kw...)
                    twin = SciMLBase.solve(
                        decaying(Float64, span .+ shift), alg; tstops = stops .+ shift, kw...,
                    )
                    @test same_steps(sol, twin, shift)
                end
            end
        end

        @testset "a tstop or a saveat on one: $R" for R in (Float64, Float32)
            for s in 1:3, dir in (1, -1), (alg, kw) in algs(R)
                span, stop = (-dir * (s + 2), 0), -dir * s
                sol = SciMLBase.solve(decaying(R, span), alg; tstops = R[stop], kw...)
                @test sol.retcode == Success
                @test R(stop) in sol.t
                @test sol.t[end] === R(0)
                if R === Float64
                    shift = -10.0 * dir
                    twin = SciMLBase.solve(
                        decaying(R, span .+ shift), alg; tstops = [stop + shift], kw...,
                    )
                    @test same_steps(sol, twin, shift)
                end
                sol = SciMLBase.solve(decaying(R, span), alg; saveat = R[stop], kw...)
                @test sol.t == R[stop]
                @test isapprox(sol.u[1][1], exp(span[1] - stop); rtol = 0.05)
                saves = R[(span[1] + stop) / 2, stop]
                sol = SciMLBase.solve(decaying(R, (span[1], stop)), alg; saveat = saves, kw...)
                @test sol.retcode == Success
                @test sol.t == saves
                @test isapprox(sol.u[end][1], exp(span[1] - stop); rtol = 0.05)
            end
        end

        @testset "dtmin ends a solve before a tstop on one" begin
            alg = PETScDiffEq.TSRK("3bs")
            for (span, stop, shift) in (((-3.0, 0.0), -1.0, -10.0), ((3.0, 0.0), 1.0, 10.0))
                sol = SciMLBase.solve(
                    decaying(Float64, span), alg; tstops = [stop], dtmin = 0.3,
                )
                twin = SciMLBase.solve(
                    decaying(Float64, span .+ shift), alg; tstops = [stop + shift], dtmin = 0.3,
                )
                @test sol.retcode == twin.retcode == SciMLBase.ReturnCode.DtLessThanMin
                @test same_steps(sol, twin, shift)
            end
        end

        @testset "the integrator steps onto one: $R" for R in (Float64, Float32)
            for s in 1:3, dir in (1, -1), (alg, kw) in algs(R)
                span, stop = (-dir * (s + 2), 0), R(-dir * s)
                integ = SciMLBase.init(decaying(R, span), alg; kw...)
                SciMLBase.step!(integ, R(2dir), true)
                @test integ.t === stop
                @test isapprox(integ.u[1], exp(-2dir); rtol = 0.05)
                @test SciMLBase.solve!(integ).retcode == Success
                integ = SciMLBase.init(decaying(R, span), alg; kw...)
                SciMLBase.add_tstop!(integ, stop)
                while dir * integ.t < dir * stop
                    SciMLBase.step!(integ)
                end
                @test integ.t === stop
                integ = SciMLBase.init(decaying(R, (span[1], stop)), alg; kw...)
                sol = SciMLBase.solve!(integ)
                @test sol.retcode == Success
                @test integ.t === stop
                @test sol.t[end] === stop
            end
        end

        @testset "a span starting on one: $R" for R in (Float64, Float32)
            for s in 1:3, span in ((-s, 1), (s, -1)), (alg, kw) in algs(R)
                sol = SciMLBase.solve(decaying(R, span), alg; kw...)
                @test sol.retcode == Success
                @test sol.t[1] === R(span[1])
                @test sol.t[end] === R(span[2])
            end
        end

        @testset "the adjoint of a span ending on one" begin
            ones!(out, u, p, t, i) = (out .= 1; nothing)
            cn = (PETScDiffEq.TSImplicit("cn"), (dt = 0.05, adaptive = false))
            for s in 1:3, span in ((-s - 2.0, -float(s)), (s + 2.0, float(s)), (-float(s), 1.0))
                for (alg, kw) in (algs(Float64)[1:2]..., cn)
                    du0, _ = PETScDiffEq._discrete_adjoint(
                        decaying(Float64, span), alg, PETScAdjoint();
                        t = [span[2]], dgdu_discrete = ones!, kw...,
                    )
                    sol = SciMLBase.solve(decaying(Float64, span), alg; kw...)
                    @test isapprox(du0[1], sol.u[end][1]; rtol = 1.0e-8)
                    @test isapprox(du0[1], exp(span[1] - span[2]); rtol = 0.05)
                end
            end
        end

        @testset "a negative tolerance or maxiters" begin
            prob, alg = decaying(Float64, (0, 1)), PETScDiffEq.TSRK("3bs")
            for tol in (-1.0, -2.0, -3.0)
                @test_throws ArgumentError SciMLBase.solve(prob, alg; abstol = tol)
                @test_throws ArgumentError SciMLBase.solve(prob, alg; reltol = tol)
                integ = SciMLBase.init(prob, alg)
                @test_throws ArgumentError (integ.opts.abstol = tol)
                @test_throws ArgumentError SciMLBase.set_reltol!(integ, tol)
            end
            for maxiters in (-1, -2, -3), (alg, kw) in algs(Float64)
                sol = SciMLBase.solve(prob, alg; maxiters, kw...)
                @test sol.retcode == MaxIters
                @test sol.t == [0.0]
            end
        end
    end

    @testset "alg_order is the order PETSc registers" begin
        @test SciMLBase.alg_order(PETScDiffEq.TSRK("5dp")) == 5
        @test SciMLBase.alg_order(PETScDiffEq.TSRK("3bs")) == 3
        @test SciMLBase.alg_order(PETScDiffEq.TSRosW("ra34pw2")) == 3
        @test SciMLBase.alg_order(PETScDiffEq.TSRosW("rodaspr")) == 4
        @test SciMLBase.alg_order(PETScDiffEq.TSARKIMEX("1bee")) == 1
        @test SciMLBase.alg_order(PETScDiffEq.TSARKIMEX("5")) == 5
        @test SciMLBase.alg_order(PETScDiffEq.TSMPRK([1], "p3")) == 3
        @test SciMLBase.alg_order(PETScDiffEq.TSIRK(2)) == 4
        @test SciMLBase.alg_order(PETScDiffEq.TSImplicit("beuler")) == 1
        @test SciMLBase.alg_order(PETScDiffEq.TSImplicit("theta", 1.0)) == 1
        @test SciMLBase.alg_order(PETScDiffEq.TSImplicit("theta", 0.5)) == 2
        @test SciMLBase.alg_order(PETScDiffEq.TSImplicit("bdf")) == 2
        @test SciMLBase.alg_order(PETScDiffEq.TSImplicit("bdf"; order = 4)) == 4
        @test SciMLBase.alg_order(PETScDiffEq.TSDAE("bdf"; order = 3)) == 3
        @test_throws ArgumentError SciMLBase.alg_order(PETScDiffEq.TSRK("9zz"))
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        for alg in (
                PETScDiffEq.TSRK("2a", ["-ts_adapt_type", "none"]),
                PETScDiffEq.TSRK("4", ["-ts_adapt_type", "none"]),
                PETScDiffEq.TSRosW("2m", ["-ts_adapt_type", "none"]),
                PETScDiffEq.TSRosW("ra34pw2", ["-ts_adapt_type", "none"]),
                PETScDiffEq.TSARKIMEX("1bee", ["-ts_adapt_type", "none"]),
                PETScDiffEq.TSARKIMEX("2e", ["-ts_adapt_type", "none"]),
            )
            errs = [
                abs(SciMLBase.solve(prob, alg; dt = dt).u[end][1] - exp(-1.0))
                    for dt in (0.04, 0.02)
            ]
            @test isapprox(log2(errs[1] / errs[2]), SciMLBase.alg_order(alg); atol = 0.2)
        end
    end

    @testset "The first step without dt" begin
        sinf!(du, u, p, t) = (du[1] = -u[1] + sin(10t); nothing)
        lv!(du, u, p, t) = (du[1] = 1.5u[1] - u[1] * u[2]; du[2] = -3u[2] + u[1] * u[2]; nothing)
        rk, rosw = PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRosW("ra34pw2")
        tight = (abstol = 1.0e-8, reltol = 1.0e-6)
        loose = (abstol = 1.0e-6, reltol = 1.0e-3)
        forced = SciMLBase.ODEProblem(sinf!, [2.0], (2.0, 0.0))
        lv = SciMLBase.ODEProblem(lv!, [2.0, 0.5], (0.0, 10.0))
        split_prob = SciMLBase.SplitODEProblem(
            (du, u, p, t) -> (du[1] = -u[1]; nothing),
            (du, u, p, t) -> (du[1] = sin(10t); nothing), [2.0], (0.0, 1.0),
        )
        oop = SciMLBase.ODEProblem((u, p, t) -> [-u[1] + sin(10t)], [2.0], (2.0, 0.0))
        for (prob, alg, kw, expected) in (
                (forced, rk, tight, -0.020195969921921846),
                (forced, rosw, tight, -0.001497756427620111),
                (
                    SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), rk,
                    (; loose..., dtmax = 1.0e-3), 0.001,
                ),
                (SciMLBase.ODEProblem(lv!, [1.0, 1.0], (0.0, 10.0)), rk, NamedTuple(), 0.0776084743154256),
                (lv, rk, (abstol = 1.0e-9,), 0.08421578155635664),
                (lv, rk, (reltol = 1.0e-7,), 0.021785462059001403),
                (lv, rosw, loose, 0.016189521278835287),
                (lv, rk, (abstol = [1.0e-8, 1.0e-4], reltol = [1.0e-6, 1.0e-3]), 0.024832977354891702),
                (split_prob, PETScDiffEq.TSARKIMEX("3"), tight, 0.001188153919924332),
                (oop, rk, tight, -0.020195969921921846),
            )
            @test isapprox(SciMLBase.init(prob, alg; kw...).dt, expected; rtol = 1.0e-12)
        end
        lin(k) = (du, u, p, t) -> (du .= k .* u; nothing)
        for (prob, kw, expected) in (
                (SciMLBase.ODEProblem(lin(-1.0e8), [1.0], (0.0, 1.0)), (; loose..., dtmin = 1.0e-3), 0.0010000000000000002),
                (SciMLBase.ODEProblem(lin(-1.0e8), [1.0], (1.0e10, 1.0e10 + 1)), loose, 1.9073486328125004e-6),
                (SciMLBase.ODEProblem(lin(-1.0e20), [1.0], (0.0, 1.0)), loose, 1.0e-6),
                (SciMLBase.ODEProblem(lin(-1.0e-6), [1.0], (0.0, 1.0)), (; loose..., dtmax = 10.0), 1.0),
            )
            @test isapprox(SciMLBase.init(prob, rk; kw...).dt, expected; rtol = 1.0e-12)
        end
        blowup = SciMLBase.ODEProblem((du, u, p, t) -> (du[1] = 1 / u[1]; nothing), [0.0], (0.0, 1.0))
        @test SciMLBase.init(blowup, rk; loose...).dt == 1.0e-6
        nan_after = SciMLBase.ODEProblem(
            (du, u, p, t) -> (du[1] = u[1] < 1 ? NaN : -u[1]; nothing), [1.0], (0.0, 1.0),
        )
        @test SciMLBase.init(nan_after, rk; loose...).dt == 1.0e-6
        res!(r, du, u, p, t) = (r .= du .+ u; nothing)
        for (span, expected) in (((0.0, 100.0), 9.999999999999999e-5), ((100.0, 0.0), -9.999999999999999e-5))
            dae = SciMLBase.DAEProblem(res!, [-1.0], [1.0], span)
            @test isapprox(SciMLBase.init(dae, PETScDiffEq.TSDAE("bdf")).dt, expected; rtol = 1.0e-12)
        end
        mass = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; mass_matrix = fill(2.0, 1, 1)), [1.0], (0.0, 1.0),
        )
        @test SciMLBase.init(mass, rosw).dt == 1.0e-6
        constant!(du, u, p, t) = (du .= 1.0; nothing)
        tiny!(du, u, p, t) = (du[1] = 1.0e-20 * (1 + t); nothing)
        for (prob, alg, kw, expected) in (
                (SciMLBase.ODEProblem(constant!, [1.0], (0.0, 10.0)), rk, loose, 1.0),
                (SciMLBase.ODEProblem(constant!, [1.0], (0.0, 0.5)), rk, loose, 0.5),
                (SciMLBase.ODEProblem(tiny!, [1.0], (0.0, 1.0)), rk, loose, 1.0e-6),
                (SciMLBase.ODEProblem(lv!, [-2.0, 0.5], (0.0, 10.0)), rk, loose, 0.0571251726271507),
                (mass, rosw, (dtmin = 1.0e-3,), 0.0010000000000000002),
                (
                    SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), rk,
                    (; loose..., tstops = [1.0e-3]), 0.001,
                ),
            )
            @test isapprox(SciMLBase.init(prob, alg; kw...).dt, expected; rtol = 1.0e-12)
        end
        osc = SciMLBase.ODEProblem(
            (du, u, p, t) -> (du[1] = u[2]; du[2] = -u[1]; nothing), [0.0, 1.0], (0.0, 10.0),
        )
        for alg in (rk, PETScDiffEq.TSRK("3bs"), rosw)
            @test SciMLBase.init(osc, alg).dt == SciMLBase.init(osc, alg; loose...).dt
            default, explicit = SciMLBase.solve(osc, alg), SciMLBase.solve(osc, alg; loose...)
            @test default.t == explicit.t
            @test default.u == explicit.u
        end
    end

    @testset "Adaptive solves run without dt" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        for alg in (
                PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSRosW("ra34pw2"),
                PETScDiffEq.TSARKIMEX("3"), PETScDiffEq.TSImplicit("bdf"),
            )
            sol = SciMLBase.solve(prob, alg; reltol = 1.0e-8, abstol = 1.0e-10)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test abs(sol.u[end][1] - exp(-1.0)) < 1.0e-5
        end
        zero_start = SciMLBase.ODEProblem(decay!, [0.0, 1.0], (0.0, 1.0))
        sol = SciMLBase.solve(zero_start, PETScDiffEq.TSRK("5dp"); abstol = 0.0, reltol = 1.0e-6)
        @test sol.retcode == SciMLBase.ReturnCode.Success
    end

    @testset "dt, its floor, the warning and isadaptive follow the type an option picks" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        back = SciMLBase.ODEProblem(decay!, [1.0], (1.0, 0.0))
        same(sol, ran) = sol.retcode == ran.retcode && sol.t == ran.t && sol.u == ran.u
        bdf = PETScDiffEq.TSImplicit("bdf")
        to_bdf = PETScDiffEq.TSImplicit("beuler", ["-ts_type", "bdf"])
        to_beuler = PETScDiffEq.TSImplicit("bdf", ["-ts_type", "beuler"])
        to_5dp = PETScDiffEq.TSRK("4", ["-ts_rk_type", "5dp"])
        to_4 = PETScDiffEq.TSRK("5dp", ["-ts_rk_type", "4"])
        for (pr, alg, runs) in (
                (prob, to_bdf, bdf), (back, to_bdf, bdf),
                (prob, PETScDiffEq.TSImplicit("bdf", ["-ts_type", "beuler", "-ts_type", "bdf"]), bdf),
                (prob, PETScDiffEq.TSGeneric("bdf"), bdf),
                (prob, to_5dp, PETScDiffEq.TSRK("5dp")),
                (prob, PETScDiffEq.TSImplicit("beuler", ["-ts_type", "bdf", "-ts_bdf_order", "3"]), PETScDiffEq.TSImplicit("bdf"; order = 3)),
            )
            @test same(SciMLBase.solve(pr, alg), SciMLBase.solve(pr, runs))
            integ, ran = SciMLBase.init(pr, alg), SciMLBase.init(pr, runs)
            @test SciMLBase.isadaptive(integ)
            @test integ.dt == ran.dt
            @test SciMLBase.get_proposed_dt(integ) == SciMLBase.get_proposed_dt(ran)
            SciMLBase.set_proposed_dt!(integ, 0.5)
            SciMLBase.auto_dt_reset!(integ)
            @test integ.dt == ran.dt
            @test same(SciMLBase.solve!(integ), SciMLBase.solve!(ran))
            SciMLBase.reinit!(integ; reset_dt = true)
            SciMLBase.reinit!(ran; reset_dt = true)
            @test integ.dt == ran.dt
            SciMLBase.terminate!(integ)
            SciMLBase.terminate!(ran)
        end
        for alg in (
                to_beuler, to_4, PETScDiffEq.TSGeneric("alpha"),
                PETScDiffEq.TSImplicit("beuler", ["-ts_type", "bdf", "-ts_type", "beuler"]),
            )
            @test_throws "needs `dt`" SciMLBase.solve(prob, alg)
            @test_throws "needs `dt`" SciMLBase.init(prob, alg)
        end
        @test_throws "`beuler` with `adaptive = true`" SciMLBase.solve(prob, to_beuler)
        @test_throws "`rk 4` with `adaptive = true`" SciMLBase.solve(prob, to_4)
        @test_throws "`bdf` with `adaptive = false`" SciMLBase.solve(prob, to_bdf; adaptive = false)
        @test_throws "`-ts_adapt_type none` is set" SciMLBase.solve(
            prob, PETScDiffEq.TSImplicit("beuler", ["-ts_type", "bdf", "-ts_adapt_type", "none"]),
        )
        integ = SciMLBase.init(prob, PETScDiffEq.TSGeneric("bdf", ["-ts_type", "alpha"]); dt = 0.1)
        SciMLBase.step!(integ)
        @test_throws "not known for `alpha`" SciMLBase.auto_dt_reset!(integ)
        @test_throws "not known for `alpha`" SciMLBase.reinit!(integ; reset_dt = true)
        @test integ.t == 0.1
        @test SciMLBase.solve!(integ).retcode == SciMLBase.ReturnCode.Success
        with_tol = (; dt = 0.05, reltol = 1.0e-6)
        for (alg, name) in ((to_beuler, "beuler"), (to_4, "rk 4"), (PETScDiffEq.TSGeneric("beuler"), "beuler"))
            @test_logs (:warn, Regex("`$name` has no embedded error estimate")) match_mode = :any SciMLBase.solve(
                prob, alg; with_tol...,
            )
        end
        for alg in (to_bdf, to_5dp, PETScDiffEq.TSGeneric("alpha"))
            @test_logs min_level = Logging.Warn SciMLBase.solve(prob, alg; with_tol...)
        end
        floored = (; dt = 0.01, dtmin = 0.05)
        for (alg, runs) in (
                (to_beuler, PETScDiffEq.TSImplicit("beuler")), (to_4, PETScDiffEq.TSRK("4")),
                (to_bdf, bdf), (to_5dp, PETScDiffEq.TSRK("5dp")),
            )
            @test same(SciMLBase.solve(prob, alg; floored...), SciMLBase.solve(prob, runs; floored...))
            integ, ran = SciMLBase.init(prob, alg; floored...), SciMLBase.init(prob, runs; floored...)
            @test SciMLBase.isadaptive(integ) == SciMLBase.isadaptive(ran)
            integ.opts.dtmin = ran.opts.dtmin = 0.2
            @test integ.h.ctx.dtmin == ran.h.ctx.dtmin
            @test same(SciMLBase.solve!(integ), SciMLBase.solve!(ran))
        end
        @test SciMLBase.solve(prob, to_beuler; floored...).retcode == SciMLBase.ReturnCode.Success
        @test SciMLBase.solve(prob, to_bdf; floored...).retcode == SciMLBase.ReturnCode.DtLessThanMin
        @test !SciMLBase.isadaptive(SciMLBase.init(prob, to_beuler; dt = 0.01))
        @test !SciMLBase.isadaptive(SciMLBase.init(prob, to_4; dt = 0.01))
    end

    @testset "solve accepts and ignores the storage DiffEqDevTools passes" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        ref = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"))
        sol = SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"), ref.u, ref.t, ref.k)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test sol.t == ref.t
        @test sol.u == ref.u
        for arg in (0.1, [0.0, 0.5, 1.0])
            @test_throws SciMLBase.NoDefaultAlgorithmError SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"), arg,
            )
        end
        resid!(r, du, u, p, t) = (r[1] = du[1] + u[1]; nothing)
        dae = SciMLBase.DAEProblem(resid!, [-1.0], [1.0], (0.0, 1.0))
        ref = SciMLBase.solve(dae, PETScDiffEq.TSDAE("bdf"))
        sol = SciMLBase.solve(dae, PETScDiffEq.TSDAE("bdf"), ref.u, ref.t)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test sol.t == ref.t
        @test sol.u == ref.u
    end

    @testset "Input validation" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        with_jac = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = decay_jac!), [1.0], (0.0, 1.0),
        )
        pair = SciMLBase.ODEProblem(decay!, [1.0, 1.0], (0.0, 1.0))
        for (p, alg) in (
                (prob, PETScDiffEq.TSRK("4")), (prob, PETScDiffEq.TSRK("1fe")),
                (prob, PETScDiffEq.TSRosW("theta1")), (prob, PETScDiffEq.TSARKIMEX("prssp2")),
                (prob, PETScDiffEq.TSARKIMEX("ars443")),
                (prob, PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"])),
                (prob, PETScDiffEq.TSImplicit("beuler")), (with_jac, PETScDiffEq.TSIRK(2)),
                (pair, PETScDiffEq.TSMPRK([1], "p2")), (prob, PETScDiffEq.TSGeneric("alpha")),
            )
            @test_throws ArgumentError SciMLBase.solve(p, alg)
        end
        @test_throws ArgumentError SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); adaptive = false)
        @test SciMLBase.solve(prob, PETScDiffEq.TSRK("4"); dt = 0.01).retcode ==
            SciMLBase.ReturnCode.Success
        @test_throws ArgumentError SciMLBase.solve(
            pair, PETScDiffEq.TSRK("5dp"); abstol = [1.0e-6, 1.0e-6, 1.0e-6],
        )
        @test_throws ArgumentError SciMLBase.solve(
            SciMLBase.ODEProblem(decay!, [1.0], (1.0, 1.0)),
            PETScDiffEq.TSRK("5dp"); dt = 0.1,
        )
    end

    @testset "PETScAdjoint" begin
        function adj_f!(du, u, p, t)
            du[1] = -p[1] * u[1] + p[2] * u[1] * u[2]
            du[2] = p[3] * u[1] - p[4] * u[2]^2 + p[1] * sin(t)
            return nothing
        end
        function adj_jac!(J, u, p, t)
            J[1, 1] = -p[1] + p[2] * u[2]
            J[1, 2] = p[2] * u[1]
            J[2, 1] = p[3]
            J[2, 2] = -2 * p[4] * u[2]
            return nothing
        end
        function adj_paramjac!(pJ, u, p, t)
            fill!(pJ, 0.0)
            pJ[1, 1] = -u[1]
            pJ[1, 2] = u[1] * u[2]
            pJ[2, 1] = sin(t)
            pJ[2, 3] = u[1]
            pJ[2, 4] = -u[2]^2
            return nothing
        end
        adj_f(u, p, t) = (du = similar(u); adj_f!(du, u, p, t); du)
        adj_jac(u, p, t) = (J = zeros(2, 2); adj_jac!(J, u, p, t); J)
        adj_paramjac(u, p, t) = (pJ = zeros(2, 4); adj_paramjac!(pJ, u, p, t); pJ)
        function adj_prob(u0, p, tspan; oop = false, sparse_jac = false)
            f = if oop
                SciMLBase.ODEFunction{false}(adj_f; jac = adj_jac, paramjac = adj_paramjac)
            else
                SciMLBase.ODEFunction{true}(
                    adj_f!; jac = adj_jac!, paramjac = adj_paramjac!,
                    jac_prototype = sparse_jac ? sparse(ones(2, 2)) : nothing,
                )
            end
            return SciMLBase.ODEProblem(f, u0, tspan, p)
        end
        u0, p0 = [1.0, 0.5], [0.7, 0.3, 0.4, 0.2]
        forward_t, backward_t = collect(0.0:0.1:1.0), collect(1.0:-0.1:0.0)
        exact = [
            "-snes_rtol", "1e-13", "-snes_atol", "1e-15", "-ksp_type", "preonly", "-pc_type", "lu",
        ]
        endpoint = [exact; "-ts_theta_endpoint"]
        half_norm(u, p, t) = sum(abs2, u) / 2
        half_norm_du!(out, u, p, t, i) = (out .= u; nothing)
        coupled(u, p, t) = sum(abs2, u) / 2 + p[2] * u[1] * u[2] + p[1]^2 * t
        function coupled_du!(out, u, p, t, i)
            out[1] = u[1] + p[2] * u[2]
            out[2] = u[2] + p[2] * u[1]
            return nothing
        end
        function coupled_dp!(out, u, p, t, i)
            fill!(out, 0.0)
            out[1] = 2 * p[1] * t
            out[2] = u[1] * u[2]
            return nothing
        end
        grad(
            prob, alg; sensealg = PETScAdjoint(), t = forward_t,
            dgdu_discrete = half_norm_du!, kwargs...,
        ) = PETScDiffEq._discrete_adjoint(
            prob, alg, sensealg; t, dgdu_discrete, dt = 0.01, adaptive = false, kwargs...,
        )
        central_differences(loss, θ; h = 1.0e-6) = [
            (loss(θ + h * (eachindex(θ) .== i)) - loss(θ - h * (eachindex(θ) .== i))) / (2h)
                for i in eachindex(θ)
        ]
        relerr(a, b) = norm(a - b) / norm(b)

        @testset "matches finite differences of the same fixed-step solve: $name" for (
                name, alg, tspan, ts, opts,
            ) in (
                ("RK4", TSRK("4"), (0.0, 1.0), forward_t, (;)),
                ("RK4 backward in time", TSRK("4"), (1.0, 0.0), backward_t, (;)),
                ("5dp at a fixed step", TSRK("5dp"), (0.0, 1.0), forward_t, (;)),
                ("backward Euler", TSImplicit("beuler", exact), (0.0, 1.0), forward_t, (;)),
                (
                    "backward Euler backward in time", TSImplicit("beuler", exact),
                    (1.0, 0.0), backward_t, (;),
                ),
                ("Crank-Nicolson", TSImplicit("cn", exact), (0.0, 1.0), forward_t, (;)),
                (
                    "Crank-Nicolson backward in time", TSImplicit("cn", exact),
                    (1.0, 0.0), backward_t, (;),
                ),
                ("theta 0.7", TSImplicit("theta", 0.7, exact), (0.0, 1.0), forward_t, (;)),
                (
                    "theta 0.7 backward in time", TSImplicit("theta", 0.7, exact),
                    (1.0, 0.0), backward_t, (;),
                ),
                (
                    "theta at its default, the implicit midpoint rule", TSImplicit("theta", exact),
                    (0.0, 1.0), forward_t, (;),
                ),
                (
                    "theta 0.7 in its endpoint form", TSImplicit("theta", 0.7, endpoint),
                    (0.0, 1.0), forward_t, (;),
                ),
                (
                    "theta 0.7 in its endpoint form backward in time",
                    TSImplicit("theta", 0.7, endpoint), (1.0, 0.0), backward_t, (coupled = true,),
                ),
                (
                    "a trajectory of states only", TSRK("4"), (0.0, 1.0), forward_t,
                    (sensealg = ["-ts_trajectory_solution_only", "1"],),
                ),
                (
                    "a trajectory of states only, theta 0.7", TSImplicit("theta", 0.7, exact),
                    (0.0, 1.0), forward_t, (sensealg = ["-ts_trajectory_solution_only", "1"],),
                ),
                # PETSc's 32-bit build fails trajectory file I/O intermittently.
                (
                    Sys.WORD_SIZE == 64 ? (
                            (
                                "a trajectory on disk", TSRK("4"), (0.0, 1.0), forward_t,
                                (sensealg = ["-ts_trajectory_type", "basic"],),
                            ),
                        ) : ()
                )...,
                (
                    "a sparse jac_prototype, RK4 backward in time", TSRK("4"), (1.0, 0.0),
                    backward_t, (sparse_jac = true,),
                ),
                (
                    "a sparse jac_prototype, backward Euler", TSImplicit("beuler", exact),
                    (0.0, 1.0), forward_t, (sparse_jac = true,),
                ),
                ("no_start", TSRK("4"), (0.0, 1.0), forward_t, (no_start = true,)),
                ("a cost that depends on p", TSRK("4"), (0.0, 1.0), forward_t, (coupled = true,)),
                (
                    "a cost that depends on p, Crank-Nicolson backward in time",
                    TSImplicit("cn", exact), (1.0, 0.0), backward_t, (coupled = true,),
                ),
                ("out of place", TSRK("4"), (0.0, 1.0), forward_t, (oop = true,)),
                (
                    "out of place, backward Euler backward in time", TSImplicit("beuler", exact),
                    (1.0, 0.0), backward_t, (oop = true,),
                ),
                (
                    "out of place with a cost that depends on p, theta 0.3 backward in time",
                    TSImplicit("theta", 0.3, exact), (1.0, 0.0), backward_t,
                    (oop = true, coupled = true),
                ),
                (
                    "a sparse jac_prototype, theta 0.7", TSImplicit("theta", 0.7, exact),
                    (0.0, 1.0), forward_t, (sparse_jac = true,),
                ),
            )
            prob = adj_prob(
                copy(u0), copy(p0), tspan;
                oop = get(opts, :oop, false), sparse_jac = get(opts, :sparse_jac, false),
            )
            is_coupled = get(opts, :coupled, false)
            no_start = get(opts, :no_start, false)
            # A disk trajectory writes its files to the working directory.
            du0, dp = cd(mktempdir()) do
                grad(
                    prob, alg; t = ts, no_start,
                    sensealg = PETScAdjoint(petsc_options = get(opts, :sensealg, String[])),
                    dgdu_discrete = is_coupled ? coupled_du! : half_norm_du!,
                    dgdp_discrete = is_coupled ? coupled_dp! : nothing,
                )
            end
            cost = is_coupled ? coupled : half_norm
            function loss(θ)
                sol = SciMLBase.solve(
                    adj_prob(θ[1:2], θ[3:6], tspan), alg; dt = 0.01, adaptive = false, saveat = ts,
                )
                return sum(
                    cost(sol.u[i], θ[3:6], sol.t[i]) for i in eachindex(sol.t)
                        if !(no_start && i == 1)
                )
            end
            @test relerr(vcat(du0, vec(dp)), central_differences(loss, vcat(u0, p0))) < 5.0e-9
        end

        @testset "an option PETSc reads while the adjoint runs is in force there" begin
            cd(mktempdir()) do
                view = ["-ts_adjoint_view_solution", "ascii:lambda.txt"]
                grad(
                    adj_prob(copy(u0), copy(p0), (0.0, 1.0)), TSRK("4");
                    sensealg = PETScAdjoint(petsc_options = view),
                )
                @test isfile("lambda.txt")
            end
        end

        @testset "dp is nothing without parameters and empty with no entries" begin
            function g!(du, u, p, t)
                du[1] = -u[1] + 0.3 * u[1] * u[2]
                du[2] = 0.4 * u[1] - 0.2 * u[2]^2 + sin(t)
                return nothing
            end
            function g_jac!(J, u, p, t)
                J[1, 1] = -1 + 0.3 * u[2]
                J[1, 2] = 0.3 * u[1]
                J[2, 1] = 0.4
                J[2, 2] = -0.4 * u[2]
                return nothing
            end
            for p in (nothing, SciMLBase.NullParameters(), Float64[]),
                    alg in (TSRK("4"), TSImplicit("theta", 0.7, exact), TSARKIMEX("3", exact))
                prob = SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(g!; jac = g_jac!), copy(u0), (0.0, 1.0), p,
                )
                du0, dp = grad(prob, alg)
                function loss(u)
                    sol = SciMLBase.solve(
                        SciMLBase.remake(prob; u0 = u), alg;
                        dt = 0.01, adaptive = false, saveat = forward_t,
                    )
                    return sum(half_norm(v, p, 0.0) for v in sol.u)
                end
                @test relerr(du0, central_differences(loss, u0)) < 5.0e-9
                if p isa Vector
                    @test dp == zeros(0)'
                else
                    @test dp === nothing
                end
            end
        end

        @testset "an adaptive solve holds its accepted steps fixed" begin
            prob = adj_prob(copy(u0), copy(p0), (0.0, 1.0))
            tolerances = (abstol = 1.0e-8, reltol = 1.0e-8, dt = 0.01)
            du0, dp = PETScDiffEq._discrete_adjoint(
                prob, TSRK("5dp"), PETScAdjoint();
                t = [0.0, 1.0], dgdu_discrete = half_norm_du!, tolerances...,
            )
            steps = SciMLBase.solve(prob, TSRK("5dp"); tolerances...).t
            stepped(θ) = SciMLBase.solve(
                adj_prob(θ[1:2], θ[3:6], (0.0, 1.0)), TSRK("5dp");
                dt = 1.0, adaptive = false, tstops = steps[2:(end - 1)],
            )
            @test length(steps) > 3
            @test stepped(vcat(u0, p0)).t == steps
            function loss(θ)
                sol = stepped(θ)
                return half_norm(sol.u[1], nothing, 0.0) + half_norm(sol.u[end], nothing, 1.0)
            end
            @test relerr(vcat(du0, vec(dp)), central_differences(loss, vcat(u0, p0))) < 4.0e-10
        end

        # Measured: 8.3e-11 for 5dp, and 1.6e-10 on PETSc 3.22 and 2.5e-10 on 3.25 for ARKIMEX 3.
        @testset "an adaptive solve ends a step on each interior cost time: $name" for (
                name, alg,
            ) in (("5dp", TSRK("5dp")), ("ARKIMEX 3", TSARKIMEX("3", exact)))
            tspan, ts = (0.0, 6.0), [0.7, 1.9, 3.2, 4.4, 6.0]
            prob = adj_prob(copy(u0), copy(p0), tspan)
            tolerances = (abstol = 1.0e-6, reltol = 1.0e-6, dt = 0.01)
            # unstable_check sees every accepted step of the adjoint's forward solve but the last.
            steps = [tspan[1]]
            du0, dp = PETScDiffEq._discrete_adjoint(
                prob, alg, PETScAdjoint();
                t = ts, dgdu_discrete = coupled_du!, dgdp_discrete = coupled_dp!,
                unstable_check = (dt, u, p, t) -> (push!(steps, t); false), tolerances...,
            )
            push!(steps, tspan[2])
            stepped(θ) = SciMLBase.solve(
                adj_prob(θ[1:2], θ[3:6], tspan), alg;
                dt = 6.0, adaptive = false, tstops = steps[2:(end - 1)],
            )
            at = indexin(ts, steps)
            @test length(steps) > 2 * length(ts)
            @test !any(isnothing, at)
            @test stepped(vcat(u0, p0)).t == steps
            function loss(θ)
                sol = stepped(θ)
                return sum(coupled(sol.u[i], θ[3:6], t) for (i, t) in zip(at, ts))
            end
            @test relerr(vcat(du0, vec(dp)), central_differences(loss, vcat(u0, p0))) < 5.0e-9
        end

        function quadrature(θ, alg, tspan, cost; dt = 0.01, kwargs...)
            aug!(dz, z, p, t) = (
                adj_f!(view(dz, 1:2), view(z, 1:2), p, t); dz[3] = cost(view(z, 1:2), p, t); nothing
            )
            sol = SciMLBase.solve(
                SciMLBase.ODEProblem(aug!, vcat(θ[1:2], 0.0), tspan, θ[3:6]), alg;
                dt, adaptive = false, kwargs...,
            )
            return sol.u[end][3]
        end
        integrand_du!(out, u, p, t) = coupled_du!(out, u, p, t, 0)
        integrand_dp!(out, u, p, t) = coupled_dp!(out, u, p, t, 0)
        integral(prob, alg; kwargs...) =
            grad(prob, alg; t = nothing, dgdu_discrete = nothing, kwargs...)

        @testset "an integral cost matches finite differences of the same fixed-step quadrature: $name" for (
                name, alg, tspan, opts,
            ) in (
                ("RK4", TSRK("4"), (0.0, 1.0), (;)),
                ("RK4 backward in time", TSRK("4"), (1.0, 0.0), (;)),
                ("5dp at a fixed step", TSRK("5dp"), (0.0, 1.0), (;)),
                ("backward Euler", TSImplicit("beuler", exact), (0.0, 1.0), (;)),
                (
                    "backward Euler backward in time", TSImplicit("beuler", exact), (1.0, 0.0),
                    (;),
                ),
                (
                    "backward Euler with PETSc's differences",
                    TSImplicit("beuler", exact; autodiff = PETScDiffEq.AutoFiniteDiff()),
                    (0.0, 1.0), (;),
                ),
                ("Crank-Nicolson", TSImplicit("cn", exact), (0.0, 1.0), (;)),
                (
                    "Crank-Nicolson backward in time", TSImplicit("cn", exact), (1.0, 0.0),
                    (;),
                ),
                ("theta 0.7", TSImplicit("theta", 0.7, exact), (0.0, 1.0), (;)),
                ("theta 0.7 backward in time", TSImplicit("theta", 0.7, exact), (1.0, 0.0), (;)),
                (
                    "theta at its default, the implicit midpoint rule", TSImplicit("theta", exact),
                    (0.0, 1.0), (;),
                ),
                ("theta 0.7 in its endpoint form", TSImplicit("theta", 0.7, endpoint), (0.0, 1.0), (;)),
                (
                    "theta 0.7 in its endpoint form backward in time",
                    TSImplicit("theta", 0.7, endpoint), (1.0, 0.0), (discrete = true,),
                ),
                (
                    "theta 0.7 with PETSc's differences",
                    TSImplicit("theta", 0.7, exact; autodiff = PETScDiffEq.AutoFiniteDiff()),
                    (0.0, 1.0), (;),
                ),
                ("out of place", TSRK("4"), (0.0, 1.0), (oop = true,)),
                (
                    "a trajectory of states only", TSRK("4"), (0.0, 1.0),
                    (sensealg = ["-ts_trajectory_solution_only", "1"],),
                ),
                ("g differentiated for both", TSImplicit("cn", exact), (0.0, 1.0), (ad = true,)),
                (
                    "g differentiated for both, theta 0.7 backward in time",
                    TSImplicit("theta", 0.7, exact), (1.0, 0.0), (ad = true,),
                ),
                ("dgdu_continuous alone takes dgdp as zero", TSRK("4"), (0.0, 1.0), (frozen = true,)),
                ("with discrete costs", TSRK("4"), (0.0, 1.0), (discrete = true,)),
                (
                    "with discrete costs, Crank-Nicolson backward in time",
                    TSImplicit("cn", exact), (1.0, 0.0), (discrete = true,),
                ),
            )
            prob = adj_prob(copy(u0), copy(p0), tspan; oop = get(opts, :oop, false))
            ts = tspan[1] < tspan[2] ? forward_t : backward_t
            frozen = get(opts, :frozen, false)
            discrete = get(opts, :discrete, false)
            costs = get(opts, :ad, false) ? (g = coupled,) :
                frozen ? (dgdu_continuous = integrand_du!,) :
                (g = coupled, dgdu_continuous = integrand_du!, dgdp_continuous = integrand_dp!)
            du0, dp = grad(
                prob, alg; t = discrete ? ts : nothing,
                dgdu_discrete = discrete ? half_norm_du! : nothing,
                sensealg = PETScAdjoint(petsc_options = get(opts, :sensealg, String[])),
                costs...,
            )
            cost = frozen ? (u, p, t) -> coupled(u, p0, t) : coupled
            function loss(θ)
                summed = quadrature(θ, alg, tspan, cost)
                discrete || return summed
                sol = SciMLBase.solve(
                    adj_prob(θ[1:2], θ[3:6], tspan), alg; dt = 0.01, adaptive = false, saveat = ts,
                )
                return summed + sum(half_norm(v, nothing, 0.0) for v in sol.u)
            end
            @test relerr(vcat(du0, vec(dp)), central_differences(loss, vcat(u0, p0))) < 5.0e-9
        end

        @testset "g is differentiated where its derivatives are not given" begin
            for alg in (
                    TSRK("4"), TSImplicit("beuler", exact), TSImplicit("cn", exact),
                    TSImplicit("theta", 0.7, exact),
                )
                prob = adj_prob(copy(u0), copy(p0), (0.0, 1.0))
                given = integral(
                    prob, alg; g = coupled, dgdu_continuous = integrand_du!,
                    dgdp_continuous = integrand_dp!,
                )
                only_g = integral(prob, alg; g = coupled)
                plain = integral(
                    SciMLBase.ODEProblem(adj_f!, copy(u0), (0.0, 1.0), copy(p0)), alg; g = coupled,
                )
                for r in (only_g, plain)
                    @test r[1] ≈ given[1] rtol = 1.0e-12
                    @test r[2] ≈ given[2] rtol = 1.0e-12
                end
            end
        end

        @testset "discrete and integral costs add" begin
            prob = adj_prob(copy(u0), copy(p0), (0.0, 1.0))
            discrete = grad(prob, TSRK("4"))
            summed = integral(prob, TSRK("4"); g = coupled)
            both = grad(prob, TSRK("4"); g = coupled)
            @test both[1] ≈ discrete[1] + summed[1] rtol = 1.0e-13
            @test both[2] ≈ discrete[2] + summed[2] rtol = 1.0e-13
        end

        @testset "an adaptive solve with an integral cost holds its accepted steps fixed" begin
            prob = adj_prob(copy(u0), copy(p0), (0.0, 1.0))
            tolerances = (abstol = 1.0e-8, reltol = 1.0e-8, dt = 0.01)
            du0, dp = PETScDiffEq._discrete_adjoint(
                prob, TSRK("5dp"), PETScAdjoint(); g = coupled, tolerances...,
            )
            steps = SciMLBase.solve(prob, TSRK("5dp"); tolerances...).t
            loss(θ) = quadrature(
                θ, TSRK("5dp"), (0.0, 1.0), coupled; dt = 1.0, tstops = steps[2:(end - 1)],
            )
            @test length(steps) > 3
            @test relerr(vcat(du0, vec(dp)), central_differences(loss, vcat(u0, p0))) < 5.0e-10
        end

        function implicit_part!(du, u, p, t)
            du[1] = -p[1] * u[1]
            du[2] = -p[4] * u[2]^2 + p[1] * sin(t)
            return nothing
        end
        function explicit_part!(du, u, p, t)
            du[1] = p[2] * u[1] * u[2]
            du[2] = p[3] * u[1]
            return nothing
        end
        function implicit_jac!(J, u, p, t)
            J[1, 1] = -p[1]
            J[1, 2] = 0.0
            J[2, 1] = 0.0
            J[2, 2] = -2 * p[4] * u[2]
            return nothing
        end
        function explicit_jac!(J, u, p, t)
            J[1, 1] = p[2] * u[2]
            J[1, 2] = p[2] * u[1]
            J[2, 1] = p[3]
            J[2, 2] = 0.0
            return nothing
        end
        function implicit_paramjac!(pJ, u, p, t)
            fill!(pJ, 0.0)
            pJ[1, 1] = -u[1]
            pJ[2, 1] = sin(t)
            pJ[2, 4] = -u[2]^2
            return nothing
        end
        function explicit_paramjac!(pJ, u, p, t)
            fill!(pJ, 0.0)
            pJ[1, 2] = u[1] * u[2]
            pJ[2, 3] = u[1]
            return nothing
        end
        function split_prob(
                u0, p, tspan; kind = :given, f2_jac = explicit_jac!,
                f2_paramjac = explicit_paramjac!,
            )
            kind === :plain &&
                return SciMLBase.SplitODEProblem(implicit_part!, explicit_part!, u0, tspan, p)
            if kind === :oop
                part(f!) = function (u, p, t)
                    du = similar(u, promote_type(eltype(u), eltype(p)))
                    f!(du, u, p, t)
                    return du
                end
                return SciMLBase.SplitODEProblem{false}(
                    part(implicit_part!), part(explicit_part!), u0, tspan, p,
                )
            end
            jac_prototype = kind === :sparse ? sparse(ones(2, 2)) : nothing
            return SciMLBase.SplitODEProblem(
                SciMLBase.ODEFunction(
                    implicit_part!; jac = implicit_jac!, paramjac = implicit_paramjac!,
                    jac_prototype,
                ),
                SciMLBase.ODEFunction(
                    explicit_part!; jac = f2_jac, paramjac = f2_paramjac, jac_prototype,
                ),
                u0, tspan, p,
            )
        end
        states_only = ["-ts_trajectory_solution_only", "1"]

        @testset "TSARKIMEX matches finite differences of the same fixed-step solve: $name" for (
                name, subtype, tspan, ts, opts,
            ) in (
                ("3", "3", (0.0, 1.0), forward_t, (;)),
                ("3 backward from 0", "3", (0.0, -1.0), -forward_t, (;)),
                (
                    "4 with a sparse jac_prototype and a cost that depends on p", "4",
                    (0.0, 1.0), forward_t, (sparse_jac = true, coupled = true),
                ),
                ("2e out of place", "2e", (0.0, 1.0), forward_t, (oop = true,)),
                ("3 from t = 1", "3", (1.0, 2.0), forward_t .+ 1, (;)),
                ("4 backward in time", "4", (1.0, 0.0), backward_t, (;)),
                (
                    "3 backward in time with a trajectory of states only", "3", (1.0, 0.0),
                    backward_t, (sensealg = states_only,),
                ),
                ("l2 backward in time", "l2", (1.0, 0.0), backward_t, (;)),
                (
                    "1bee backward in time with a trajectory of states only", "1bee",
                    (1.0, 0.0), backward_t, (sensealg = states_only,),
                ),
                ("prssp2 backward in time with no_start", "prssp2", (1.0, 0.0), backward_t, (no_start = true,)),
                ("a split problem, 3", "3", (0.0, 1.0), forward_t, (split = :given,)),
                (
                    "a split problem, 3 backward in time with a cost that depends on p", "3",
                    (1.0, 0.0), backward_t, (split = :given, coupled = true),
                ),
                (
                    "a split problem, 4 backward in time, differentiated", "4", (1.0, 0.0),
                    backward_t, (split = :plain,),
                ),
                (
                    "a split problem, ars122 out of place", "ars122", (0.0, 1.0), forward_t,
                    (split = :oop,),
                ),
                (
                    "a split problem, 2e backward in time with sparse jac_prototypes", "2e",
                    (1.0, 0.0), backward_t, (split = :sparse,),
                ),
                (
                    "a split problem, 3 backward in time with a trajectory of states only", "3",
                    (1.0, 0.0), backward_t, (split = :given, sensealg = states_only),
                ),
                # PETSc's 32-bit build fails trajectory file I/O intermittently.
                (
                    Sys.WORD_SIZE == 64 ? (
                            (
                                "a split problem, 3 with a trajectory on disk", "3", (0.0, 1.0),
                                forward_t,
                                (split = :given, sensealg = ["-ts_trajectory_type", "basic"]),
                            ),
                        ) : ()
                )...,
            )
            alg = TSARKIMEX(subtype, exact)
            kind = get(opts, :split, nothing)
            make(u, p; kw...) = kind === nothing ? adj_prob(u, p, tspan; kw...) :
                split_prob(u, p, tspan; kw...)
            prob = kind === nothing ?
                make(
                    copy(u0), copy(p0);
                    oop = get(opts, :oop, false), sparse_jac = get(opts, :sparse_jac, false),
                ) : make(copy(u0), copy(p0); kind)
            is_coupled = get(opts, :coupled, false)
            no_start = get(opts, :no_start, false)
            du0, dp = cd(mktempdir()) do
                grad(
                    prob, alg; t = ts, no_start,
                    sensealg = PETScAdjoint(petsc_options = get(opts, :sensealg, String[])),
                    dgdu_discrete = is_coupled ? coupled_du! : half_norm_du!,
                    dgdp_discrete = is_coupled ? coupled_dp! : nothing,
                )
            end
            cost = is_coupled ? coupled : half_norm
            function loss(θ)
                sol = SciMLBase.solve(
                    make(θ[1:2], θ[3:6]), alg; dt = 0.01, adaptive = false, saveat = ts,
                )
                return sum(
                    cost(sol.u[i], θ[3:6], sol.t[i]) for i in eachindex(sol.t)
                        if !(no_start && i == 1)
                )
            end
            @test relerr(vcat(du0, vec(dp)), central_differences(loss, vcat(u0, p0))) < 1.0e-8
        end

        @testset "TSARKIMEX on a split problem reads f2's jac and paramjac" begin
            alg = TSARKIMEX("3", exact)
            for (tspan, t) in (((0.0, 1.0), forward_t), ((1.0, 0.0), backward_t))
                given = grad(split_prob(copy(u0), copy(p0), tspan), alg; t)
                for kind in (:plain, :oop, :sparse)
                    r = grad(split_prob(copy(u0), copy(p0), tspan; kind), alg; t)
                    @test r[1] ≈ given[1] rtol = 1.0e-12
                    @test r[2] ≈ given[2] rtol = 1.0e-12
                end
            end
            given = grad(split_prob(copy(u0), copy(p0), (0.0, 1.0)), alg)
            doubled(f!) = (out, args...) -> (f!(out, args...); out .*= 2; nothing)
            for wrong in (
                    (f2_jac = doubled(explicit_jac!),), (f2_paramjac = doubled(explicit_paramjac!),),
                )
                r = grad(split_prob(copy(u0), copy(p0), (0.0, 1.0); wrong...), alg)
                @test relerr(vcat(r[1], vec(r[2])), vcat(given[1], vec(given[2]))) > 0.2
            end
            differenced = TSARKIMEX("3", exact; autodiff = PETScDiffEq.AutoFiniteDiff())
            @test grad(split_prob(copy(u0), copy(p0), (0.0, 1.0)), differenced) == given
            thrower(key) = (args...) -> throw(KeyError(key))
            for key in (:f2_jac, :f2_paramjac)
                @test_throws KeyError(key) grad(
                    split_prob(copy(u0), copy(p0), (0.0, 1.0); NamedTuple{(key,)}((thrower(key),))...),
                    alg,
                )
            end
            @test grad(split_prob(copy(u0), copy(p0), (0.0, 1.0)), alg) == given
            unsplit = grad(adj_prob(copy(u0), copy(p0), (0.0, 1.0)), alg)
            @test grad(adj_prob(copy(u0), copy(p0), (0.0, 1.0)), TSGeneric("arkimex", exact)) == unsplit
            none = SciMLBase.SplitODEProblem(
                (du, u, p, t) -> (du .= -u; nothing), (du, u, p, t) -> (du .= 0.3 .* u .^ 2; nothing),
                copy(u0), (0.0, 1.0),
            )
            du0, dp = grad(none, alg)
            @test dp === nothing
            function loss(u)
                sol = SciMLBase.solve(
                    SciMLBase.remake(none; u0 = u), alg; dt = 0.01, adaptive = false,
                    saveat = forward_t,
                )
                return sum(half_norm(v, nothing, 0.0) for v in sol.u)
            end
            @test relerr(du0, central_differences(loss, u0)) < 1.0e-8
        end

        @testset "an adaptive TSARKIMEX holds its accepted steps fixed: $name" for (
                name, subtype, make, tspan,
            ) in (
                ("3", "3", adj_prob, (0.0, 1.0)),
                ("4 backward in time", "4", adj_prob, (1.0, 0.0)),
                ("a split problem, 3", "3", split_prob, (0.0, 1.0)),
                ("a split problem, 4 backward in time", "4", split_prob, (1.0, 0.0)),
            )
            alg = TSARKIMEX(subtype, exact)
            prob = make(copy(u0), copy(p0), tspan)
            tolerances = (abstol = 1.0e-8, reltol = 1.0e-8, dt = 0.01)
            du0, dp = PETScDiffEq._discrete_adjoint(
                prob, alg, PETScAdjoint();
                t = collect(tspan), dgdu_discrete = half_norm_du!, tolerances...,
            )
            steps = SciMLBase.solve(prob, alg; tolerances...).t
            stepped(θ) = SciMLBase.solve(
                make(θ[1:2], θ[3:6], tspan), alg;
                dt = 1.0, adaptive = false, tstops = steps[2:(end - 1)],
            )
            @test length(steps) > 3
            @test stepped(vcat(u0, p0)).t == steps
            function loss(θ)
                sol = stepped(θ)
                return half_norm(sol.u[1], nothing, 0.0) + half_norm(sol.u[end], nothing, 1.0)
            end
            @test relerr(vcat(du0, vec(dp)), central_differences(loss, vcat(u0, p0))) < 5.0e-9
        end

        @testset "inputs are left alone and a repeated call gives the same numbers" begin
            alg = TSImplicit("cn", copy(exact))
            options = ["-ts_trajectory_solution_only", "1"]
            sensealg = PETScAdjoint(petsc_options = copy(options))
            prob = adj_prob(copy(u0), copy(p0), (1.0, 0.0))
            ts = copy(backward_t)
            first_call = grad(prob, alg; sensealg, t = ts)
            @test prob.u0 == u0
            @test prob.p == p0
            @test ts == backward_t
            @test alg.petsc_options == exact
            @test sensealg.petsc_options == options
            @test grad(prob, alg; sensealg, t = ts) == first_call
        end

        @testset "saving keywords from the call or the problem do not reach the adjoint" begin
            prob = adj_prob(copy(u0), copy(p0), (0.0, 1.0))
            reference = grad(prob, TSRK("4"))
            saving = SciMLBase.remake(prob; saveat = 0.05, dense = true, sensealg = PETScAdjoint())
            result = @test_logs min_level = Logging.Warn grad(
                saving, TSRK("4");
                save_everystep = true, save_start = false, save_end = false, saveat = [0.3],
                extra_options = String[],
            )
            @test result == reference
        end

        @testset "an empty callback set is no callback" begin
            prob = adj_prob(copy(u0), copy(p0), (0.0, 1.0))
            reference = grad(prob, TSRK("4"))
            none = SciMLBase.CallbackSet()
            @test grad(prob, TSRK("4"); callback = none) == reference
            @test grad(SciMLBase.remake(prob; callback = none), TSRK("4")) == reference
        end

        @testset "cost times on the grid are found after many steps" begin
            decay!(du, u, p, t) = (du[1] = -p[1] * u[1]; nothing)
            prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    decay!; jac = (J, u, p, t) -> (J[1, 1] = -p[1]; nothing),
                    paramjac = (pJ, u, p, t) -> (pJ[1, 1] = -u[1]; nothing),
                ),
                [1.0], (0.0, 32.0), [0.05],
            )
            dt, ts = 0.001, collect(0.0:1.0:32.0)
            du0, dp = PETScDiffEq._discrete_adjoint(
                prob, TSRK("1fe"),
                PETScAdjoint(petsc_options = ["-ts_trajectory_solution_only", "1"]);
                t = ts, dgdu_discrete = half_norm_du!, dt, adaptive = false,
            )
            a, k = 1 - 0.05 * dt, round.(Int, ts ./ dt)
            expected_du0 = sum(a .^ (2k))
            expected_dp = -sum(k .* dt .* a .^ (2k .- 1))
            @test abs(du0[1] - expected_du0) / expected_du0 < 2.0e-12
            @test abs(dp[1] - expected_dp) / abs(expected_dp) < 2.0e-12
        end

        @testset "an empty state with a sparse jac_prototype" begin
            prob = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(
                    (du, u, p, t) -> nothing; jac = (J, u, p, t) -> nothing,
                    jac_prototype = spzeros(0, 0),
                ),
                Float64[], (0.0, 1.0),
            )
            @test grad(prob, TSRK("4"); t = [0.0, 1.0]) == (Float64[], nothing)
        end

        @testset "without jac or paramjac both are differentiated" begin
            for (tspan, t) in (((0.0, 1.0), forward_t), ((1.0, 0.0), backward_t)),
                    alg in (
                        TSRK("4"), TSImplicit("beuler", exact), TSImplicit("cn", exact),
                        TSImplicit("theta", 0.7, exact), TSARKIMEX("l2", exact),
                    )
                given = grad(adj_prob(copy(u0), copy(p0), tspan), alg; t)
                plain = grad(SciMLBase.ODEProblem(adj_f!, copy(u0), tspan, copy(p0)), alg; t)
                @test plain[1] ≈ given[1] rtol = 1.0e-12
                @test plain[2] ≈ given[2] rtol = 1.0e-12
            end
            given = grad(adj_prob(copy(u0), copy(p0), (0.0, 1.0)), TSRK("4"))
            viewed = grad(
                SciMLBase.ODEProblem(adj_f!, copy(u0), (0.0, 1.0), view([0.0; p0], 2:5)),
                TSRK("4"),
            )
            @test viewed[2] ≈ given[2] rtol = 1.0e-12
            decay!(du, u, p, t) = (du .= -p[1] .* u .+ p[2]; nothing)
            eight = SciMLBase.ODEProblem(decay!, ones(8), (0.0, 1.0), [1.0, 0.5])
            chunked = grad(
                eight, TSImplicit(
                    "beuler", exact; autodiff = PETScDiffEq.AutoForwardDiff(; chunksize = 8),
                ),
            )
            @test chunked[2] ≈ grad(eight, TSImplicit("beuler", exact))[2] rtol = 1.0e-12
        end

        @testset "a user exception reaches the caller and leaves nothing behind" begin
            live() = count(h -> !h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
            before = live()
            prob = adj_prob(copy(u0), copy(p0), (0.0, 1.0))
            reference = grad(prob, TSRK("4"))
            thrower(key) = (args...) -> throw(KeyError(key))
            @test_throws KeyError(:dgdu) grad(prob, TSRK("4"); dgdu_discrete = thrower(:dgdu))
            with(; jac = adj_jac!, paramjac = adj_paramjac!) = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(adj_f!; jac, paramjac), copy(u0), (0.0, 1.0), copy(p0),
            )
            for alg in (TSRK("4"), TSImplicit("beuler", exact))
                @test_throws KeyError(:paramjac) grad(with(paramjac = thrower(:paramjac)), alg)
            end
            @test_throws KeyError(:jac) grad(with(jac = thrower(:jac)), TSRK("4"))
            for (key, cost) in (
                    (:g, (g = thrower(:g),)),
                    (:gu, (dgdu_continuous = thrower(:gu),)),
                    (:gp, (dgdu_continuous = integrand_du!, dgdp_continuous = thrower(:gp))),
                )
                @test_throws KeyError(key) integral(prob, TSImplicit("cn", exact); cost...)
            end
            @test grad(prob, TSRK("4")) == reference
            @test live() <= before
        end

        @testset "a Float32 problem is differentiated in the double build" begin
            plain(u, p, tspan) = SciMLBase.ODEProblem(adj_f!, u, tspan, p)
            u32, p32 = Float32.(u0), Float32.(p0)
            for make in (adj_prob, plain), alg in (TSRK("4"), TSImplicit("cn", exact))
                single = grad(make(u32, p32, (0.0f0, 1.0f0)), alg)
                double = grad(make(Float64.(u32), Float64.(p32), (0.0, 1.0)), alg)
                @test single[1] isa Vector{Float32}
                @test eltype(single[2]) === Float32
                @test single[1] == Float32.(double[1])
                @test single[2] == Float32.(double[2])
                mixed = grad(make(u32, Float64.(p32), (0.0, 1.0)), alg)
                @test mixed[1] isa Vector{Float32}
                @test mixed[2] == double[2]
            end
            for alg in (TSRK("4"), TSImplicit("cn", exact))
                prob = adj_prob(u32, p32, (0.0f0, 1.0f0))
                sol = SciMLBase.solve(prob, alg; dt = 0.01f0, adaptive = false)
                for t in (sol.t, sol.t[1:10:end])
                    single = grad(prob, alg; t, dt = 0.01f0)
                    double = grad(
                        adj_prob(Float64.(u32), Float64.(p32), (0.0, 1.0)), alg;
                        t = collect(range(0.0, 1.0; length = length(t))),
                    )
                    @test relerr(single[1], double[1]) < 5.0e-7
                    @test relerr(vec(single[2]), vec(double[2])) < 5.0e-7
                end
            end
            single = integral(adj_prob(u32, p32, (0.0f0, 1.0f0)), TSRK("4"); g = coupled, dt = 0.01f0)
            double = integral(adj_prob(Float64.(u32), Float64.(p32), (0.0, 1.0)), TSRK("4"); g = coupled)
            @test single[1] isa Vector{Float32}
            @test eltype(single[2]) === Float32
            @test relerr(single[1], double[1]) < 5.0e-7
            @test relerr(vec(single[2]), vec(double[2])) < 5.0e-7
        end

        @testset "a partitioned problem runs on its flat [v; u]: $name" for (name, alg) in (
                ("RK4", TSRK("4")), ("backward Euler", TSImplicit("beuler", exact)),
                ("Crank-Nicolson", TSImplicit("cn", exact)),
                ("theta 0.7", TSImplicit("theta", 0.7, exact)),
            )
            kick!(dv, v, u, p, t) = (
                dv[1] = -p[1] * u[1] - p[2] * v[1] + u[2] * v[2];
                dv[2] = -p[3] * sin(u[2]) + p[1] * cos(t) * u[1]; nothing
            )
            drift!(du, v, u, p, t) = (du[1] = v[1]; du[2] = v[2] + p[4] * u[1]; nothing)
            velocity!(du, v, u, p, t) = (du .= v; nothing)
            kick(v, u, p, t) =
                [-p[1] * u[1] - p[2] * v[1] + u[2] * v[2], -p[3] * sin(u[2]) + p[1] * cos(t) * u[1]]
            drift(v, u, p, t) = [v[1], v[2] + p[4] * u[1]]
            function joined!(dx, x, p, t, second = false)
                v, u = view(x, 1:2), view(x, 3:4)
                kick!(view(dx, 1:2), v, u, p, t)
                (second ? velocity! : drift!)(view(dx, 3:4), v, u, p, t)
                return nothing
            end
            # The flat system's Jacobians, in the order of [v; u].
            function joined_jac!(J, x, p, t)
                fill!(J, 0.0)
                J[1, 1], J[1, 2], J[1, 3], J[1, 4] = -p[2], x[4], -p[1], x[2]
                J[2, 3], J[2, 4] = p[1] * cos(t), -p[3] * cos(x[4])
                J[3, 1], J[4, 2], J[4, 3] = 1.0, 1.0, p[4]
                return nothing
            end
            function joined_paramjac!(pJ, x, p, t)
                fill!(pJ, 0.0)
                pJ[1, 1], pJ[1, 2] = -x[3], -x[1]
                pJ[2, 1], pJ[2, 3] = cos(t) * x[3], -sin(x[4])
                pJ[4, 4] = x[3]
                return nothing
            end
            v0 = [0.3, -0.2]
            θ0 = vcat(v0, u0, p0)
            function make(θ, tspan; kind = :dynamical, given = false)
                v, u, p = θ[1:2], θ[3:4], θ[5:8]
                kind === :flat && return SciMLBase.ODEProblem(joined!, vcat(v, u), tspan, p)
                kind === :oop && return SciMLBase.DynamicalODEProblem(kick, drift, v, u, tspan, p)
                kind === :second && return SciMLBase.SecondOrderODEProblem(kick!, v, u, tspan, p)
                kind === :flat_second && return SciMLBase.ODEProblem(
                    (dx, x, p, t) -> joined!(dx, x, p, t, true), vcat(v, u), tspan, p,
                )
                given || return SciMLBase.DynamicalODEProblem(kick!, drift!, v, u, tspan, p)
                f = SciMLBase.DynamicalODEFunction{true}(
                    kick!, drift!; jac = joined_jac!, paramjac = joined_paramjac!,
                )
                return SciMLBase.DynamicalODEProblem(f, v, u, tspan, p)
            end
            for (tspan, t) in (((0.0, 1.0), forward_t), ((1.0, 0.0), backward_t))
                flat = grad(make(θ0, tspan; kind = :flat), alg; t)
                for (kind, given) in ((:dynamical, false), (:dynamical, true), (:oop, false))
                    prob = make(θ0, tspan; kind, given)
                    du0, dp = grad(prob, alg; t)
                    @test du0 isa typeof(prob.u0)
                    @test collect(du0) ≈ flat[1] rtol = 1.0e-12
                    @test dp ≈ flat[2] rtol = 1.0e-12
                end
                du0, dp = grad(make(θ0, tspan), alg; t)
                function loss(θ)
                    sol = SciMLBase.solve(
                        make(θ, tspan), alg; dt = 0.01, adaptive = false, saveat = t,
                    )
                    @test all(u -> u isa typeof(sol.prob.u0), sol.u)
                    return sum(half_norm(u, nothing, 0.0) for u in sol.u)
                end
                @test relerr(vcat(collect(du0), vec(dp)), central_differences(loss, θ0)) < 5.0e-9
                second = grad(make(θ0, tspan; kind = :second), alg; t)
                flat_second = grad(make(θ0, tspan; kind = :flat_second), alg; t)
                @test collect(second[1]) ≈ flat_second[1] rtol = 1.0e-12
                @test second[2] ≈ flat_second[2] rtol = 1.0e-12
            end
            # Costs see the state as an ArrayPartition and write their derivative into one.
            seen(cost, parts) =
                (args...) -> (@test all(a -> hasproperty(a, :x), args[parts]); cost(args...))
            for costs in (
                    (dgdu_discrete = coupled_du!, dgdp_discrete = coupled_dp!),
                    (t = nothing, dgdu_discrete = nothing, g = coupled),
                    (
                        t = nothing, dgdu_discrete = nothing, dgdu_continuous = integrand_du!,
                        dgdp_continuous = integrand_dp!,
                    ),
                )
                flat = grad(make(θ0, (0.0, 1.0); kind = :flat), alg; costs...)
                wrapped = map(keys(costs), values(costs)) do k, c
                    c === nothing || k === :t ? c : k === :g ? seen(c, 1:1) :
                        seen(c, k in (:dgdu_discrete, :dgdu_continuous) ? (1:2) : (2:2))
                end
                du0, dp = grad(
                    make(θ0, (0.0, 1.0); given = true), alg; NamedTuple{keys(costs)}(wrapped)...,
                )
                @test collect(du0) ≈ flat[1] rtol = 1.0e-12
                @test dp ≈ flat[2] rtol = 1.0e-12
            end
            single = grad(make(Float32.(θ0), (0.0f0, 1.0f0)), alg)
            double = grad(make(Float64.(Float32.(θ0)), (0.0, 1.0)), alg)
            @test single[1] isa typeof(make(Float32.(θ0), (0.0f0, 1.0f0)).u0)
            @test collect(single[1]) == Float32.(collect(double[1]))
            @test single[2] == Float32.(double[2])
        end

        @testset "a solve with a dm" begin
            PETSc, LibPETSc = PETScDiffEq.PETSc, PETScDiffEq.LibPETSc
            grid(u, da) = PETScDiffEq.reshape_local_array(u, da)
            pl = PETSc.getlib(; PetscScalar = Float64)
            N = 15
            dx = 1 / (N + 1)
            da = PETSc.DMDA(pl, MPI.COMM_SELF, (LibPETSc.DM_BOUNDARY_GHOSTED,), (N,), 1, 1)
            source(i) = i * dx * (1 - i * dx)
            function rd_dm!(du, u, p, t)
                U, D = grid(u, da), grid(du, da)
                for i in 1:N
                    D[1, i] = p[1] * ((U[1, i - 1] - 2U[1, i] + U[1, i + 1]) / dx^2) +
                        p[2] * source(i) - p[3] * U[1, i]^3
                end
                return nothing
            end
            function rd_jac_dm!(J, u, p, t)
                U = grid(u, da)
                for i in 1:N
                    row = (p[1] / dx^2, -2p[1] / dx^2 - 3p[3] * U[1, i]^2, p[1] / dx^2)
                    set_stencil_values!(J, (1, i), ((1, i - 1), (1, i), (1, i + 1)), row)
                end
                return nothing
            end
            function rd_paramjac_dm!(pJ, u, p, t)
                U = grid(u, da)
                L, S, C = (grid(view(pJ, :, k), da) for k in 1:3)
                for i in 1:N
                    L[1, i] = (U[1, i - 1] - 2U[1, i] + U[1, i + 1]) / dx^2
                    S[1, i] = source(i)
                    C[1, i] = -U[1, i]^3
                end
                return nothing
            end
            at(u, i) = 1 <= i <= N ? u[i] : 0.0
            laplacian(u, i) = (at(u, i - 1) - 2u[i] + at(u, i + 1)) / dx^2
            function rd!(du, u, p, t)
                for i in 1:N
                    du[i] = p[1] * laplacian(u, i) + p[2] * source(i) - p[3] * u[i]^3
                end
                return nothing
            end
            function rd_jac!(J, u, p, t)
                for i in 1:N, j in max(1, i - 1):min(N, i + 1)
                    J[i, j] = i == j ? -2p[1] / dx^2 - 3p[3] * u[i]^2 : p[1] / dx^2
                end
                return nothing
            end
            function rd_paramjac!(pJ, u, p, t)
                for i in 1:N
                    pJ[i, 1], pJ[i, 2], pJ[i, 3] = laplacian(u, i), source(i), -u[i]^3
                end
                return nothing
            end
            q0 = [0.8, 1.5, 0.5]
            v0 = sinpi.((1:N) .* dx)
            on_dm(tspan = (0.0, 0.1); p = q0, u0 = v0, jac = rd_jac_dm!, paramjac = rd_paramjac_dm!) =
                SciMLBase.ODEProblem(SciMLBase.ODEFunction(rd_dm!; jac, paramjac), u0, tspan, p)
            plain(tspan = (0.0, 0.1); p = q0) = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(rd!; jac = rd_jac!, paramjac = rd_paramjac!), v0, tspan, p,
            )
            weights = (1:N) .* dx
            cost(u, p) = sum(abs2, u) / 2 + p[2] * sum(u .* weights)
            cost_du!(out, u, p, t, i...) = (out .= u .+ p[2] .* weights; nothing)
            cost_dp!(out, u, p, t, i...) = (fill!(out, 0.0); out[2] = sum(u .* weights); nothing)
            times = collect(0.0:0.01:0.1)
            function dm_grad(prob, alg; t = times, dgdu_discrete = cost_du!, kw...)
                du0, dp = PETScDiffEq._discrete_adjoint(
                    prob, alg, PETScAdjoint(); t, dgdu_discrete, dt = 1.0e-3, adaptive = false,
                    kw...,
                )
                return vcat(du0, vec(dp))
            end
            beuler(; kw...) = TSImplicit("beuler", exact; kw...)
            cn(; kw...) = TSImplicit("cn", exact; kw...)
            methods = (
                ("RK4", (; kw...) -> TSRK("4"; kw...)), ("backward Euler", beuler),
                ("Crank-Nicolson", cn), ("theta 0.7", (; kw...) -> TSImplicit("theta", 0.7, exact; kw...)),
                ("ARKIMEX l2", (; kw...) -> TSARKIMEX("l2", exact; kw...)),
            )
            # Measured: within 5.1e-16 of the adjoint without the DM and 1.2e-9 of the differences.
            @testset "matches the adjoint without the DM and finite differences: $name" for (
                    name, make,
                ) in methods
                for (tspan, t, p) in (
                        ((0.0, 0.1), times, q0),
                        ((0.1, 0.0), reverse(times), [-q0[1], q0[2], -q0[3]]),
                    )
                    got = dm_grad(on_dm(tspan; p), make(; dm = da); t, dgdp_discrete = cost_dp!)
                    ref = dm_grad(plain(tspan; p), make(); t, dgdp_discrete = cost_dp!)
                    @test relerr(got, ref) < 1.0e-14
                end
                alg = make(; dm = da)
                function loss(θ)
                    sol = SciMLBase.solve(
                        on_dm(; u0 = θ[1:N], p = θ[(N + 1):end]), alg; dt = 1.0e-3,
                        adaptive = false, saveat = times,
                    )
                    return sum(cost(u, θ[(N + 1):end]) for u in sol.u)
                end
                got = dm_grad(on_dm(), alg; dgdp_discrete = cost_dp!)
                @test relerr(got, central_differences(loss, vcat(v0, q0))) < 5.0e-9
            end

            @testset "the costs get the owned state as the solve saves it" begin
                seen = Vector{Float64}[]
                record(out, u, p, t, i) = (push!(seen, copy(u)); out .= u; nothing)
                dm_grad(on_dm(), TSRK("4"; dm = da); dgdu_discrete = record)
                sol = SciMLBase.solve(
                    on_dm(), TSRK("4"; dm = da); dt = 1.0e-3, adaptive = false, saveat = times,
                )
                @test reverse(seen) == sol.u
            end

            @testset "integral costs, an adaptive solve and a split problem, as without it" begin
                g(u, p, t) = cost(u, p)
                gu!(out, u, p, t) = cost_du!(out, u, p, t)
                gp!(out, u, p, t) = cost_dp!(out, u, p, t)
                for (make, costs) in (
                        ((; kw...) -> TSRK("4"; kw...), (; g)),
                        (cn, (; dgdu_continuous = gu!, dgdp_continuous = gp!)),
                    )
                    cost_kw = (; t = nothing, dgdu_discrete = nothing, costs...)
                    got = dm_grad(on_dm(), make(; dm = da); cost_kw...)
                    @test relerr(got, dm_grad(plain(), make(); cost_kw...)) < 1.0e-14
                end
                adaptive = (; t = [0.0, 0.1], adaptive = true, abstol = 1.0e-8, reltol = 1.0e-8)
                got = dm_grad(on_dm(), TSRK("5dp"; dm = da); adaptive...)
                @test relerr(got, dm_grad(plain(), TSRK("5dp"); adaptive...)) < 1.0e-14
                f1_dm!(du, u, p, t) = rd_dm!(du, u, [p[1], 0.0, 0.0], t)
                f2_dm!(du, u, p, t) = rd_dm!(du, u, [0.0, p[2], p[3]], t)
                halves = SciMLBase.SplitODEProblem(
                    SciMLBase.ODEFunction(
                        f1_dm!; jac = (J, u, p, t) -> rd_jac_dm!(J, u, [p[1], 0.0, 0.0], t),
                        paramjac = (pJ, u, p, t) -> (rd_paramjac_dm!(pJ, u, p, t); pJ[:, 2:3] .= 0),
                    ),
                    SciMLBase.ODEFunction(
                        f2_dm!; jac = (J, u, p, t) -> rd_jac_dm!(J, u, [0.0, 0.0, p[3]], t),
                        paramjac = (pJ, u, p, t) -> (rd_paramjac_dm!(pJ, u, p, t); pJ[:, 1] .= 0),
                    ),
                    v0, (0.0, 0.1), q0,
                )
                summed = SciMLBase.SplitODEProblem(
                    (du, u, p, t) -> rd!(du, u, [p[1], 0.0, 0.0], t),
                    (du, u, p, t) -> rd!(du, u, [0.0, p[2], p[3]], t), v0, (0.0, 0.1), q0,
                )
                got = dm_grad(halves, TSARKIMEX("3", exact; dm = da); dgdp_discrete = cost_dp!)
                ref = dm_grad(summed, TSARKIMEX("3", exact); dgdp_discrete = cost_dp!)
                @test relerr(got, ref) < 1.0e-14
            end

            @testset "an explicit solve takes the jac and ignores it" begin
                fixed = (; dt = 1.0e-3, adaptive = false)
                @test SciMLBase.solve(on_dm(), TSRK("4"; dm = da); fixed...).u ==
                    SciMLBase.solve(on_dm(; jac = nothing), TSRK("4"; dm = da); fixed...).u
            end

            @testset "what it refuses with a dm" begin
                pl32 = PETSc.getlib(; PetscScalar = Float32)
                PETScDiffEq.PETScCompat.isinitialized(pl32) || PETSc.initialize(pl32)
                da32 = PETSc.DMDA(pl32, MPI.COMM_SELF, (LibPETSc.DM_BOUNDARY_GHOSTED,), (N,), 1, 1)
                swing = SciMLBase.DynamicalODEProblem(
                    (dv, v, u, p, t) -> (dv .= -u; nothing), (du, v, u, p, t) -> (du .= v; nothing),
                    v0, v0, (0.0, 0.1),
                )
                for (message, call) in (
                        (
                            "PETScAdjoint needs the ODEFunction's `jac` with a `dm`",
                            () -> dm_grad(on_dm(; jac = nothing), TSRK("4"; dm = da)),
                        ),
                        (
                            "PETScAdjoint needs the ODEFunction's `jac` with a `dm`",
                            () -> dm_grad(on_dm(; jac = nothing), TSImplicit("cn"; dm = da)),
                        ),
                        (
                            "PETScAdjoint needs the ODEFunction's `paramjac` with a `dm`",
                            () -> dm_grad(on_dm(; paramjac = nothing), cn(; dm = da)),
                        ),
                        (
                            "PETScAdjoint runs in PETSc's double real build, so its `dm` has to " *
                                "belong to that build, not to the Float32 real one",
                            () -> dm_grad(on_dm(), TSRK("4"; dm = da32)),
                        ),
                        (
                            "PETScAdjoint takes an ODEProblem or a SplitODEProblem with a `dm`",
                            () -> dm_grad(swing, TSRK("4"; dm = da)),
                        ),
                    )
                    @test_throws "ArgumentError: $message" call()
                end
                PETScDiffEq.PETScCompat.destroy!(da32)
            end
            PETScDiffEq.PETScCompat.destroy!(da)
        end

        @testset "what it refuses, and why" begin
            prob = adj_prob(copy(u0), copy(p0), (0.0, 1.0))
            never = SciMLBase.DiscreteCallback((u, t, integ) -> false, integ -> nothing)
            residual!(r, du, u, p, t) = (r .= du .+ u; nothing)
            without(; jac = adj_jac!, paramjac = adj_paramjac!, mass_matrix = I, p = p0) =
                SciMLBase.ODEProblem(
                SciMLBase.ODEFunction(adj_f!; jac, paramjac, mass_matrix), copy(u0), (0.0, 1.0), p,
            )
            solely_states = SciMLBase.ODEProblem(
                SciMLBase.ODEFunction((du, u, p, t) -> (du .= -u); jac = (J, u, p, t) -> (J .= -I)),
                copy(u0), (0.0, 1.0),
            )
            second = SciMLBase.SecondOrderODEProblem(
                (ddu, du, u, p, t) -> (ddu .= -p[1] .* u; nothing), [0.0], [1.0], (0.0, 1.0), [1.0],
            )
            runs(type) = "PETScAdjoint supports PETSc's rk, beuler, cn, theta and arkimex as " *
                "this package drives them, but this solve runs `$type`"
            parts = split_prob(copy(u0), copy(p0), (0.0, 1.0))
            differenced = PETScDiffEq.AutoFiniteDiff()
            trajectory(type) = "PETScAdjoint keeps its trajectory in memory, or on disk " *
                "with `-ts_trajectory_type basic`, but this solve's is `$type`"
            no_ksp = [
                "-ksp_type", "gmres", "-ksp_max_it", "1", "-pc_type", "none",
                "-snes_max_linear_solve_fail", "1000", "-ts_max_snes_failures", "-1",
            ]
            @testset "$message" for (message, call) in (
                    ("PETSc has no adjoint for TSRosW", () -> grad(prob, TSRosW())),
                    ("PETSc has no adjoint for TSIRK", () -> grad(prob, TSIRK(2))),
                    ("PETSc has no adjoint for TSMPRK", () -> grad(prob, TSMPRK([1]))),
                    ("PETSc has no adjoint for TSDAE", () -> grad(prob, TSDAE("beuler"))),
                    (
                        "PETSc has no adjoint for TSBasicSymplectic",
                        () -> grad(second, TSBasicSymplectic("velverlet")),
                    ),
                    ("PETSc has no adjoint for TSAlpha2", () -> grad(second, TSAlpha2())),
                    (
                        "PETSc has no adjoint for TSImplicit(\"bdf\")",
                        () -> grad(prob, TSImplicit("bdf")),
                    ),
                    (
                        "PETScAdjoint cannot take an integral cost with TSARKIMEX",
                        () -> integral(prob, TSARKIMEX(); g = coupled),
                    ),
                    (
                        "PETScAdjoint cannot take an integral cost with TSARKIMEX",
                        () -> grad(parts, TSARKIMEX(); g = coupled),
                    ),
                    (
                        "PETScAdjoint does not support `-ts_arkimex_fully_implicit` on a " *
                            "SplitODEProblem",
                        () -> grad(parts, TSARKIMEX("3", ["-ts_arkimex_fully_implicit"])),
                    ),
                    (
                        "PETScAdjoint supports a SplitODEProblem with TSARKIMEX only",
                        () -> grad(parts, TSRK("4")),
                    ),
                    (
                        "PETScAdjoint differentiates a SplitODEProblem with PETSc's arkimex " *
                            "only, but this solve runs `beuler`",
                        () -> grad(parts, TSARKIMEX("3", ["-ts_type", "beuler"])),
                    ),
                    (
                        "PETScAdjoint needs `f2`'s `jac` under `autodiff = AutoFiniteDiff()`",
                        () -> grad(
                            split_prob(copy(u0), copy(p0), (0.0, 1.0); f2_jac = nothing),
                            TSARKIMEX("3"; autodiff = differenced),
                        ),
                    ),
                    (
                        "PETScAdjoint needs `f2`'s `paramjac` under `autodiff = AutoFiniteDiff()`",
                        () -> grad(
                            split_prob(copy(u0), copy(p0), (0.0, 1.0); f2_paramjac = nothing),
                            TSARKIMEX("3"; autodiff = differenced),
                        ),
                    ),
                    (
                        "PETScAdjoint does not support a mass matrix",
                        () -> grad(without(mass_matrix = [2.0 0.0; 0.0 1.0]), TSARKIMEX()),
                    ),
                    (runs("euler"), () -> grad(prob, TSGeneric("euler"; explicit = true))),
                    (runs("bdf"), () -> grad(prob, TSImplicit("beuler", ["-ts_type", "bdf"]))),
                    (
                        "PETScAdjoint does not support multirate TSRK",
                        () -> grad(prob, TSRK("4", ["-ts_rk_multirate", "1"])),
                    ),
                    (
                        "PETScAdjoint needs the solve to end on a step at tspan's end",
                        () -> grad(prob, TSRK("4", ["-ts_exact_final_time", "interpolate"])),
                    ),
                    (
                        "`-ts_adjoint_solve` makes PETSc start the adjoint",
                        () -> grad(prob, TSRK("4", ["-ts_adjoint_solve", "1"])),
                    ),
                    (
                        "`-ts_adjoint_solve` makes PETSc start the adjoint",
                        () -> grad(
                            prob, TSRK("4");
                            sensealg = PETScAdjoint(petsc_options = ["-ts_adjoint_solve", "1"]),
                        ),
                    ),
                    (
                        "`-ts_adjoint_solve` makes PETSc start the adjoint",
                        () -> grad(prob, TSRK("4", ["-ts_adjoint_solve=1"])),
                    ),
                    (
                        "`-ts_adjoint_solve` makes PETSc start the adjoint",
                        () -> grad(
                            prob, TSRK("4");
                            sensealg = PETScAdjoint(petsc_options = ["-TS_ADJOINT_SOLVE", "1"]),
                        ),
                    ),
                    (
                        trajectory("singlefile"),
                        () -> cd(mktempdir()) do
                            grad(
                                prob, TSRK("4"); sensealg = PETScAdjoint(
                                    petsc_options = ["-ts_trajectory_type", "singlefile"],
                                ),
                            )
                        end,
                    ),
                    (
                        trajectory("none"),
                        () -> grad(
                            prob, TSRK("4");
                            sensealg = PETScAdjoint(petsc_options = ["-ts_save_trajectory", "0"]),
                        ),
                    ),
                    (
                        "PETScAdjoint supports an ODEProblem, not a DAEProblem",
                        () -> grad(
                            SciMLBase.DAEProblem(residual!, zeros(2), ones(2), (0.0, 1.0), p0),
                            TSDAE("beuler"),
                        ),
                    ),
                    (
                        "PETScAdjoint supports a real state only",
                        () -> grad(adj_prob(ComplexF64.(u0), copy(p0), (0.0, 1.0)), TSRK("4")),
                    ),
                    (
                        "PETScAdjoint does not support a mass matrix",
                        () -> grad(without(mass_matrix = [2.0 0.0; 0.0 1.0]), TSImplicit("beuler")),
                    ),
                    (
                        "PETScAdjoint needs the ODEFunction's `jac`",
                        () -> grad(
                            without(jac = nothing),
                            TSImplicit("beuler", exact; autodiff = PETScDiffEq.AutoFiniteDiff()),
                        ),
                    ),
                    (
                        "PETScAdjoint needs the ODEFunction's `paramjac`",
                        () -> grad(
                            without(paramjac = nothing),
                            TSImplicit("beuler", exact; autodiff = PETScDiffEq.AutoFiniteDiff()),
                        ),
                    ),
                    (
                        "PETScAdjoint needs `p` to be a vector of real numbers",
                        () -> grad(without(p = (0.7, 0.3, 0.4, 0.2)), TSRK("4")),
                    ),
                    (
                        "`dgdp_discrete` was given, but the problem has no parameters",
                        () -> grad(
                            solely_states, TSRK("4"); dgdp_discrete = (out, u, p, t, i) -> nothing,
                        ),
                    ),
                    (
                        "PETScAdjoint does not support callbacks",
                        () -> grad(prob, TSRK("4"); callback = never),
                    ),
                    (
                        "PETScAdjoint does not support callbacks",
                        () -> grad(SciMLBase.remake(prob; callback = never), TSRK("4")),
                    ),
                    (
                        "PETScAdjoint does not support `tstops`",
                        () -> grad(prob, TSRK("4"); tstops = [0.5]),
                    ),
                    (
                        "PETScAdjoint does not support `isoutofdomain`",
                        () -> grad(prob, TSRK("4"); isoutofdomain = (u, p, t) -> false),
                    ),
                    (
                        "the forward solve's `unstable_check` fired at t = 0.5",
                        () -> grad(prob, TSRK("4"); unstable_check = (dt, u, p, t) -> t >= 0.5),
                    ),
                    (
                        "PETScAdjoint does not support `tstops`",
                        () -> grad(SciMLBase.remake(prob; tstops = [0.5]), TSRK("4")),
                    ),
                    (
                        "PETScAdjoint does not support `d_discontinuities`",
                        () -> grad(prob, TSRK("4"); d_discontinuities = [0.5]),
                    ),
                    (
                        "PETScAdjoint does not support `save_idxs`",
                        () -> grad(prob, TSRK("4"); save_idxs = [1]),
                    ),
                    ("PETScAdjoint needs cost times", () -> grad(prob, TSRK("4"); t = nothing)),
                    ("PETScAdjoint needs cost times", () -> grad(prob, TSRK("4"); t = Float64[])),
                    (
                        "PETScAdjoint needs cost times",
                        () -> grad(prob, TSRK("4"); dgdu_discrete = nothing),
                    ),
                    (
                        "PETScAdjoint needs the cost times `t` as a vector of real numbers",
                        () -> grad(prob, TSRK("4"); t = 1.0),
                    ),
                    (
                        "PETScAdjoint needs cost times `t` and `dgdu_discrete(out, u, p, t, i)` together",
                        () -> grad(prob, TSRK("4"); dgdu_discrete = nothing, g = coupled),
                    ),
                    (
                        "`dgdp_continuous` was given without `g` or `dgdu_continuous`",
                        () -> grad(prob, TSRK("4"); dgdp_continuous = integrand_dp!),
                    ),
                    (
                        "`dgdp_continuous` was given, but the problem has no parameters",
                        () -> integral(
                            solely_states, TSRK("4"); g = (u, p, t) -> sum(u),
                            dgdp_continuous = integrand_dp!,
                        ),
                    ),
                    (
                        "PETScAdjoint needs `dgdu_continuous` under `autodiff = AutoFiniteDiff()`",
                        () -> integral(
                            prob, TSImplicit("beuler", exact; autodiff = PETScDiffEq.AutoFiniteDiff());
                            g = coupled,
                        ),
                    ),
                    (
                        "the derivative of the integral cost `g` from automatic differentiation " *
                            "has a non-finite entry",
                        () -> integral(prob, TSRK("4"); g = (u, p, t) -> sqrt(abs(u[1] - 1))),
                    ),
                    (
                        "cost time 1.5 lies outside tspan = (0.0, 1.0)",
                        () -> grad(prob, TSRK("4"); t = [1.5]),
                    ),
                    (
                        "cost time -0.1 lies outside tspan = (1.0, 0.0)",
                        () -> grad(adj_prob(copy(u0), copy(p0), (1.0, 0.0)), TSRK("4"); t = [-0.1]),
                    ),
                    (
                        "cost time t[1] = 0.105 is not a time the solve stepped to",
                        () -> grad(prob, TSRK("4"); t = [0.105]),
                    ),
                    (
                        "the forward solve stopped with TS_CONVERGED_ITS",
                        () -> grad(prob, TSRK("4"); maxiters = 5),
                    ),
                    (
                        "the forward solve ended on a state that is not finite",
                        () -> grad(
                            SciMLBase.ODEProblem(
                                SciMLBase.ODEFunction(
                                    (du, u, p, t) -> (du .= u .^ 2; nothing);
                                    jac = (J, u, p, t) -> (J .= Diagonal(2 .* u); nothing),
                                ),
                                [1.0, 0.5], (0.0, 2.0),
                            ),
                            TSRK("4"); t = [2.0],
                        ),
                    ),
                    (
                        "PETSc's adjoint solve stopped with TSADJOINT_DIVERGED_LINEAR_SOLVE",
                        () -> grad(
                            prob, TSImplicit("beuler");
                            sensealg = PETScAdjoint(petsc_options = no_ksp),
                        ),
                    ),
                    (
                        "PETScAdjoint is reached through `adjoint_sensitivities",
                        () -> SciMLBase._concrete_solve_adjoint(
                            prob, TSRK("4"), PETScAdjoint(), u0, p0,
                            SciMLBase.ChainRulesOriginator(),
                        ),
                    ),
                )
                @test_throws "ArgumentError: $message" call()
            end
        end
    end

    @testset "second-order and partitioned problems" begin
        osc!(ddu, du, u, p, t) = (ddu .= -u; nothing)
        forced!(ddu, du, u, p, t) = (ddu .= -u .+ cos(2t); nothing)
        pend!(ddu, du, u, p, t) = (ddu .= -sin.(u); nothing)
        osc_exact(t) = [-sin(t), cos(t)]
        forced_exact(t) = [-(4 / 3) * sin(t) + (2 / 3) * sin(2t), (4 / 3) * cos(t) - cos(2t) / 3]
        final_err(sol, exact) = maximum(abs.(collect(sol.u[end]) .- exact))
        function orders(prob, alg, exact)
            errs = map((0.1, 0.05, 0.025, 0.0125)) do dt
                sol = SciMLBase.solve(prob, alg; dt, adaptive = false, save_everystep = false)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                final_err(sol, exact)
            end
            return [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
        end

        @testset "convergence order: $name" for (name, alg, p) in (
                ("sieuler", PETScDiffEq.TSBasicSymplectic("sieuler"), 1),
                ("velverlet", PETScDiffEq.TSBasicSymplectic("velverlet"), 2),
                ("3", PETScDiffEq.TSBasicSymplectic("3"), 3),
                ("4", PETScDiffEq.TSBasicSymplectic("4"), 4),
                ("alpha2", PETScDiffEq.TSAlpha2(), 2),
                ("alpha2 radius 0.5", PETScDiffEq.TSAlpha2(; radius = 0.5), 2),
            )
            @test SciMLBase.alg_order(alg) == p
            osc = SciMLBase.SecondOrderODEProblem(osc!, [0.0], [1.0], (0.0, 10.0))
            forced = SciMLBase.SecondOrderODEProblem(forced!, [0.0], [1.0], (0.0, 10.0))
            @test all(o -> isapprox(o, p; atol = 0.1), orders(osc, alg, osc_exact(10.0)))
            @test all(o -> isapprox(o, p; atol = 0.1), orders(forced, alg, forced_exact(10.0)))
        end

        @testset "a subtype picked by option keeps its order" begin
            forced = SciMLBase.SecondOrderODEProblem(forced!, [0.0], [1.0], (0.0, 10.0))
            alg = PETScDiffEq.TSBasicSymplectic("velverlet", ["-ts_basicsymplectic_type", "4"])
            @test all(o -> isapprox(o, 4; atol = 0.1), orders(forced, alg, forced_exact(10.0)))
        end

        @testset "velverlet is velocity Verlet with the force at the positions' time" begin
            a(u, t) = -sin(u) + 0.3cos(t)
            f!(ddu, du, u, p, t) = (ddu .= a.(u, t); nothing)
            v, u, h = 0.0, 2.0, 0.1
            for n in 0:99
                v += h / 2 * a(u, n * h)
                u += h * v
                v += h / 2 * a(u, (n + 1) * h)
            end
            prob = SciMLBase.SecondOrderODEProblem(f!, [0.0], [2.0], (0.0, 10.0))
            sol = SciMLBase.solve(prob, PETScDiffEq.TSBasicSymplectic(); dt = h)
            @test collect(sol.u[end]) ≈ [v, u] rtol = 1.0e-12
        end

        @testset "velverlet reuses the kick that ended the last step" begin
            kick!(dv, v, u, p, t) = (dv .= -p[1] .* u .+ 0.3cos(t); nothing)
            prob = SciMLBase.SecondOrderODEProblem(kick!, [0.0], [1.0], (0.0, 1.0), [1.0])
            sol = SciMLBase.solve(prob, PETScDiffEq.TSBasicSymplectic(); dt = 0.1, dense = false)
            @test (sol.stats.nf, sol.stats.nf2) == (11, 10)
            v, u = 0.0, 1.0
            for n in 0:9
                v += 0.05 * (-u + 0.3cos(0.1n))
                u += 0.1v
                v += 0.05 * (-u + 0.3cos(0.1(n + 1)))
            end
            @test collect(sol.u[end]) ≈ [v, u] rtol = 1.0e-12
            # A changed p or state takes a fresh kick.
            flip = PresetTimeCallback([0.5], integ -> (integ.p[1] = 4.0))
            cb = SciMLBase.solve(
                SciMLBase.remake(prob; p = [1.0]), PETScDiffEq.TSBasicSymplectic();
                dt = 0.1, callback = flip, dense = false,
            )
            v, u = 0.0, 1.0
            for n in 0:9
                k = n < 5 ? 1.0 : 4.0
                v += 0.05 * (-k * u + 0.3cos(0.1n))
                u += 0.1v
                v += 0.05 * (-k * u + 0.3cos(0.1(n + 1)))
            end
            @test collect(cb.u[end]) ≈ [v, u] rtol = 1.0e-12
            integ = SciMLBase.init(prob, PETScDiffEq.TSBasicSymplectic(); dt = 0.1)
            SciMLBase.step!(integ)
            SciMLBase.set_u!(integ, 2 .* integ.u)
            SciMLBase.solve!(integ)
            fresh = SciMLBase.solve(
                SciMLBase.remake(prob; u0 = 2 .* sol(0.1), tspan = (0.1, 1.0)),
                PETScDiffEq.TSBasicSymplectic(); dt = 0.1,
            )
            @test collect(integ.sol.u[end]) ≈ collect(fresh.u[end]) rtol = 1.0e-12
        end

        # PETSc's absolute-eps step check fails a span this long on 32-bit x86.
        Sys.WORD_SIZE == 64 && @testset "the energy error stays bounded" begin
            energy(s) = s.x[1][1]^2 / 2 - cos(s.x[2][1])
            prob = SciMLBase.SecondOrderODEProblem(pend!, [0.0], [2.0], (0.0, 1000.0))
            function drift(alg)
                sol = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false)
                dE = [abs(energy(s) - energy(sol.u[1])) for s in sol.u]
                n = length(dE) ÷ 2
                return maximum(dE[1:n]), maximum(dE[(n + 1):end])
            end
            for (alg, bound) in (
                    (PETScDiffEq.TSBasicSymplectic(), 3.0e-3),
                    (PETScDiffEq.TSBasicSymplectic("4"), 7.0e-6),
                )
                first, second = drift(alg)
                @test second < 1.001 * first
                @test second < bound
            end
            first, second = drift(PETScDiffEq.TSRK("4"))
            @test second > 1.9 * first
        end

        @testset "nonlinear pendulum" begin
            prob = SciMLBase.SecondOrderODEProblem(pend!, [0.0], [2.0], (0.0, 10.0))
            ref = SciMLBase.solve(prob, PETScDiffEq.TSRK("8vr"); abstol = 1.0e-13, reltol = 1.0e-13)
            for (alg, p) in (
                    (PETScDiffEq.TSBasicSymplectic("sieuler"), 1),
                    (PETScDiffEq.TSBasicSymplectic(), 2),
                    (PETScDiffEq.TSBasicSymplectic("3"), 3),
                    (PETScDiffEq.TSBasicSymplectic("4"), 4),
                    (PETScDiffEq.TSAlpha2(), 2),
                )
                @test all(o -> isapprox(o, p; atol = 0.1), orders(prob, alg, collect(ref.u[end])))
            end
        end

        @testset "TSAlpha2 on a stiff damped oscillator" begin
            Q = [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)]
            K = Q * Diagonal([1.0, 1.0e6]) * Q'
            C = Q * Diagonal([0.02, 200.0]) * Q'
            damped!(ddu, du, u, p, t) = (mul!(ddu, K, u); mul!(ddu, C, du, -1.0, -1.0); nothing)
            function damped_jac!(J, x, p, t)
                @test hasproperty(x, :x)
                fill!(J, 0.0)
                J[1:2, 1:2] .= -C
                J[1:2, 3:4] .= -K
                J[3, 1] = J[4, 2] = 1.0
                return nothing
            end
            velocity!(du, v, u, p, t) = (du .= v; nothing)
            x0 = [0.0, 0.0, 1.0, 1.0]
            exact = exp([-C -K; I zeros(2, 2)] * 5.0) * x0
            prob = SciMLBase.SecondOrderODEProblem(damped!, x0[1:2], x0[3:4], (0.0, 5.0))
            jprob = SciMLBase.SecondOrderODEProblem(
                SciMLBase.DynamicalODEFunction{true}(damped!, velocity!; jac = damped_jac!),
                x0[1:2], x0[3:4], (0.0, 5.0),
            )
            @test all(o -> isapprox(o, 2; atol = 0.05), orders(prob, PETScDiffEq.TSAlpha2(; radius = 0.5), exact))
            solve_at(pr, alg) = SciMLBase.solve(pr, alg; dt = 0.1, adaptive = false)
            ad = solve_at(prob, PETScDiffEq.TSAlpha2(; radius = 0.5))
            user = solve_at(jprob, PETScDiffEq.TSAlpha2(; radius = 0.5))
            fd = solve_at(prob, PETScDiffEq.TSAlpha2(; radius = 0.5, autodiff = PETScDiffEq.AutoFiniteDiff()))
            @test collect(user.u[end]) ≈ collect(ad.u[end]) rtol = 1.0e-10
            @test ad.stats.njacs > 0 && user.stats.njacs == ad.stats.njacs && fd.stats.njacs == 0
            for sol in (ad, user, fd)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test final_err(sol, exact) < 7.0e-3
                @test abs((Q' * sol.u[end].x[2])[2]) < 1.0e-12
            end
            undamped = solve_at(prob, PETScDiffEq.TSAlpha2())
            @test abs((Q' * undamped.u[end].x[2])[2]) > 0.1
            explicit = SciMLBase.solve(prob, PETScDiffEq.TSBasicSymplectic(); dt = 0.01)
            @test explicit.retcode == SciMLBase.ReturnCode.Unstable
            adaptive = SciMLBase.solve(
                prob, PETScDiffEq.TSAlpha2(; radius = 0.5); abstol = 1.0e-5, reltol = 1.0e-5,
            )
            @test adaptive.retcode == SciMLBase.ReturnCode.Success
            @test final_err(adaptive, exact) < 5.0e-5
        end

        @testset "TSAlpha2 with a sparse jac_prototype" begin
            Q = [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)]
            K = Q * Diagonal([1.0, 1.0e6]) * Q'
            C = Q * Diagonal([0.02, 200.0]) * Q'
            damped!(ddu, du, u, p, t) = (mul!(ddu, K, u); mul!(ddu, C, du, -1.0, -1.0); nothing)
            function damped_jac!(J, x, p, t)
                fill!(J, 0.0)
                J[1:2, 1:2] .= -C
                J[1:2, 3:4] .= -K
                J[3, 1] = J[4, 2] = 1.0
                return nothing
            end
            velocity!(du, v, u, p, t) = (du .= v; nothing)
            x0 = [0.0, 0.0, 1.0, 1.0]
            exact = exp([-C -K; I zeros(2, 2)] * 5.0) * x0
            proto = sparse([ones(2, 2) ones(2, 2); Matrix(I, 2, 2) zeros(2, 2)])
            second(f = damped!; kw...) = SciMLBase.SecondOrderODEProblem(
                SciMLBase.DynamicalODEFunction{true}(f, velocity!; kw...), x0[1:2], x0[3:4],
                (0.0, 5.0),
            )
            # DiffEqBase's solve cannot rebuild a DynamicalODEFunction that carries a jac_prototype.
            solve_at(pr, alg; kw...) = SciMLBase.__solve(pr, alg; dt = 0.1, adaptive = false, kw...)
            alg = PETScDiffEq.TSAlpha2(; radius = 0.5)
            fd = PETScDiffEq.TSAlpha2(; radius = 0.5, autodiff = PETScDiffEq.AutoFiniteDiff())
            dense = solve_at(second(; jac = damped_jac!), alg)
            user = solve_at(second(; jac = damped_jac!, jac_prototype = proto), alg)
            ad = solve_at(second(; jac_prototype = proto), alg)
            coloured = solve_at(second(; jac_prototype = proto), fd)
            oop = solve_at(
                SciMLBase.SecondOrderODEProblem(
                    SciMLBase.DynamicalODEFunction{false}(
                        (du, u, p, t) -> -K * u - C * du, (v, u, p, t) -> v;
                        jac = (x, p, t) -> sparse([-C -K; I zeros(2, 2)]), jac_prototype = proto,
                    ), x0[1:2], x0[3:4], (0.0, 5.0),
                ), alg,
            )
            for sol in (user, ad, coloured, oop)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test final_err(sol, exact) < 7.0e-3
            end
            for sol in (user, ad, oop)
                @test collect(sol.u[end]) ≈ collect(dense.u[end]) rtol = 1.0e-10
                @test sol.stats.njacs == dense.stats.njacs > 0
            end
            @test coloured.stats.njacs == 0
            if second(; jac_prototype = proto).f.jac_prototype isa SparseMatrixCSC
                @test ad.stats.nf > user.stats.nf == dense.stats.nf
                @test_throws "of the first-order system `[v; u]' = [f(v, u, p, t); v]`, 4 x 4" solve_at(
                    second(; jac_prototype = sparse(ones(2, 2))), alg,
                )
                @test_throws "but this one is 4 x 3" solve_at(
                    second(; jac_prototype = sparse(ones(4, 3))), alg,
                )
            end

            function wave(N; jac = true, proto = true)
                h = 1 / (N + 1)
                x = h .* (1:N)
                s = 1 / h^2
                Lap = spdiagm(-1 => fill(s, N - 1), 0 => fill(-2s, N), 1 => fill(s, N - 1))
                Jc = [spzeros(N, N) Lap; sparse(1.0I, N, N) spzeros(N, N)]
                f!(ddu, du, u, p, t) = (mul!(ddu, Lap, u); nothing)
                jac!(J, x, p, t) = (
                    J isa SparseMatrixCSC ? (nonzeros(J) .= nonzeros(Jc)) : copyto!(J, Jc); nothing
                )
                kw = (; (jac ? (:jac => jac!,) : ())..., (proto ? (:jac_prototype => Jc,) : ())...)
                w = 2 / h * sin(pi * h / 2)
                standing(t) = vcat(-w .* sin.(pi .* x) .* sin(w * t), sin.(pi .* x) .* cos(w * t))
                prob = SciMLBase.SecondOrderODEProblem(
                    SciMLBase.DynamicalODEFunction{true}(f!, velocity!; kw...), zeros(N),
                    sin.(pi .* x), (0.0, 1.0),
                )
                return prob, standing
            end
            function wave_errors(sol, standing, N)
                ev = maximum(
                    maximum(abs, sol.u[i].x[1] .- standing(sol.t[i])[1:N]) for i in eachindex(sol.t)
                )
                eu = maximum(
                    maximum(abs, sol.u[i].x[2] .- standing(sol.t[i])[(N + 1):end]) for
                        i in eachindex(sol.t)
                )
                return ev, eu
            end
            wave_at(pr, alg) = SciMLBase.__solve(pr, alg; dt = 0.005, adaptive = false)
            small, standing = wave(200)
            small_dense = wave_at(first(wave(200; proto = false)), PETScDiffEq.TSAlpha2())
            for (pr, alg, rtol) in (
                    (small, PETScDiffEq.TSAlpha2(), 1.0e-11),
                    (first(wave(200; jac = false)), PETScDiffEq.TSAlpha2(), 1.0e-11),
                    (
                        first(wave(200; jac = false)),
                        PETScDiffEq.TSAlpha2(; autodiff = PETScDiffEq.AutoFiniteDiff()), 1.0e-6,
                    ),
                )
                sol = wave_at(pr, alg)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.t == small_dense.t
                @test collect(sol.u[end]) ≈ collect(small_dense.u[end]) rtol = rtol
            end
            # The dense path takes about 28 s here against 0.05 s.
            for (pr, alg) in (
                    (first(wave(2000)), PETScDiffEq.TSAlpha2()),
                    (first(wave(2000; jac = false)), PETScDiffEq.TSAlpha2()),
                    (
                        first(wave(2000; jac = false)),
                        PETScDiffEq.TSAlpha2(; autodiff = PETScDiffEq.AutoFiniteDiff()),
                    ),
                )
                sol = wave_at(pr, alg)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test length(sol.t) == 201
                ev, eu = wave_errors(sol, last(wave(2000)), 2000)
                @test ev < 1.0e-3
                @test eu < 2.0e-4
            end
        end

        @testset "TSAlpha2 adapts on PETSc's estimate" begin
            prob = SciMLBase.SecondOrderODEProblem(osc!, [0.0], [1.0], (0.0, 10.0))
            steps = map((1.0e-4, 1.0e-6)) do tol
                sol = SciMLBase.solve(prob, PETScDiffEq.TSAlpha2(); abstol = tol, reltol = tol)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test final_err(sol, osc_exact(10.0)) < 3.0 * tol
                sol.stats.naccept
            end
            @test 9 < steps[2] / steps[1] < 11
            fixed = SciMLBase.solve(prob, PETScDiffEq.TSAlpha2(); dt = 0.01, adaptive = false)
            @test fixed.stats.naccept == 1000
            # PETSc weighs the velocity and the position alike, so each pair takes the tighter.
            two = SciMLBase.SecondOrderODEProblem(osc!, [0.0, 0.0], [1.0, 2.0], (0.0, 10.0))
            scalar = SciMLBase.solve(two, PETScDiffEq.TSAlpha2(); abstol = 1.0e-6, reltol = 1.0e-6)
            for tol in (fill(1.0e-6, 4), [1.0e-2, 1.0e-2, 1.0e-6, 1.0e-6], [1.0e-6, 1.0e-2, 1.0e-2, 1.0e-6])
                sol = SciMLBase.solve(two, PETScDiffEq.TSAlpha2(); abstol = tol, reltol = tol)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.u[end] ≈ scalar.u[end] rtol = 1.0e-8
                @test abs(sol.stats.naccept - scalar.stats.naccept) <= 1
            end
            integ = SciMLBase.init(two, PETScDiffEq.TSAlpha2(); abstol = 1.0e-2, reltol = 1.0e-2)
            integ.opts.abstol = [1.0e-2, 1.0e-2, 1.0e-6, 1.0e-6]
            integ.opts.reltol = [1.0e-6, 1.0e-6, 1.0e-2, 1.0e-2]
            SciMLBase.solve!(integ)
            @test integ.sol.u[end] ≈ scalar.u[end] rtol = 1.0e-5
            @test abs(integ.sol.stats.naccept - scalar.stats.naccept) <= 5
            @test_throws "needs `dt`" SciMLBase.solve(prob, PETScDiffEq.TSBasicSymplectic())
        end

        @testset "states are ArrayPartitions, as in OrdinaryDiffEq" begin
            prob = SciMLBase.SecondOrderODEProblem(osc!, [0.0], [1.0], (0.0, 1.0))
            oop = SciMLBase.SecondOrderODEProblem((du, u, p, t) -> -u, [0.0], [1.0], (0.0, 1.0))
            for alg in (PETScDiffEq.TSBasicSymplectic(), PETScDiffEq.TSAlpha2(), PETScDiffEq.TSRK())
                sol = SciMLBase.solve(prob, alg; dt = 0.01)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test all(u -> u isa typeof(prob.u0), sol.u)
                @test sol(0.555) isa typeof(prob.u0)
                @test collect(sol(0.555)) ≈ osc_exact(0.555) atol = 1.0e-4
                @test collect(SciMLBase.solve(oop, alg; dt = 0.01).u[end]) == collect(sol.u[end])
                idxs = SciMLBase.solve(prob, alg; dt = 0.01, save_idxs = [2])
                @test idxs.u[end] isa Vector{Float64}
                @test idxs.u[end] == [sol.u[end][2]]
                integ = SciMLBase.init(prob, alg; dt = 0.01, adaptive = false)
                SciMLBase.step!(integ)
                @test integ.u isa typeof(prob.u0)
                @test integ(0.005) isa typeof(prob.u0)
                @test integ(0.005, Val{1}) isa typeof(prob.u0)
                @test collect(integ(0.005, Val{1})) ≈ [-cos(0.005), -sin(0.005)] atol = 1.0e-4
                @test integ(0.005; idxs = 2) isa Float64
                @test integ([0.0, 0.005]) isa Vector{typeof(prob.u0)}
                @test integ(similar(integ.u), 0.005, Val{1}) isa typeof(prob.u0)
                @test SciMLBase.get_du(integ) isa typeof(prob.u0)
                @test collect(SciMLBase.get_du(integ)) ≈ [-integ.u[2], integ.u[1]]
                SciMLBase.terminate!(integ)
            end
            m, k = 2.0, 3.0
            ω = sqrt(k / m)
            dyn = SciMLBase.DynamicalODEProblem(
                (dp, p, q, par, t) -> (dp .= -k .* q; nothing),
                (dq, p, q, par, t) -> (dq .= p ./ m; nothing), [0.0], [1.0], (0.0, 1.0),
            )
            sol = SciMLBase.solve(dyn, PETScDiffEq.TSBasicSymplectic(); dt = 0.01)
            @test final_err(sol, [-m * ω * sin(ω), cos(ω)]) < 5.0e-5
            complex = SciMLBase.SecondOrderODEProblem(osc!, ComplexF64[0], [1.0 + 1.0im], (0.0, 1.0))
            sol = SciMLBase.solve(complex, PETScDiffEq.TSBasicSymplectic(); dt = 0.01)
            @test sol.u[end] isa typeof(complex.u0)
            @test final_err(sol, (1 + 1im) .* osc_exact(1.0)) < 2.0e-5
            p32 = SciMLBase.SecondOrderODEProblem(osc!, Float32[0], Float32[1], (0.0f0, 1.0f0))
            sol = SciMLBase.solve(p32, PETScDiffEq.TSBasicSymplectic(); dt = 0.01f0)
            @test sol.u[end] isa typeof(p32.u0)
            @test final_err(sol, Float32.(osc_exact(1.0))) < 2.0f-5
            quiet = SciMLBase.solve(prob, PETScDiffEq.TSBasicSymplectic(); dt = 0.1, dense = false)
            @test (quiet.stats.nf, quiet.stats.nf2) == (11, 10)
        end

        @testset "a reversed span" begin
            prob = SciMLBase.SecondOrderODEProblem(osc!, [-sin(1.0)], [cos(1.0)], (1.0, 0.0))
            sol = SciMLBase.solve(prob, PETScDiffEq.TSBasicSymplectic("4"); dt = 0.01)
            @test sol.t[end] == 0.0
            @test final_err(sol, [0.0, 1.0]) < 1.0e-9
        end

        @testset "TSAlpha2 on a reversed span" begin
            alpha, damping = PETScDiffEq.TSAlpha2(), PETScDiffEq.TSAlpha2(; radius = 0.5)
            back(f!, x, t0 = 10.0) = SciMLBase.SecondOrderODEProblem(f!, [x[1]], [x[2]], (t0, 0.0))
            for alg in (alpha, damping)
                @test all(
                    o -> isapprox(o, 2; atol = 0.1),
                    orders(back(osc!, osc_exact(10.0)), alg, osc_exact(0.0)),
                )
                @test all(
                    o -> isapprox(o, 2; atol = 0.1),
                    orders(back(forced!, forced_exact(10.0)), alg, forced_exact(0.0)),
                )
            end
            pend = SciMLBase.SecondOrderODEProblem(pend!, [0.0], [2.0], (0.0, 10.0))
            ref = SciMLBase.solve(pend, PETScDiffEq.TSRK("8vr"); abstol = 1.0e-13, reltol = 1.0e-13)
            @test all(
                o -> isapprox(o, 2; atol = 0.1),
                orders(back(pend!, collect(ref.u[end])), alpha, [0.0, 2.0]),
            )
            adaptive = SciMLBase.solve(
                back(pend!, collect(ref.u[end])), alpha; abstol = 1.0e-6, reltol = 1.0e-6,
            )
            @test adaptive.retcode == SciMLBase.ReturnCode.Success
            @test issorted(adaptive.t; rev = true) && adaptive.t[end] == 0.0
            @test final_err(adaptive, [0.0, 2.0]) < 2.0e-5
            for (f!, x0, bound) in ((osc!, [0.0, 1.0], 1.0e-12), (pend!, [0.0, 2.0], 2.0e-7))
                there = SciMLBase.solve(
                    SciMLBase.SecondOrderODEProblem(f!, x0[1:1], x0[2:2], (0.0, 10.0)), alpha;
                    dt = 0.01, adaptive = false,
                )
                home = SciMLBase.solve(
                    back(f!, collect(there.u[end])), alpha; dt = 0.01, adaptive = false,
                )
                @test final_err(home, x0) < bound
            end

            # s = -t turns u'' = f(u', u, t) into w'' = f(-w', w, -s), which runs forward.
            drag!(ddu, du, u, p, t) = (@. ddu = -sin(u) - 0.3 * du + cos(2t); nothing)
            mirror!(ddu, du, u, p, t) = (@. ddu = -sin(u) + 0.3 * du + cos(2t); nothing)
            reversed = SciMLBase.SecondOrderODEProblem(drag!, [0.3], [1.2], (10.0, 0.0))
            mirrored = SciMLBase.SecondOrderODEProblem(mirror!, [-0.3], [1.2], (-10.0, 0.0))
            flip(x) = [-x.x[1]; x.x[2]]
            for (kw, mkw) in (
                    ((; dt = 0.1, adaptive = false), (; dt = 0.1, adaptive = false)),
                    ((; abstol = 1.0e-6, reltol = 1.0e-6), (; abstol = 1.0e-6, reltol = 1.0e-6)),
                    (
                        (; dt = 0.1, adaptive = false, saveat = [9.0, 7.55, 0.0], tstops = [5.55]),
                        (; dt = 0.1, adaptive = false, saveat = [-9.0, -7.55, 0.0], tstops = [-5.55]),
                    ),
                )
                b = SciMLBase.solve(reversed, damping; kw...)
                m = SciMLBase.solve(mirrored, damping; mkw...)
                @test b.retcode == SciMLBase.ReturnCode.Success
                @test b.t ≈ -m.t rtol = 1.0e-12
                @test all(i -> isapprox(flip(b.u[i]), collect(m.u[i]); rtol = 1.0e-10), eachindex(b.u))
                @test b.stats.naccept == m.stats.naccept
            end
            b = SciMLBase.solve(reversed, damping; dt = 0.1, adaptive = false)
            m = SciMLBase.solve(mirrored, damping; dt = 0.1, adaptive = false)
            @test b.t[1:3] ≈ [10.0, 9.9, 9.8]
            @test flip(b(4.44)) ≈ collect(m(-4.44)) rtol = 1.0e-10
            @test collect(b(4.44, Val{1})) ≈ [1.0, -1.0] .* collect(m(-4.44, Val{1})) rtol = 1.0e-10

            K, C = [2.0 -1.0; -1.0 2.0], [0.2 -0.1; -0.1 0.2]
            linear!(ddu, du, u, p, t) = (ddu .= .-(K * u) .- C * du .+ cos(2t); nothing)
            velocity!(du, v, u, p, t) = (du .= v; nothing)
            function linear_jac!(J, x, p, t)
                @test hasproperty(x, :x)
                fill!(J, 0.0)
                J[1:2, 1:2] .= -C
                J[1:2, 3:4] .= -K
                J[3, 1] = J[4, 2] = 1.0
                return nothing
            end
            proto = sparse([ones(2, 2) ones(2, 2); Matrix(I, 2, 2) zeros(2, 2)])
            x3 = [0.3, -0.2, 1.0, 0.5]
            second(span; kw...) = SciMLBase.SecondOrderODEProblem(
                SciMLBase.DynamicalODEFunction{true}(linear!, velocity!; kw...), x3[1:2], x3[3:4], span,
            )
            stepped(pr, alg = alpha) = SciMLBase.__solve(pr, alg; dt = 0.05, adaptive = false)
            first_order!(dx, x, p, t) = (
                dx[1:2] .= .-(K * x[3:4]) .- C * x[1:2] .+ cos(2t); dx[3:4] .= x[1:2]; nothing
            )
            exact = SciMLBase.solve(
                SciMLBase.ODEProblem(first_order!, x3, (3.0, 0.0)), Tsit5();
                abstol = 1.0e-12, reltol = 1.0e-12,
            ).u[end]
            forward = stepped(second((0.0, 3.0); jac = linear_jac!))
            fd = PETScDiffEq.TSAlpha2(; autodiff = PETScDiffEq.AutoFiniteDiff())
            dense = stepped(second((3.0, 0.0)))
            for (sol, exact_jac) in (
                    (dense, true), (stepped(second((3.0, 0.0); jac = linear_jac!)), true),
                    (stepped(second((3.0, 0.0); jac = linear_jac!, jac_prototype = proto)), true),
                    (stepped(second((3.0, 0.0); jac_prototype = proto)), true),
                    (stepped(second((3.0, 0.0)), fd), false),
                    (stepped(second((3.0, 0.0); jac_prototype = proto), fd), false),
                )
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test final_err(sol, exact) < 5.0e-3
                @test collect(sol.u[end]) ≈ collect(dense.u[end]) rtol = exact_jac ? 1.0e-10 : 1.0e-6
                # A linear system takes one Newton iteration a solve only with the right Jacobian.
                @test sol.stats.nnonliniter == forward.stats.nnonliniter
                @test (sol.stats.njacs > 0) == exact_jac
            end

            prob = back(osc!, osc_exact(2.0), 2.0)
            integ = SciMLBase.init(prob, alpha; dt = 0.1, adaptive = false)
            SciMLBase.step!(integ)
            @test integ.t ≈ 1.9 && integ.dt ≈ -0.1
            @test integ.u isa typeof(prob.u0)
            @test collect(integ.u) ≈ osc_exact(1.9) atol = 2.0e-4
            @test collect(integ(1.95)) ≈ osc_exact(1.95) atol = 1.0e-4
            @test collect(integ(1.95, Val{1})) ≈ [-cos(1.95), -sin(1.95)] atol = 3.0e-3
            @test collect(SciMLBase.get_du(integ)) ≈ [-integ.u[2], integ.u[1]]
            SciMLBase.step!(integ, -0.35, true)
            @test integ.t ≈ 1.55
            @test collect(integ.u) ≈ osc_exact(1.55) atol = 1.0e-3
            SciMLBase.set_u!(integ, 2 .* integ.u)
            SciMLBase.solve!(integ)
            @test integ.sol.t[end] == 0.0
            @test collect(integ.sol.u[end]) ≈ 2 .* osc_exact(0.0) atol = 1.0e-2
            twice = SciMLBase.solve(prob, alpha; dt = 0.1, adaptive = false)
            SciMLBase.reinit!(integ)
            @test integ.t == 2.0 && integ.u == prob.u0
            SciMLBase.solve!(integ)
            @test collect(integ.sol.u[end]) ≈ collect(twice.u[end]) rtol = 1.0e-14
            saved = SciMLBase.solve(
                prob, alpha; dt = 0.1, adaptive = false, saveat = [1.55, 0.7], tstops = [1.23],
            )
            @test saved.t == [1.55, 0.7]
            @test maximum(abs, collect(saved.u[1]) .- osc_exact(1.55)) < 1.0e-3
            @test maximum(abs, collect(saved.u[2]) .- osc_exact(0.7)) < 2.0e-3
            stops = SciMLBase.solve(prob, alpha; dt = 0.1, adaptive = false, tstops = [1.23])
            @test 1.23 in stops.t && issorted(stops.t; rev = true)
            bounce = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u.x[2][1] - 0.5, integ -> (integ.u.x[1] .*= -1; nothing),
            )
            sol = SciMLBase.solve(prob, alpha; dt = 0.01, adaptive = false, callback = bounce)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test maximum(u -> u.x[2][1], sol.u) ≈ 0.5 atol = 1.0e-12
            hit = sol.t[argmax([u.x[2][1] for u in sol.u])]
            @test hit ≈ pi / 3 atol = 1.0e-4
            kick = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t == 1.25, integ -> (integ.u.x[1] .+= 1.0; nothing),
            )
            kicked = SciMLBase.solve(
                prob, alpha; dt = 0.1, adaptive = false, callback = kick, tstops = [1.25],
            )
            at = findall(==(1.25), kicked.t)
            @test length(at) == 2
            @test kicked.u[at[2]].x[1][1] - kicked.u[at[1]].x[1][1] == 1.0
            rest = SciMLBase.solve(
                SciMLBase.remake(prob; u0 = kicked.u[at[2]], tspan = (1.25, 0.0)), alpha;
                dt = 0.1, adaptive = false,
            )
            @test collect(kicked.u[end]) ≈ collect(rest.u[end]) rtol = 1.0e-12
            seen = []
            sol = SciMLBase.solve(
                prob, alpha; dt = 0.01, adaptive = false,
                unstable_check = (dt, u, p, t) -> (push!(seen, copy(u)); u.x[2][1] > 0.5),
            )
            @test sol.retcode == SciMLBase.ReturnCode.Unstable
            @test seen == sol.u[2:end]
        end

        @testset "callbacks and the integrator: $(nameof(typeof(alg)))" for alg in (
                PETScDiffEq.TSBasicSymplectic(), PETScDiffEq.TSAlpha2(),
            )
            prob = SciMLBase.SecondOrderODEProblem(osc!, [0.0], [1.0], (0.0, 2.0))
            bounce = SciMLBase.ContinuousCallback(
                (u, t, integ) -> u.x[2][1] - 0.5, nothing, integ -> (integ.u.x[1] .*= -1; nothing),
            )
            sol = SciMLBase.solve(prob, alg; dt = 0.01, adaptive = false, callback = bounce)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test minimum(u -> u.x[2][1], sol.u) ≈ 0.5 atol = 1.0e-12
            kick = SciMLBase.DiscreteCallback(
                (u, t, integ) -> t == 1.0, integ -> (integ.u.x[1] .+= 1.0; nothing),
            )
            kicked = SciMLBase.solve(
                prob, alg; dt = 0.1, adaptive = false, callback = kick, tstops = [1.0],
            )
            at = findall(==(1.0), kicked.t)
            @test length(at) == 2
            @test kicked.u[at[2]].x[1][1] - kicked.u[at[1]].x[1][1] == 1.0
            rest = SciMLBase.solve(
                SciMLBase.remake(prob; u0 = kicked.u[at[2]], tspan = (1.0, 2.0)), alg;
                dt = 0.1, adaptive = false,
            )
            @test collect(kicked.u[end]) ≈ collect(rest.u[end]) rtol = 1.0e-12
            integ = SciMLBase.init(prob, alg; dt = 0.1, adaptive = false)
            SciMLBase.step!(integ)
            SciMLBase.set_u!(integ, 2 .* integ.u)
            SciMLBase.solve!(integ)
            twice = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false)
            @test collect(integ.sol.u[end]) ≈ 2 .* collect(twice.u[end]) rtol = 1.0e-10
            SciMLBase.reinit!(integ)
            @test integ.u == prob.u0
            SciMLBase.solve!(integ)
            @test collect(integ.sol.u[end]) ≈ collect(twice.u[end]) rtol = 1.0e-14
        end

        @testset "unstable_check gets the state: $(nameof(typeof(alg)))" for alg in (
                PETScDiffEq.TSAlpha2(), PETScDiffEq.TSRK(), PETScDiffEq.TSBasicSymplectic(),
            )
            prob = SciMLBase.SecondOrderODEProblem(osc!, [0.0], [1.0], (0.0, 2.0))
            seen = []
            sol = SciMLBase.solve(
                prob, alg; dt = 0.01, adaptive = false,
                unstable_check = (dt, u, p, t) -> (push!(seen, copy(u)); u.x[2][1] < 0.5),
            )
            @test sol.retcode == SciMLBase.ReturnCode.Unstable
            @test sol.t[end] ≈ 1.05
            @test all(u -> u isa typeof(prob.u0), seen)
            @test seen == sol.u[2:end]
        end

        @testset "erase_sol = false across a change of saving: $(nameof(typeof(alg)))" for alg in (
                PETScDiffEq.TSRK("4"), PETScDiffEq.TSBasicSymplectic("4"), PETScDiffEq.TSAlpha2(),
            )
            prob = SciMLBase.SecondOrderODEProblem(osc!, [0.0], [1.0], (0.0, 1.0))
            integ = SciMLBase.init(prob, alg; dt = 0.01, adaptive = false, saveat = 0.5)
            kept = SciMLBase.solve!(integ)
            @test !kept.dense
            SciMLBase.reinit!(
                integ, kept.u[end]; t0 = 1.0, tf = 2.0, saveat = Float64[], erase_sol = false,
            )
            sol = SciMLBase.solve!(integ)
            @test sol.retcode == SciMLBase.ReturnCode.Success
            @test sol.dense
            @test length(sol.interp.du) == length(sol.u)
            @test all(du -> du isa typeof(prob.u0), sol.interp.du)
            @test maximum(abs.(collect(sol(0.25)) .- osc_exact(0.25))) < 3.0e-4
            @test maximum(abs.(collect(sol(1.5)) .- osc_exact(1.5))) < 3.0e-5
        end

        @testset "reinit! with a flat state: $(nameof(typeof(alg)))" for alg in (
                PETScDiffEq.TSRK("4"), PETScDiffEq.TSBasicSymplectic(), PETScDiffEq.TSAlpha2(),
            )
            prob = SciMLBase.SecondOrderODEProblem(osc!, [0.0], [1.0], (0.0, 1.0))
            integ = SciMLBase.init(prob, alg; dt = 0.1, adaptive = false)
            SciMLBase.reinit!(integ, [0.5, 0.25])
            @test integ.u isa typeof(prob.u0)
            @test collect(integ.u) == [0.5, 0.25]
            fresh = SciMLBase.SecondOrderODEProblem(osc!, [0.5], [0.25], (0.0, 1.0))
            @test collect(SciMLBase.solve!(integ).u[end]) ==
                collect(SciMLBase.solve(fresh, alg; dt = 0.1, adaptive = false).u[end])
        end

        @testset "the other algorithms step the first-order form" begin
            prob = SciMLBase.SecondOrderODEProblem(osc!, [0.0], [1.0], (0.0, 1.0))
            jac!(J, x, p, t) = (@test hasproperty(x, :x); J .= [0 -1; 1 0]; nothing)
            jprob = SciMLBase.SecondOrderODEProblem(
                SciMLBase.DynamicalODEFunction{true}(osc!, (du, v, u, p, t) -> (du .= v; nothing); jac = jac!),
                [0.0], [1.0], (0.0, 1.0),
            )
            for (pr, alg) in (
                    (prob, PETScDiffEq.TSRK("5dp")), (prob, PETScDiffEq.TSRosW()),
                    (jprob, PETScDiffEq.TSImplicit("bdf"; order = 5)),
                )
                sol = SciMLBase.solve(pr, alg; abstol = 1.0e-9, reltol = 1.0e-9)
                @test sol.u[end] isa typeof(prob.u0)
                @test final_err(sol, osc_exact(1.0)) < 5.0e-8
            end
        end

        @testset "operator-valued parts" begin
            A = SciMLOperators.MatrixOperator([-1.0 0.0; 0.0 -2.0])
            B = SciMLOperators.MatrixOperator([1.0 0.0; 0.0 1.0])
            prob = SciMLBase.DynamicalODEProblem(A, B, [1.0, 1.0], [1.0, 1.0], (0.0, 1.0))
            exact = [exp.([-1.0, -2.0]); 1 .+ (1 .- exp.([-1.0, -2.0])) ./ [1.0, 2.0]]
            for alg in (PETScDiffEq.TSRK(), PETScDiffEq.TSRosW())
                sol = SciMLBase.solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test final_err(sol, exact) < 5.0e-10
            end
        end

        Sys.WORD_SIZE == 64 && @testset "a communicator of one rank" begin
            world = MPI.COMM_WORLD
            function springs!(ddu, du, u, p, t)
                n = length(u)
                for i in 1:n
                    ddu[i] = (i > 1 ? u[i - 1] : 0.0) - 2u[i] + (i < n ? u[i + 1] : 0.0)
                end
                return nothing
            end
            function springs_jac!(J, x, p, t)
                for i in 1:5, j in max(1, i - 1):min(5, i + 1)
                    J[i, 5 + j] = i == j ? -2.0 : 1.0
                end
                for i in 1:5
                    J[5 + i, i] = 1.0
                end
                return nothing
            end
            springs_proto = sparse(
                vcat([i for i in 1:5 for _ in max(1, i - 1):min(5, i + 1)], 6:10),
                vcat([5 + j for i in 1:5 for j in max(1, i - 1):min(5, i + 1)], 1:5), 1.0, 10, 10,
            )
            springs(; kw...) = SciMLBase.SecondOrderODEProblem(
                SciMLBase.DynamicalODEFunction{true}(
                    springs!, (du, v, u, p, t) -> (du .= v; nothing); kw...,
                ), zeros(5), [0.1, 0.5, 1.0, 0.5, 0.1], (0.0, 2.0),
            )
            carries = springs(; jac_prototype = springs_proto).f.jac_prototype isa SparseMatrixCSC
            for (pr, alg, serial_alg, kw) in (
                    (
                        springs(), PETScDiffEq.TSBasicSymplectic("velverlet"; comm = world),
                        PETScDiffEq.TSBasicSymplectic("velverlet"), (; dt = 0.05),
                    ),
                    (
                        springs(; jac = springs_jac!, jac_prototype = springs_proto),
                        PETScDiffEq.TSAlpha2(; comm = world), PETScDiffEq.TSAlpha2(),
                        (; abstol = 1.0e-8, reltol = 1.0e-8),
                    ),
                    (
                        springs(; jac_prototype = springs_proto), PETScDiffEq.TSAlpha2(; comm = world),
                        PETScDiffEq.TSAlpha2(; autodiff = PETScDiffEq.AutoFiniteDiff()),
                        (; dt = 0.05, adaptive = false),
                    ),
                    (
                        springs(), PETScDiffEq.TSRK("5dp"; comm = world), PETScDiffEq.TSRK("5dp"),
                        (; abstol = 1.0e-8, reltol = 1.0e-8),
                    ),
                    (
                        springs(; jac = springs_jac!, jac_prototype = springs_proto),
                        PETScDiffEq.TSImplicit("bdf"; comm = world), PETScDiffEq.TSImplicit("bdf"),
                        (; abstol = 1.0e-8, reltol = 1.0e-8),
                    ),
                )
                carries || alg isa PETScDiffEq.TSBasicSymplectic || alg isa PETScDiffEq.TSRK || continue
                sol = SciMLBase.__solve(pr, alg; kw...)
                ref = SciMLBase.__solve(pr, serial_alg; kw...)
                @test sol.retcode == SciMLBase.ReturnCode.Success
                @test sol.u[end] isa typeof(pr.u0)
                @test sol.t == ref.t
                @test maximum(maximum(abs, collect(a) - collect(b)) for (a, b) in zip(sol.u, ref.u)) <=
                    1.0e-13
            end
            @test_throws "on a DynamicalODEProblem or SecondOrderODEProblem on a communicator" SciMLBase.__solve(
                springs(; jac_prototype = springs_proto),
                PETScDiffEq.TSAlpha2(; comm = world, autodiff = PETScDiffEq.AutoForwardDiff());
                dt = 0.05,
            )
            @test_throws "SecondOrderODEProblem on MPI.COMM_SELF only" PETScDiffEq._discrete_adjoint(
                springs(; jac = springs_jac!, jac_prototype = springs_proto),
                PETScDiffEq.TSRK("4"; comm = world), PETScAdjoint(); t = [2.0],
                dgdu_discrete = (out, u, p, t, i) -> (out .= u; nothing), dt = 0.05,
                adaptive = false,
            )
            line = PETScDiffEq.PETSc.DMDA(
                PETScDiffEq.PETSc.getlib(; PetscScalar = Float64), MPI.COMM_SELF,
                (PETScDiffEq.LibPETSc.DM_BOUNDARY_NONE,), (10,), 1, 1,
            )
            @test_throws "SecondOrderODEProblem with a `dm` yet" SciMLBase.solve(
                springs(), PETScDiffEq.TSRK("5dp"; dm = line),
            )
            PETScDiffEq.PETScCompat.destroy!(line)
        end

        @testset "what these algorithms refuse" begin
            osc = SciMLBase.SecondOrderODEProblem(osc!, [0.0], [1.0], (0.0, 1.0))
            dyn = SciMLBase.DynamicalODEProblem(
                (dv, v, u, p, t) -> (dv .= -u; nothing), (du, v, u, p, t) -> (du .= v; nothing),
                [0.0], [1.0], (0.0, 1.0),
            )
            plain = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
            @test_throws "TSBasicSymplectic needs a DynamicalODEProblem" SciMLBase.solve(
                plain, PETScDiffEq.TSBasicSymplectic(); dt = 0.1,
            )
            @test_throws "TSAlpha2 needs a SecondOrderODEProblem" SciMLBase.solve(
                plain, PETScDiffEq.TSAlpha2(); dt = 0.1,
            )
            @test_throws "TSAlpha2 needs a SecondOrderODEProblem, where u'" SciMLBase.solve(
                dyn, PETScDiffEq.TSAlpha2(); dt = 0.1,
            )
            @test SciMLBase.solve(dyn, PETScDiffEq.TSBasicSymplectic(); dt = 0.1).retcode ==
                SciMLBase.ReturnCode.Success
            @test_throws ArgumentError PETScDiffEq.TSBasicSymplectic("5")
            @test_throws ArgumentError PETScDiffEq.TSAlpha2(; radius = 1.5)
            @test_throws "use TSAlpha2" PETScDiffEq.TSGeneric("alpha2")
            @test_throws "use TSBasicSymplectic" PETScDiffEq.TSGeneric("basicsymplectic")
            @test_throws "an option cannot change it to `rk`" SciMLBase.solve(
                osc, PETScDiffEq.TSBasicSymplectic("velverlet", ["-ts_type", "rk"]); dt = 0.1,
            )
            @test_throws "`abstol` has length 3, but the state has 2" SciMLBase.solve(
                osc, PETScDiffEq.TSAlpha2(); abstol = [1.0e-6, 1.0e-6, 1.0e-6],
            )
            mass = SciMLBase.SecondOrderODEProblem(
                SciMLBase.DynamicalODEFunction{true}(
                    osc!, (du, v, u, p, t) -> (du .= v; nothing); mass_matrix = Diagonal([2.0, 1.0]),
                ), [0.0], [1.0], (0.0, 1.0),
            )
            for alg in (
                    PETScDiffEq.TSAlpha2(), PETScDiffEq.TSBasicSymplectic(), PETScDiffEq.TSRK(),
                    PETScDiffEq.TSRosW(),
                )
                @test_throws "does not take a mass matrix" SciMLBase.solve(mass, alg; dt = 0.1)
            end
            sparse_proto = SciMLBase.SecondOrderODEProblem(
                SciMLBase.DynamicalODEFunction{true}(
                    osc!, (du, v, u, p, t) -> (du .= v; nothing); jac_prototype = sparse(ones(2, 2)),
                ), [0.0], [1.0], (0.0, 1.0),
            )
            @test SciMLBase.__solve(sparse_proto, PETScDiffEq.TSAlpha2(); dt = 0.1).retcode ==
                SciMLBase.ReturnCode.Success
            if sparse_proto.f.jac_prototype isa SparseMatrixCSC
                @test_throws "TSAlpha2 takes a `jac_prototype` of the first-order system" SciMLBase.__solve(
                    SciMLBase.SecondOrderODEProblem(
                        SciMLBase.DynamicalODEFunction{true}(
                            osc!, (du, v, u, p, t) -> (du .= v; nothing); jac_prototype = sparse(ones(1, 1)),
                        ), [0.0], [1.0], (0.0, 1.0),
                    ), PETScDiffEq.TSAlpha2(); dt = 0.1,
                )
            end
            scalars = SciMLBase.SecondOrderODEProblem((du, u, p, t) -> -u, 0.0, 1.0, (0.0, 1.0))
            @test_throws "are both vectors" SciMLBase.solve(
                scalars, PETScDiffEq.TSBasicSymplectic(); dt = 0.1,
            )
            @test_throws "PETSc has no adjoint for TSBasicSymplectic" PETScDiffEq._discrete_adjoint(
                osc, PETScDiffEq.TSBasicSymplectic(), PETScAdjoint(); t = [0.0, 1.0],
                dgdu_discrete = (out, u, p, t, i) -> (out .= u; nothing), dt = 0.1, adaptive = false,
            )
        end
    end

    @testset "MPI" begin
        # MPI_RANKS, such as 2 or 1,3, narrows the rank counts, so CI runs each in its own job.
        listed = isempty(GROUP_RANKS) ? get(ENV, "MPI_RANKS", "") : GROUP_RANKS
        rank_counts = isempty(listed) ? [1, 2, 3] : tryparse.(Int, split(listed, ','))
        allunique(rank_counts) && rank_counts ⊆ 1:3 ||
            error("MPI_RANKS is a comma-separated list of distinct rank counts from 1, 2 and 3, not $listed")
        if Sys.WORD_SIZE == 64 && !Sys.iswindows()
            dir = joinpath(@__DIR__, "mpi")
            julia = Base.julia_cmd()
            root = dirname(@__DIR__)
            # Pkg.test leaves the stdlib out of the load path.
            setup = "push!(LOAD_PATH, \"@stdlib\"); using Pkg; " *
                "Pkg.develop(path = $(repr(root))); Pkg.instantiate()"
            run(`$julia --project=$dir -e $setup`)
            # exit.jl leaves integrators alive for the exit hooks, so it runs last.
            launches = (
                (
                    1, [
                        "explicit.jl", "implicit.jl", "adjoint.jl", "dm.jl", "types.jl",
                        "second_order.jl", "exit.jl",
                    ],
                ),
                (2, ["ensemble.jl"]),
            )
            for np in rank_counts, (threads, scripts) in launches
                rank_cmd = `$julia --threads=$threads --project=$dir $(joinpath(dir, "all.jl")) $scripts`
                cmd = `$(MPI.mpiexec()) -n $np $rank_cmd`
                proc = run(pipeline(cmd; stdout, stderr); wait = false)
                # A rank left waiting in a collective hangs rather than fails, and a rank
                # killed there can hang again in its exit hooks.
                timer = Timer(3600) do _
                    kill(proc)
                    sleep(30)
                    process_running(proc) && kill(proc, Base.SIGKILL)
                end
                wait(proc)
                close(timer)
                @test success(proc)
            end
        end
    end
end
# The 32-bit jobs run near their address space, so each run logs its peak.
Sys.islinux() &&
    foreach(println, filter(startswith(r"VmPeak|VmHWM"), readlines("/proc/self/status")))
Test.finish(ALL_TESTS)
