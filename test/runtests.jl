using PETScDiffEq
using SciMLBase
using LinearAlgebra
using Logging
using SparseArrays
using DiffEqCallbacks
using DiffEqCallbacks: PresetTimeCallback
using MPI
using Test

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

# As one block these testsets take Julia 1.12 about an hour to compile, so each stands alone.
macro each_toplevel(block)
    return esc(Expr(:toplevel, block.args...))
end

const ALL_TESTS = Test.DefaultTestSet("PETScDiffEq.jl")
Test.push_testset(ALL_TESTS)
@each_toplevel begin
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

        @test_throws ArgumentError SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"); dt = 0.01)
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
        @test every_other[] == 20
        @test capped.stats.naccept == 10 == length(capped.t) - 1

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
            runaway = SciMLBase.ODEProblem(square!, [1.0], (0.0, 2.0))
            over = @test_logs (:warn, r"floating point exception") SciMLBase.solve(
                runaway, PETScDiffEq.TSRK("5dp");
                dt = 0.01, abstol = 1.0e-10, reltol = 1.0e-10,
            )
            @test over.retcode == SciMLBase.ReturnCode.Unstable
            @test over.t[end] < 2.0
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
                @test 0.5 - plain.t[end] < 1.0e-12
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
            plain = @test_logs (:warn, r"floating point exception") SciMLBase.solve(
                runaway, PETScDiffEq.TSRK("5dp"),
            )
            @test plain.retcode == SciMLBase.ReturnCode.Unstable
            @test plain.t == sol.t
            @test plain.u == sol.u
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
                SciMLBase.ReturnCode.ConvergenceFailure
            @test SciMLBase.solve!(SciMLBase.init(breaks, bdf; dt = 0.1, dtmin = 0.1)).retcode ==
                SciMLBase.ReturnCode.ConvergenceFailure

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
            @test wrong.retcode == SciMLBase.ReturnCode.Success
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
        @test length(exported) == 10
        for n in exported
            i = findfirst(l -> occursin(Regex("^(mutable )?struct \\Q$(n)\\E\\b"), l), lines)
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
                @test PETScDiffEq._adapts(alg) == (adapt_type == "basic")
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
            @test SciMLBase.get_proposed_dt(integ) > 0
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
                (
                    "a trajectory of states only", TSRK("4"), (0.0, 1.0), forward_t,
                    (sensealg = ["-ts_trajectory_solution_only", "1"],),
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
            for p in (nothing, SciMLBase.NullParameters(), Float64[])
                prob = SciMLBase.ODEProblem(
                    SciMLBase.ODEFunction(g!; jac = g_jac!), copy(u0), (0.0, 1.0), p,
                )
                du0, dp = grad(prob, TSRK("4"))
                function loss(u)
                    sol = SciMLBase.solve(
                        SciMLBase.remake(prob; u0 = u), TSRK("4");
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
                    alg in (TSRK("4"), TSImplicit("beuler", exact), TSImplicit("cn", exact))
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
            runs(type) = "PETScAdjoint supports PETSc's rk, beuler and cn as this package " *
                "drives them, but this solve runs `$type`"
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
                        "PETSc has no adjoint for TSImplicit(\"bdf\")",
                        () -> grad(prob, TSImplicit("bdf")),
                    ),
                    ("PETScAdjoint does not support TSARKIMEX", () -> grad(prob, TSARKIMEX())),
                    (
                        "PETScAdjoint has not been verified on TSImplicit(\"theta\")",
                        () -> grad(prob, TSImplicit("theta", 0.7)),
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
                        "PETScAdjoint supports an ODEProblem, not a DAEProblem or SplitODEProblem",
                        () -> grad(
                            SciMLBase.SplitODEProblem(
                                SciMLBase.ODEFunction(adj_f!; jac = adj_jac!),
                                SciMLBase.ODEFunction(adj_f!), copy(u0), (0.0, 1.0), p0,
                            ),
                            TSARKIMEX(),
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
                        "an adaptive solve steps onto no time but tspan's ends",
                        () -> grad(prob, TSRK("5dp"); t = [0.5], adaptive = true),
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

    @testset "MPI" begin
        if Sys.WORD_SIZE == 64 && !Sys.iswindows()
            dir = joinpath(@__DIR__, "mpi")
            julia = Base.julia_cmd()
            root = dirname(@__DIR__)
            # Pkg.test leaves the stdlib out of the load path.
            setup = "push!(LOAD_PATH, \"@stdlib\"); using Pkg; " *
                "Pkg.develop(path = $(repr(root))); Pkg.instantiate()"
            run(`$julia --project=$dir -e $setup`)
            for script in ("explicit.jl", "implicit.jl", "exit.jl"), np in (1, 2, 3)
                cmd = `$(MPI.mpiexec()) -n $np $julia --project=$dir $(joinpath(dir, script))`
                proc = run(pipeline(cmd; stdout, stderr); wait = false)
                # A rank left waiting in a collective hangs rather than fails, and a rank
                # killed there can hang again in its exit hooks.
                timer = Timer(900) do _
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
Test.pop_testset()
Test.finish(ALL_TESTS)
