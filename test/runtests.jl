using PETScDiffEq
using SciMLBase
using LinearAlgebra
using Logging
using SparseArrays
using DiffEqCallbacks
using DiffEqCallbacks: PresetTimeCallback
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
# Deliberately omits the structural zero at (1,1): u1' = u2 does not depend
# on u1 at all, so a correct sparsity pattern must not reserve that entry.
const OSCILLATOR_PROTOTYPE = sparse([1, 2, 2], [2, 1, 2], ones(3), 2, 2)

@testset "PETScDiffEq.jl" begin
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
        # u' = (a+b)u split into an implicit a*u and an explicit b*u.
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

        # The stiff half has to land on the implicit side: dt * 2000 is far
        # outside any explicit stability region, so the reversed split diverges.
        bounded = PETScDiffEq.TSARKIMEX(
            "3", ["-ts_adapt_type", "none", "-ts_max_snes_failures", "1"],
        )
        @test SciMLBase.solve(halves(-2000.0, -1.0), bounded; dt = 0.01).retcode ==
            SciMLBase.ReturnCode.Success
        @test SciMLBase.solve(halves(-1.0, -2000.0), bounded; dt = 0.01).retcode !=
            SciMLBase.ReturnCode.Success
    end

    @testset "an integrator dropped part-way still exits cleanly" begin
        # PETSc objects freed after MPI shuts down abort the process, so this can
        # only be seen from the outside: the exit code is the assertion.
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
        # An explicit method runs no nonlinear solve at all.
        explicit = SciMLBase.solve(
            prob, PETScDiffEq.TSRK("5dp"); dt = 0.01, adaptive = false,
        )
        @test explicit.stats.nnonliniter == 0
    end

    @testset "maxiters that no PetscInt can hold" begin
        # SciML spells "no limit" as typemax(Int), which need not fit a PetscInt.
        # Which of the two is wider depends on the platform, so clamp to the smaller.
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
        # The index vectors PETSc is handed are typed at precompile time from a
        # constant, so a library with a different index width would corrupt silently.
        @test PETScDiffEq.PETSc.inttype(lib) === PETScDiffEq.LibPETSc.PetscInt
    end

    @testset "which side of the bracket the root lands on" begin
        rootprob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        # The condition falls through zero, so the left end of the bracket is
        # still positive while the right end has already crossed.
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
        # Both ends are resolved to machine precision, not merely to the step.
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
        @test integ.tstops == [0.5]
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
            integ = SciMLBase.init(prob, alg; dt = 0.1, saveat = 0.5)
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

    @testset "a solve that never uses the Jacobian is called out" begin
        prob = SciMLBase.ODEProblem(
            SciMLBase.ODEFunction(decay!; jac = decay_jac!), [1.0], (0.0, 1.0),
        )
        plain = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))

        # "glee" is an explicit method.
        quiet = SciMLBase.solve(
            prob, PETScDiffEq.TSGeneric("glee"); dt = 0.1, adaptive = false,
        )
        @test quiet.u[end] == [1.0]
        @test quiet.stats.njacs == 0
        @test_logs (:warn,) match_mode = :any SciMLBase.solve(
            prob, PETScDiffEq.TSGeneric("glee"); dt = 0.1, adaptive = false,
        )

        @testset "and the same type is fine when told it is explicit" begin
            sol = SciMLBase.solve(
                plain, PETScDiffEq.TSGeneric("glee"; explicit = true); dt = 0.1,
                adaptive = false,
            )
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
            # PETSc defaults to order 2, which costs steps against a higher order.
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
        # u1' = -u1 with the algebraic constraint u2 = u1, so both are exp(-t).
        # Integrating the second row as an ODE instead would give cosh(1).
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
                PETScDiffEq.TSDAE(); dt = 1.0e-3, adaptive = false,
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
            # u' = -cbrt(u), so dG/du' is 3du^2 rather than a constant.
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
            # An ODE algorithm and a DAEProblem are an incompatible pairing.
            @test_throws Exception SciMLBase.solve(
                prob, PETScDiffEq.TSRK("5dp"); dt = 1.0e-3,
            )
            # A residual gives no derivative for a Hermite interpolant.
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
            # Each trajectory carries its own decay rate.
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
        @test sol.retcode == SciMLBase.ReturnCode.Failure
        @test sol.t[end] > 0.0

        integ = SciMLBase.init(
            stiff, PETScDiffEq.TSImplicit("beuler"); dt = 1.0, adaptive = false,
        )
        n = 0
        while !SciMLBase.done(integ) && n < 100
            SciMLBase.step!(integ)
            n += 1
        end
        # A step PETSc cannot take returns without advancing, which used to spin
        # this loop forever.
        @test SciMLBase.done(integ)
        @test integ.sol.retcode == SciMLBase.ReturnCode.Failure

        # Choosing an explicit type through the implicit path is a misuse, and
        # stays an exception rather than a quiet failure code.
        plain = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        @test_throws Exception SciMLBase.solve(
            plain, PETScDiffEq.TSGeneric("euler"); dt = 0.01, adaptive = false,
        )

        # A user's own exception still reaches the caller unchanged.
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
            # What the callbacks actually ask of it.
            @test integ.f isa SciMLBase.AbstractODEFunction
            @test SciMLBase.isinplace(integ.f)
            @test integ.opts.abstol isa Real
            @test SciMLBase.get_du(integ)[1] ≈ -integ.u[1]
            mid = (integ.tprev + integ.t) / 2
            @test abs(integ(mid)[1] - exp(-mid)) < 1.0e-6
            # PETSc interpolates only inside the step it just took.
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
                SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0)), PETScDiffEq.TSIRK();
                dt = 0.1,
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
        for alg in (
                PETScDiffEq.TSImplicit("bdf"), PETScDiffEq.TSRK("5dp"),
                PETScDiffEq.TSRosW("ra34pw2"),
                # An arbitrary named type may or may not adapt.
                PETScDiffEq.TSGeneric("alpha"),
            )
            @test_logs min_level = Logging.Warn SciMLBase.solve(
                prob, alg; dt = 0.05, reltol = 1.0e-6,
            )
        end
    end

    @testset "the exported names carry a docstring" begin
        # `Base.doc` is not available on every supported version, and `@doc`
        # reports a docstring even for a name that has none.
        lines = split(read(joinpath(@__DIR__, "..", "src", "PETScDiffEq.jl"), String), '\n')
        exported = filter(!=(:PETScDiffEq), names(PETScDiffEq))
        @test length(exported) == 8
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
        for st in ("3bs", "5dp", "5f", "5bs")
            @test runs(PETScDiffEq.TSRK(st))
        end
        for st in ("2m", "ra34pw2", "ra3pw", "sandu3")
            @test runs(PETScDiffEq.TSRosW(st))
        end
        for st in ("2e", "3", "4", "5")
            @test runs(PETScDiffEq.TSARKIMEX(st))
        end
        for st in ("beuler", "cn", "theta", "bdf")
            @test runs(PETScDiffEq.TSImplicit(st))
        end
        @test runs(PETScDiffEq.TSGeneric("alpha"))
        # PETSc wants the right-hand side for these, not the implicit residual.
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
            # exp(-2t) reaches 1/2 first, at half the time exp(-t) does.
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
            # Lifted at its own earlier crossing, the second component is at 3/4
            # when the first triggers, against 1/4 without the lift.
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
            # Only the stiff second component carries error worth controlling.
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
            # The buffer is reused every step, so an entry the new Jacobian does
            # not have must not keep the old one's value.
            PETScDiffEq._copy_jac!(J, sparse([1, 2], [1, 2], [3.0, 4.0], 2, 2))
            @test J[1, 1] == 3.0
            @test J[2, 2] == 4.0
            @test J[1, 2] == 0.0

            # A dense Jacobian's zeros land where the prototype declares nothing,
            # so they have to be skipped rather than written.
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

            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1)
            SciMLBase.step!(integ)
            SciMLBase.reinit!(integ)
            @test abs(SciMLBase.solve!(integ).u[end][1] - exp(-1)) < 1.0e-5
        end
    end

    @testset "ContinuousCallback" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 2.0))
        # exp(-t) hits 1/2 at log 2; the affect! lifts it to 3/2, which hits 1/2 at log 6.
        jump(hits) = SciMLBase.ContinuousCallback(
            (u, t, integ) -> u[1] - 0.5, integ -> (push!(hits, integ.t); integ.u[1] += 1.0),
        )
        after(t) = 1.5 * exp(-(t - log(6.0)))

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
            # Each flight is 0.9 of the one before it, the restitution coefficient.
            gaps = diff(bounces)
            @test all(isapprox(0.9; atol = 1.0e-6), gaps[2:end] ./ gaps[1:(end - 1)])
            @test minimum(u[1] for u in sol.u) > -1.0e-10
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
            # An affect! that changes nothing must still fire once per crossing.
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
            # exp(-t) is first below 1/2 at the step ending at 0.7.
            @test hits[1] > log(2.0)
            @test abs(hits[1] - 0.7) < 1.0e-8
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
            @test !SciMLBase.has_tstop(integ)
            SciMLBase.add_tstop!(integ, 0.33)
            SciMLBase.add_tstop!(integ, 0.33)
            SciMLBase.add_tstop!(integ, 0.15)
            @test SciMLBase.has_tstop(integ)
            @test SciMLBase.first_tstop(integ) == 0.15
            @test integ.tstops == [0.15, 0.33]
            @test SciMLBase.pop_tstop!(integ) == 0.15
            while integ.t < 0.33
                SciMLBase.step!(integ)
            end
            @test integ.t == 0.33
            @test !SciMLBase.has_tstop(integ)
            @test_throws ArgumentError SciMLBase.add_tstop!(integ, 0.1)
            @test_throws ArgumentError SciMLBase.add_tstop!(integ, 1.5)
            SciMLBase.terminate!(integ)
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
        # +5000 flips the sign of shift*I - J, so Newton walks away from the root.
        stiff_wrong_jac!(J, u, p, t) = (J[1, 1] = 5000.0; nothing)
        alg = PETScDiffEq.TSARKIMEX("3", ["-ts_adapt_type", "none"])
        tspan = (0.0, 0.1)
        split(f1) = SciMLBase.SplitODEProblem(f1, forcing!, [1.0], tspan)

        plain = SciMLBase.solve(split(stiff!), alg; dt = 1.0e-3)
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
            # The dense block is handed to PETSc row-major, so a transposition
            # would survive any symmetric test. du1 = -u1 + 3u2, du2 = -2u2.
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
            @test vec.t ≈ [0.0, 0.3, 0.7, 1.0]
            # interpolated points must carry the solver's accuracy, not a
            # linear fallback between stored steps
            for sol in (scalar, vec), i in eachindex(sol.t)
                @test abs(sol.u[i][1] - exact(sol.t[i])) < 1.0e-7
            end
        end

        @testset "save_start and save_end" begin
            nostart = SciMLBase.solve(
                prob, alg; dt = 0.1, saveat = [0.3, 0.7], save_start = false,
            )
            @test nostart.t ≈ [0.3, 0.7, 1.0]
            # Without saveat the first saved point is t0 itself, which is the
            # only case where it has to be dropped.
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

    @testset "Solution statistics" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        sol = SciMLBase.solve(
            prob, PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"]); dt = 0.1,
        )
        @test sol.stats !== nothing
        @test sol.stats.nf > 0
        @test sol.stats.naccept == 10
        @test sol.stats.nreject == 0
    end

    @testset "Unsupported keywords warn rather than being dropped" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        alg = PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"])
        @test_logs (:warn,) match_mode = :any SciMLBase.solve(
            prob, alg; dt = 0.1, isoutofdomain = (u, p, t) -> false,
        )
        @test_logs (:warn,) match_mode = :any SciMLBase.solve(
            prob, alg; dt = 0.1, d_discontinuities = [0.5],
        )
        @test_logs min_level = Logging.Warn SciMLBase.solve(prob, alg; dt = 0.1)
    end

    @testset "No Jacobian buffer is allocated when none is used" begin
        # An explicit method never forms a Jacobian, so a solve must not pay
        # for a dense n-by-n buffer. At n = 500 that buffer would be ~1.9 MiB.
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
            # u1' = -u1 with the algebraic constraint 0 = u2 - u1, so both
            # components follow exp(-t). Integrating the second row as an ODE
            # instead would give cosh(1).
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
            # M u' = -u with M = [2 0.5; 0 1] has the closed form
            # u1 = 1.5exp(-t/2) - 0.5exp(-t), u2 = exp(-t).
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
            # The residual multiplies by M directly, so dropping M's off-diagonal
            # leaves the answer right and only W wrong, and Newton still finds the
            # root. Under a strong coupling it stops finding it at all.
            # M = [1 5; 0 1] gives u1 = (5t + 1)exp(-t), u2 = exp(-t).
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
            @test_throws ArgumentError SciMLBase.solve(
                offdiag, PETScDiffEq.TSImplicit("bdf"); dt = 0.005,
            )
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
        # the default controller is PETSc's own, so dt is only the first step
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
        # Each subtype here differs in order from its family's default.
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
            # A silent no-op here spins SciMLBase's generic loop forever, since
            # it advances only on `integ.t` and breaks only on a bad retcode.
            @test_throws ArgumentError SciMLBase.step!(integ, 0.3)
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
            # The accuracy bound is per method: 5dp resolves exp(-t) to 1e-7 at
            # dt = 0.1, second-order BDF only to about 1e-3.
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
            # exp(-t) until the jump, then (exp(-0.5) + 1) exp(-(t - 0.5)) after it
            expected(t) = t < 0.5 ? exp(-t) : (exp(-0.5) + 1.0) * exp(-(t - 0.5))

            @testset "affect! changes the state PETSc integrates from" begin
                for alg in (PETScDiffEq.TSRK("5dp"), PETScDiffEq.TSImplicit("bdf"))
                    fired[] = false
                    sol = SciMLBase.solve(prob, alg; dt = 0.1, adaptive = false, callback = jump)
                    @test sol.retcode == SciMLBase.ReturnCode.Success
                    @test fired[]
                    @test abs(sol.u[end][1] - expected(1.0)) < 5.0e-3
                    # save_positions defaults to (true, true): both sides of the jump
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
                # no TSRestartStep, so BDF keeps its history and the answer is unchanged
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
                SciMLBase.savevalues!(integ)
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
                # fixed dt over a unit span, whatever the tolerance
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

    @testset "Requested subtypes actually take effect" begin
        # Each of these differs in order from its family's PETSc default.
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
        # rk defaults to 3bs (order 3); asking for 5dp must give order 5
        @test measured_order(
            PETScDiffEq.TSRK("5dp", ["-ts_adapt_type", "none"]),
        ) > 4.5
        @test measured_order(
            PETScDiffEq.TSRK("3bs", ["-ts_adapt_type", "none"]),
        ) < 3.5
        # theta defaults to 0.5 (order 2); theta = 1 is backward Euler, order 1
        @test measured_order(
            PETScDiffEq.TSImplicit("theta", 1.0, ["-ts_adapt_type", "none"]),
        ) < 1.5
        @test measured_order(
            PETScDiffEq.TSImplicit("theta", 0.5, ["-ts_adapt_type", "none"]),
        ) > 1.8
    end

    @testset "Input validation" begin
        prob = SciMLBase.ODEProblem(decay!, [1.0], (0.0, 1.0))
        @test_throws ArgumentError SciMLBase.solve(prob, PETScDiffEq.TSRK("5dp"))
        @test_throws ArgumentError SciMLBase.solve(prob, PETScDiffEq.TSRosW("ra34pw2"))
        @test_throws ArgumentError SciMLBase.solve(prob, PETScDiffEq.TSImplicit("beuler"))
        @test_throws ArgumentError SciMLBase.solve(prob, PETScDiffEq.TSARKIMEX("3"))
        @test_throws ArgumentError SciMLBase.solve(prob, PETScDiffEq.TSGeneric("alpha"))
        @test_throws ArgumentError SciMLBase.solve(
            SciMLBase.ODEProblem(decay!, [1.0], (1.0, 0.0)),
            PETScDiffEq.TSRK("5dp"); dt = 0.1,
        )
    end
end
