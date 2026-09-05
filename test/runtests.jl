using PETScDiffEq
using SciMLBase
using LinearAlgebra
using Logging
using SciMLOperators
using SparseArrays
using DiffEqCallbacks: PresetTimeCallback
using Test

decay!(du, u, p, t) = (
    @inbounds for i in eachindex(du)
        du[i] = -u[i]
    end; nothing
)
decay_oop(u, p, t) = -u

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
            # The kept points saved no derivatives; a Hermite interpolant needs one each.
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

        @test_throws Exception SciMLBase.solve(
            split(SciMLBase.ODEFunction(stiff!; jac = stiff_wrong_jac!)), alg; dt = 1.0e-3,
        )

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
            @test_throws Exception SciMLBase.solve(
                wrong_prob, PETScDiffEq.TSImplicit("cn"); dt = 0.05,
            )
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
            @test_throws Exception SciMLBase.solve(
                wrong_prob, PETScDiffEq.TSImplicit("cn"); dt = 0.05,
            )
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
            prob, alg; dt = 0.1, save_idxs = [1],
        )
        @test_logs (:warn,) match_mode = :any SciMLBase.solve(
            prob, alg; dt = 0.1, save_idxs = [1],
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
        # Each non-default subtype has a different order from its family's
        # default, so a setter that silently did nothing would fail here.
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
            SciMLBase.step!(integ)
            @test integ.t ≈ 0.2
        end

        @testset "stepping past the end is a no-op" begin
            integ = SciMLBase.init(prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, adaptive = false)
            SciMLBase.solve!(integ)
            t_end = integ.t
            SciMLBase.step!(integ)
            @test integ.t == t_end
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

            @testset "a ContinuousCallback is rejected up front" begin
                cc = SciMLBase.ContinuousCallback((u, t, integ) -> u[1] - 0.5, integ -> nothing)
                @test_throws ArgumentError SciMLBase.init(
                    prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = cc,
                )
                @test_throws ArgumentError SciMLBase.solve(
                    prob, PETScDiffEq.TSRK("5dp"); dt = 0.1, callback = SciMLBase.CallbackSet(cc, jump),
                )
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
        # Each of these differs in order from its family's PETSc default, so a
        # dropped subtype call shows up as the wrong convergence order.
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
            SciMLBase.ODEProblem(decay_oop, [1.0], (0.0, 1.0)),
            PETScDiffEq.TSRK("5dp"); dt = 0.1,
        )
        @test_throws ArgumentError SciMLBase.solve(
            SciMLBase.ODEProblem(decay!, [1.0], (1.0, 0.0)),
            PETScDiffEq.TSRK("5dp"); dt = 0.1,
        )
    end
end
