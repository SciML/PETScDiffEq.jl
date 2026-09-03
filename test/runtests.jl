using PETScDiffEq
using SciMLBase
using Logging
using SparseArrays
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
            prob, alg; dt = 0.1, tstops = [0.55],
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
