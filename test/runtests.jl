using PETScDiffEq
using SciMLBase
using Test

decay!(du, u, p, t) = (du[1] = -u[1]; nothing)
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
