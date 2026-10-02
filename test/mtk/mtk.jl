using ModelingToolkit, PETScDiffEq, SciMLBase, NonlinearSolve, Test
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters g = 9.81 L = 1.0
@variables x(t) y(t) λ(t)
@mtkcompile pendulum = System([D(D(x)) ~ λ * x, D(D(y)) ~ λ * y - g, x^2 + y^2 ~ L^2], t)
const TOL = (abstol = 1.0e-8, reltol = 1.0e-8)
# Rodas5P at 1e-13 from the consistent start x = 0.6, y = -0.8 at rest.
const X_END, Y_END = 0.38425119502492877, -0.9232285844371989

@testset "initial equations make an SCCNonlinearProblem" begin
    prob = ODEProblem(
        pendulum, [x => 0.6, D(y) => 0.0], (0.0, 0.3);
        guesses = [y => -0.5, λ => 1.0, D(x) => 0.0],
    )
    @test prob.f.initialization_data.initializeprob isa SciMLBase.SCCNonlinearProblem
    @test_throws SciMLBase.CheckInitFailureError solve(
        prob, TSRosW(); initializealg = SciMLBase.CheckInit(),
    )
    for (alg, err) in ((TSRosW(), 1.0e-8), (TSImplicit("bdf"), 5.0e-5))
        sol = solve(prob, alg; TOL...)
        @test sol.retcode == ReturnCode.Success
        @test sol[y][1] ≈ -0.8 atol = 1.0e-12
        @test sol[λ][1] ≈ -0.8 * 9.81 atol = 1.0e-12
        @test sol[x][end] ≈ X_END atol = err
        @test sol[y][end] ≈ Y_END atol = err
        integ = init(prob, alg; TOL...)
        step!(integ)
        SciMLBase.initialize_dae!(integ)
        @test integ[x] ≈ 0.6 atol = 1.0e-12
        @test integ[y] ≈ -0.8 atol = 1.0e-12
        reinit!(integ)
        @test integ[y] ≈ -0.8 atol = 1.0e-12
        @test solve!(integ)[x][end] ≈ X_END atol = err
    end
end

@testset "a parameter the initialization solves for" begin
    @variables a(t) b(t)
    @parameters total k = 1.0
    @mtkcompile sys = System(
        [D(a) ~ -k * a, 0 ~ total^3 + total - a - b], t; guesses = [total => 1.0],
    )
    prob = ODEProblem(sys, [a => 1.0, b => 9.0, total => missing], (0.0, 1.0))
    @test prob.ps[total] == 1
    for (alg, err) in ((TSRosW(), 1.0e-7), (TSImplicit("bdf"), 5.0e-5))
        sol = solve(prob, alg; TOL...)
        @test sol.retcode == ReturnCode.Success
        @test sol.ps[total] ≈ 2 atol = 1.0e-12
        @test sol[a][end] ≈ exp(-1) atol = err
        @test sol[b][end] ≈ 10 - exp(-1) atol = err
        @test init(prob, alg; TOL...).ps[total] ≈ 2 atol = 1.0e-12
    end
end

@testset "a system that is not square takes an nlsolve" begin
    under = ODEProblem(
        pendulum, [x => 0.6], (0.0, 0.3); warn_initialize_determined = false,
        guesses = [y => -0.5, λ => 1.0, D(x) => 0.0, D(y) => 0.0],
    )
    over = ODEProblem(
        pendulum, [x => 0.6, y => -0.8, D(x) => 0.0], (0.0, 0.3); guesses = [λ => 1.0],
        warn_initialize_determined = false,
    )
    @test_throws "3 equations for 4 unknowns" solve(under, TSRosW())
    @test_throws "4 equations for 3 unknowns" solve(over, TSRosW())
    for (prob, nlsolve) in ((under, FastShortcutNLLSPolyalg()), (over, GaussNewton()))
        sol = solve(prob, TSRosW(); initializealg = SciMLBase.OverrideInit(; nlsolve), TOL...)
        @test sol.retcode == ReturnCode.Success
        @test sol[y][1] ≈ -0.8 atol = 1.0e-10
        @test sol[λ][1] ≈ -0.8 * 9.81 atol = 1.0e-10
        @test sol[x][end] ≈ X_END atol = 1.0e-8
    end
end
