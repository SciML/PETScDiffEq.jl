using PETScDiffEq, SciMLBase, SciMLSensitivity, OrdinaryDiffEqTsit5, Zygote
using LinearAlgebra, Test

function f!(du, u, p, t)
    du[1] = -p[1] * u[1] + p[2] * u[1] * u[2]
    du[2] = p[3] * u[1] - p[4] * u[2]^2 + p[1] * sin(t)
    return nothing
end
function jac!(J, u, p, t)
    J[1, 1] = -p[1] + p[2] * u[2]
    J[1, 2] = p[2] * u[1]
    J[2, 1] = p[3]
    J[2, 2] = -2 * p[4] * u[2]
    return nothing
end
function paramjac!(pJ, u, p, t)
    fill!(pJ, 0.0)
    pJ[1, 1] = -u[1]
    pJ[1, 2] = u[1] * u[2]
    pJ[2, 1] = sin(t)
    pJ[2, 3] = u[1]
    pJ[2, 4] = -u[2]^2
    return nothing
end
dg!(out, u, p, t, i) = (out .= u; nothing)
g(u, p, t) = sum(abs2, u) / 2 + p[2] * u[1] * u[2] + p[1]^2 * t
function gu!(out, u, p, t)
    out[1] = u[1] + p[2] * u[2]
    out[2] = u[2] + p[2] * u[1]
    return nothing
end
function gp!(out, u, p, t)
    fill!(out, 0.0)
    out[1] = 2 * p[1] * t
    out[2] = u[1] * u[2]
    return nothing
end
relerr(a, b) = norm(a - b) / norm(b)
flat(r) = vcat(r[1], vec(r[2]))

const U0 = [1.0, 0.5]
const P0 = [0.7, 0.3, 0.4, 0.2]
const TS = collect(0.0:0.1:1.0)
const EXACT = ["-snes_rtol", "1e-13", "-snes_atol", "1e-15", "-ksp_type", "preonly", "-pc_type", "lu"]
prob = ODEProblem(ODEFunction(f!; jac = jac!, paramjac = paramjac!), U0, (0.0, 1.0), P0)

function stiff!(du, u, p, t)
    du[1] = -p[1] * u[1]
    du[2] = -p[4] * u[2]^2 + p[1] * sin(t)
    return nothing
end
function rest!(du, u, p, t)
    du[1] = p[2] * u[1] * u[2]
    du[2] = p[3] * u[1]
    return nothing
end
parts = SplitODEProblem(stiff!, rest!, U0, (0.0, 1.0), P0)

@testset "PETScAdjoint through SciMLSensitivity" begin
    @test Base.get_extension(PETScDiffEq, :PETScDiffEqSciMLSensitivityExt) !== nothing

    @testset "adjoint_sensitivities reaches the driver" begin
        sol = solve(prob, TSRK("4"); dt = 0.01, adaptive = false, saveat = TS)
        via = adjoint_sensitivities(
            sol, TSRK("4"); sensealg = PETScAdjoint(),
            t = TS, dgdu_discrete = dg!, dt = 0.01, adaptive = false,
        )
        direct = PETScDiffEq._discrete_adjoint(
            prob, TSRK("4"), PETScAdjoint();
            t = TS, dgdu_discrete = dg!, dt = 0.01, adaptive = false,
        )
        @test via == direct
        @test via[2] isa LinearAlgebra.Adjoint
        single = remake(prob; u0 = Float32.(U0), tspan = (0.0f0, 1.0f0), p = Float32.(P0))
        sol32 = solve(single, TSRK("4"); dt = 0.01f0, adaptive = false, saveat = 0.1f0)
        via32 = adjoint_sensitivities(
            sol32, TSRK("4"); sensealg = PETScAdjoint(),
            t = sol32.t, dgdu_discrete = dg!, dt = 0.01f0, adaptive = false,
        )
        @test via32 == PETScDiffEq._discrete_adjoint(
            single, TSRK("4"), PETScAdjoint();
            t = sol32.t, dgdu_discrete = dg!, dt = 0.01f0, adaptive = false,
        )
        @test via32[1] isa Vector{Float32}
        @test eltype(via32[2]) === Float32
        accel!(ddu, du, u, p, t) = (ddu .= -p[1] .* u .- p[2] .* du; nothing)
        second = SecondOrderODEProblem(accel!, [0.5], [1.0], (0.0, 1.0), [2.0, 0.3])
        sol2 = solve(second, TSImplicit("cn", EXACT); dt = 0.01, adaptive = false, saveat = TS)
        via2 = adjoint_sensitivities(
            sol2, TSImplicit("cn", EXACT); sensealg = PETScAdjoint(),
            t = TS, dgdu_discrete = dg!, dt = 0.01, adaptive = false,
        )
        @test via2 == PETScDiffEq._discrete_adjoint(
            second, TSImplicit("cn", EXACT), PETScAdjoint();
            t = TS, dgdu_discrete = dg!, dt = 0.01, adaptive = false,
        )
        @test via2[1] isa typeof(second.u0)
    end

    @testset "agrees with GaussAdjoint, closer at the method's order" begin
        gsol = solve(prob, Tsit5(); abstol = 1.0e-13, reltol = 1.0e-13, saveat = TS)
        gdu0, gdp = adjoint_sensitivities(
            gsol, Tsit5(); t = TS, dgdu_discrete = dg!, sensealg = GaussAdjoint(),
            abstol = 1.0e-13, reltol = 1.0e-13,
        )
        reference = vcat(gdu0, vec(gdp))
        for (problem, alg, bound, lo, hi) in (
                (prob, TSRK("4"), 1.0e-10, 13.0, 19.0),
                (prob, TSImplicit("beuler", EXACT), 1.0e-2, 1.9, 2.1),
                (prob, TSImplicit("cn", EXACT), 2.0e-5, 3.8, 4.2),
                (prob, TSImplicit("theta", 0.7, EXACT), 2.0e-3, 1.9, 2.1),
                (prob, TSImplicit("theta", EXACT), 5.0e-6, 3.8, 4.2),
                (prob, TSARKIMEX("3", EXACT), 1.0e-8, 7.6, 8.4),
                (prob, TSARKIMEX("l2", EXACT), 3.0e-6, 3.8, 4.2),
                (parts, TSARKIMEX("3", EXACT), 2.0e-8, 7.6, 8.4),
                (parts, TSARKIMEX("2e", EXACT), 3.0e-6, 3.8, 4.2),
            )
            gaps = map((0.01, 0.005)) do dt
                sol = solve(problem, alg; dt, adaptive = false, saveat = TS)
                du0, dp = adjoint_sensitivities(
                    sol, alg; sensealg = PETScAdjoint(),
                    t = TS, dgdu_discrete = dg!, dt, adaptive = false,
                )
                relerr(vcat(du0, vec(dp)), reference)
            end
            @test gaps[1] < bound
            @test lo < gaps[1] / gaps[2] < hi
        end
        ends = [0.0, 1.0]
        esol = solve(prob, Tsit5(); abstol = 1.0e-13, reltol = 1.0e-13, saveat = ends)
        at_ends = flat(
            adjoint_sensitivities(
                esol, Tsit5(); t = ends, dgdu_discrete = dg!, sensealg = GaussAdjoint(),
                abstol = 1.0e-13, reltol = 1.0e-13,
            ),
        )
        for problem in (prob, parts)
            alg = TSARKIMEX("3", EXACT)
            gaps = map((1.0e-6, 1.0e-8)) do tol
                sol = solve(problem, alg; abstol = tol, reltol = tol)
                mine = adjoint_sensitivities(
                    sol, alg; sensealg = PETScAdjoint(), t = ends, dgdu_discrete = dg!,
                    abstol = tol, reltol = tol,
                )
                relerr(flat(mine), at_ends)
            end
            @test gaps[1] < 1.0e-5
            @test gaps[2] < 1.0e-7
        end
    end

    # Measured: 2.5e-8 and 8.4e-10 for 5dp, 4.7e-6 and 5.0e-8 for ARKIMEX 3.
    @testset "interior cost times of an adaptive solve agree with GaussAdjoint" begin
        times = [0.25, 0.5, 0.75, 1.0]
        # A dense reference: one saved at these times alone left GaussAdjoint 2e-7 off.
        dense = solve(prob, Tsit5(); abstol = 1.0e-13, reltol = 1.0e-13)
        reference = flat(
            adjoint_sensitivities(
                dense, Tsit5(); t = times, dgdu_discrete = dg!, sensealg = GaussAdjoint(),
                abstol = 1.0e-13, reltol = 1.0e-13,
            ),
        )
        for (alg, bounds) in (
                (TSRK("5dp"), (1.0e-7, 5.0e-9)), (TSARKIMEX("3", EXACT), (2.0e-5, 2.0e-7)),
            )
            for (tol, bound) in zip((1.0e-6, 1.0e-8), bounds)
                sol = solve(prob, alg; abstol = tol, reltol = tol, saveat = times)
                mine = adjoint_sensitivities(
                    sol, alg; sensealg = PETScAdjoint(), t = times, dgdu_discrete = dg!,
                    abstol = tol, reltol = tol,
                )
                @test relerr(flat(mine), reference) < bound
            end
        end
    end

    @testset "integral costs agree with QuadratureAdjoint and InterpolatingAdjoint" begin
        tight = (abstol = 1.0e-13, reltol = 1.0e-13)
        dense = solve(prob, Tsit5(); tight...)
        references = map((QuadratureAdjoint(), InterpolatingAdjoint())) do sensealg
            flat(
                adjoint_sensitivities(
                    dense, Tsit5(); sensealg, g, dgdu_continuous = gu!, dgdp_continuous = gp!,
                    tight...,
                ),
            )
        end
        @test relerr(references[1], references[2]) < 1.0e-12
        for (alg, bound, lo, hi) in (
                (TSRK("4"), 2.0e-10, 13.0, 19.0),
                (TSImplicit("beuler", EXACT), 2.0e-2, 1.9, 2.1),
                (TSImplicit("cn", EXACT), 1.0e-4, 3.8, 4.2),
                (TSImplicit("theta", 0.7, EXACT), 3.0e-3, 1.9, 2.1),
                (TSImplicit("theta", EXACT), 2.0e-5, 3.8, 4.2),
                (TSImplicit("theta", 0.7, [EXACT; "-ts_theta_endpoint"]), 3.0e-3, 1.9, 2.1),
            )
            gaps = map((0.01, 0.005)) do dt
                sol = solve(prob, alg; dt, adaptive = false)
                mine = flat(
                    adjoint_sensitivities(
                        sol, alg; sensealg = PETScAdjoint(), g, dgdu_continuous = gu!,
                        dgdp_continuous = gp!, dt, adaptive = false,
                    ),
                )
                maximum(r -> relerr(mine, r), references)
            end
            @test gaps[1] < bound
            @test lo < gaps[1] / gaps[2] < hi
        end
        sol = solve(prob, TSRK("4"); dt = 0.01, adaptive = false)
        via = adjoint_sensitivities(
            sol, TSRK("4"); sensealg = PETScAdjoint(), g, dt = 0.01, adaptive = false,
        )
        @test via == PETScDiffEq._discrete_adjoint(
            prob, TSRK("4"), PETScAdjoint(); g, dt = 0.01, adaptive = false,
        )
        differentiated = adjoint_sensitivities(
            dense, Tsit5(); sensealg = InterpolatingAdjoint(), g, tight...,
        )
        @test relerr(flat(via), flat(differentiated)) < 2.0e-10
        mixed = adjoint_sensitivities(
            dense, Tsit5(); sensealg = QuadratureAdjoint(), t = TS, dgdu_discrete = dg!,
            dgdu_continuous = gu!, dgdp_continuous = gp!, tight...,
        )
        both = adjoint_sensitivities(
            sol, TSRK("4"); sensealg = PETScAdjoint(), t = TS, dgdu_discrete = dg!, g,
            dt = 0.01, adaptive = false,
        )
        @test relerr(flat(both), flat(mixed)) < 2.0e-10
    end

    @testset "what it refuses" begin
        sol = solve(prob, TSRK("4"); dt = 0.01, adaptive = false, saveat = TS)
        tsit = solve(prob, Tsit5(); saveat = TS)
        @test_throws "ArgumentError: PETScAdjoint runs PETSc's own adjoint" adjoint_sensitivities(
            tsit, Tsit5(); sensealg = PETScAdjoint(), t = TS, dgdu_discrete = dg!,
        )
        imex = solve(parts, TSARKIMEX(); dt = 0.01, adaptive = false)
        @test_throws "ArgumentError: PETScAdjoint cannot take an integral cost with TSARKIMEX" adjoint_sensitivities(
            imex, TSARKIMEX(); sensealg = PETScAdjoint(), g, dt = 0.01, adaptive = false,
        )
        @test_throws "ArgumentError: `dgdp_continuous` was given without `g` or `dgdu_continuous`" adjoint_sensitivities(
            sol, TSRK("4"); sensealg = PETScAdjoint(), dt = 0.01, adaptive = false,
            t = TS, dgdu_discrete = dg!, dgdp_continuous = gp!,
        )
        @test_throws "ArgumentError: PETScAdjoint is reached through" Zygote.gradient(
            p -> sum(
                Array(
                    solve(
                        prob, TSRK("4"); p, dt = 0.01, adaptive = false, saveat = TS,
                        sensealg = PETScAdjoint(),
                    ),
                ),
            ),
            P0,
        )
    end

    @testset "no method mentioning PETScAdjoint is ambiguous" begin
        ext = Base.get_extension(PETScDiffEq, :PETScDiffEqSciMLSensitivityExt)
        ambiguous = Test.detect_ambiguities(PETScDiffEq, ext, SciMLSensitivity, SciMLBase)
        mine = filter(pair -> any(m -> occursin("PETScAdjoint", string(m.sig)), pair), ambiguous)
        @test isempty(mine)
    end
end
