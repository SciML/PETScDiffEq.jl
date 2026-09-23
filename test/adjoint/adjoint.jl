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
relerr(a, b) = norm(a - b) / norm(b)

const U0 = [1.0, 0.5]
const P0 = [0.7, 0.3, 0.4, 0.2]
const TS = collect(0.0:0.1:1.0)
const EXACT = ["-snes_rtol", "1e-13", "-snes_atol", "1e-15", "-ksp_type", "preonly", "-pc_type", "lu"]
prob = ODEProblem(ODEFunction(f!; jac = jac!, paramjac = paramjac!), U0, (0.0, 1.0), P0)

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
        # A Float32 solution is differentiated in Float64 at the times it saved, with `dt`
        # repeated as its solve was given it, and its gradients come back Float32.
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
    end

    @testset "agrees with GaussAdjoint, closer at the method's order" begin
        gsol = solve(prob, Tsit5(); abstol = 1.0e-13, reltol = 1.0e-13, saveat = TS)
        gdu0, gdp = adjoint_sensitivities(
            gsol, Tsit5(); t = TS, dgdu_discrete = dg!, sensealg = GaussAdjoint(),
            abstol = 1.0e-13, reltol = 1.0e-13,
        )
        reference = vcat(gdu0, vec(gdp))
        # Measured gaps at dt = 0.01 and their ratio to dt = 0.005: RK4 1.7e-11 and 16.07,
        # backward Euler 2.7e-3 and 2.001, Crank-Nicolson 3.5e-6 and 4.000.
        for (alg, bound, lo, hi) in (
                (TSRK("4"), 1.0e-10, 13.0, 19.0),
                (TSImplicit("beuler", EXACT), 1.0e-2, 1.9, 2.1),
                (TSImplicit("cn", EXACT), 2.0e-5, 3.8, 4.2),
            )
            gaps = map((0.01, 0.005)) do dt
                sol = solve(prob, alg; dt, adaptive = false, saveat = TS)
                du0, dp = adjoint_sensitivities(
                    sol, alg; sensealg = PETScAdjoint(),
                    t = TS, dgdu_discrete = dg!, dt, adaptive = false,
                )
                relerr(vcat(du0, vec(dp)), reference)
            end
            @test gaps[1] < bound
            @test lo < gaps[1] / gaps[2] < hi
        end
    end

    @testset "what it refuses" begin
        sol = solve(prob, TSRK("4"); dt = 0.01, adaptive = false, saveat = TS)
        tsit = solve(prob, Tsit5(); saveat = TS)
        @test_throws "ArgumentError: PETScAdjoint runs PETSc's own adjoint" adjoint_sensitivities(
            tsit, Tsit5(); sensealg = PETScAdjoint(), t = TS, dgdu_discrete = dg!,
        )
        for cost in (
                (g = (u, p, t) -> sum(u),),
                (dgdu_continuous = (out, u, p, t) -> (out .= 1),),
                (t = TS, dgdu_discrete = dg!, dgdp_continuous = (out, u, p, t) -> (out .= 0)),
            )
            @test_throws "ArgumentError: PETScAdjoint supports discrete costs only" adjoint_sensitivities(
                sol, TSRK("4"); sensealg = PETScAdjoint(), dt = 0.01, adaptive = false, cost...,
            )
        end
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
        # The extension's methods belong to the extension module, not to PETScDiffEq.
        ext = Base.get_extension(PETScDiffEq, :PETScDiffEqSciMLSensitivityExt)
        ambiguous = Test.detect_ambiguities(PETScDiffEq, ext, SciMLSensitivity, SciMLBase)
        mine = filter(pair -> any(m -> occursin("PETScAdjoint", string(m.sig)), pair), ambiguous)
        @test isempty(mine)
    end
end
