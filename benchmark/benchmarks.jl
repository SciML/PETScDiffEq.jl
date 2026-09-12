using BenchmarkTools
using LinearAlgebra
using PETScDiffEq
using SciMLBase
using SparseArrays

const SUITE = BenchmarkGroup()

function decay!(du, u, p, t)
    @inbounds for i in eachindex(du)
        du[i] = -u[i]
    end
    return nothing
end

function lotka_volterra!(du, u, p, t)
    a, b, c, d = p
    du[1] = a * u[1] - b * u[1] * u[2]
    return du[2] = -c * u[2] + d * u[1] * u[2]
end

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

prob_decay = ODEProblem(decay!, [1.0], (0.0, 1.0))
prob_lv = ODEProblem(lotka_volterra!, [1.0, 1.0], (0.0, 5.0), [1.5, 1.0, 3.0, 1.0])
prob_osc = ODEProblem(
    ODEFunction(
        damped_oscillator!;
        jac = damped_oscillator_jac!, jac_prototype = OSCILLATOR_PROTOTYPE
    ),
    [1.0, 0.0], (0.0, 5.0)
)

SUITE["tsrk"] = BenchmarkGroup()
SUITE["tsrk"]["construct"] = @benchmarkable PETScDiffEq.TSRK("3bs")
SUITE["tsrk"]["decay_3bs"] = @benchmarkable solve(
    $prob_decay, PETScDiffEq.TSRK("3bs"); dt = 0.01
)
SUITE["tsrk"]["lv_5dp"] = @benchmarkable solve(
    $prob_lv, PETScDiffEq.TSRK("5dp"); dt = 0.01
)

SUITE["tsrosw"] = BenchmarkGroup()
SUITE["tsrosw"]["decay"] = @benchmarkable solve(
    $prob_decay, PETScDiffEq.TSRosW("ra34pw2"); dt = 0.05
)
SUITE["tsrosw"]["oscillator"] = @benchmarkable solve(
    $prob_osc, PETScDiffEq.TSRosW("ra34pw2"); dt = 0.05
)

SUITE["tsimplicit"] = BenchmarkGroup()
SUITE["tsimplicit"]["decay_beuler"] = @benchmarkable solve(
    $prob_decay, PETScDiffEq.TSImplicit("beuler"); dt = 0.05
)
SUITE["tsimplicit"]["oscillator_cn"] = @benchmarkable solve(
    $prob_osc, PETScDiffEq.TSImplicit("cn"); dt = 0.05
)
SUITE["tsimplicit"]["lv_bdf"] = @benchmarkable solve(
    $prob_lv, PETScDiffEq.TSImplicit("bdf"); dt = 0.05
)
