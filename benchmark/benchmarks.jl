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

function stiff_relaxation!(du, u, p, t)
    du[1] = -1000.0 * (u[1] - cos(t))
    return nothing
end

function relaxation_forcing!(du, u, p, t)
    du[1] = -sin(t)
    return nothing
end

function two_rates!(du, u, p, t)
    du[1] = -u[1]
    du[2] = -100.0 * u[2]
    return nothing
end

function algebraic_decay!(r, du, u, p, t)
    r[1] = du[1] + u[1]
    return r[2] = u[2] - u[1]
end

function algebraic_decay_jac!(J, du, u, p, gamma, t)
    J[1, 1] = gamma + 1.0
    J[1, 2] = 0.0
    J[2, 1] = -1.0
    return J[2, 2] = 1.0
end

prob_split = SplitODEProblem(stiff_relaxation!, relaxation_forcing!, [1.0], (0.0, 1.0))
prob_two_rates = ODEProblem(two_rates!, [1.0, 1.0], (0.0, 1.0))
prob_dae = DAEProblem(
    DAEFunction(algebraic_decay!; jac = algebraic_decay_jac!),
    [-1.0, -1.0], [1.0, 1.0], (0.0, 1.0)
)

SUITE["tsarkimex"] = BenchmarkGroup()
SUITE["tsarkimex"]["split_3"] = @benchmarkable solve(
    $prob_split, PETScDiffEq.TSARKIMEX("3"); dt = 0.01
)
SUITE["tsarkimex"]["oscillator_3"] = @benchmarkable solve(
    $prob_osc, PETScDiffEq.TSARKIMEX("3"); dt = 0.05
)

SUITE["tsirk"] = BenchmarkGroup()
SUITE["tsirk"]["oscillator_3stage"] = @benchmarkable solve(
    $prob_osc, PETScDiffEq.TSIRK(3); dt = 0.05
)

SUITE["tsmprk"] = BenchmarkGroup()
SUITE["tsmprk"]["two_rates_p2"] = @benchmarkable solve(
    $prob_two_rates, PETScDiffEq.TSMPRK([1], "p2"); dt = 0.01, adaptive = false
)

SUITE["tsdae"] = BenchmarkGroup()
SUITE["tsdae"]["algebraic_decay_bdf"] = @benchmarkable solve(
    $prob_dae, PETScDiffEq.TSDAE("bdf"); dt = 1.0e-3
)
