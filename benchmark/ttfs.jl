# julia --project=<env with PETScDiffEq, SciMLBase, SparseArrays> benchmark/ttfs.jl [case]
# With no case it times every case, each in a fresh process.
const CASES = [
    "rk", "rk_p", "rk_oop", "rk_saveat", "rosw", "bdf", "bdf_sparse", "arkimex", "dae",
    "integ", "callback", "f32", "complex", "adjoint",
]

if isempty(ARGS)
    println(rpad("case", 12), lpad("load", 9), lpad("first", 9), lpad("second", 9))
    for name in CASES
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__) $name`
        print(readchomp(cmd), "\n")
    end
    exit()
end

t_load = @elapsed using PETScDiffEq, SciMLBase, SparseArrays

function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
    return nothing
end
function lorenz_p!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (p[2] - u[3]) - u[2]
    du[3] = u[1] * u[2] - p[3] * u[3]
    return nothing
end
lorenz(u, p, t) = [10.0 * (u[2] - u[1]), u[1] * (28.0 - u[3]) - u[2], u[1] * u[2] - (8 / 3) * u[3]]
function rober!(du, u, p, t)
    k1, k2, k3 = p
    du[1] = -k1 * u[1] + k3 * u[2] * u[3]
    du[2] = k1 * u[1] - k2 * u[2]^2 - k3 * u[2] * u[3]
    du[3] = k2 * u[2]^2
    return nothing
end
function rober_dae!(res, du, u, p, t)
    k1, k2, k3 = p
    res[1] = -k1 * u[1] + k3 * u[2] * u[3] - du[1]
    res[2] = k1 * u[1] - k2 * u[2]^2 - k3 * u[2] * u[3] - du[2]
    res[3] = u[1] + u[2] + u[3] - 1
    return nothing
end
function heat!(du, u, p, t)
    n = length(u)
    for i in 1:n
        du[i] = -2u[i] + (i > 1 ? u[i - 1] : 0.0) + (i < n ? u[i + 1] : 0.0)
    end
    return nothing
end
stiff!(du, u, p, t) = (du .= -100 .* u; nothing)
nonstiff!(du, u, p, t) = (du .= sin(t); nothing)
decay!(du, u, p, t) = (du .= p[1] .* u; nothing)
cdecay!(du, u, p, t) = (du .= (-1.0 + 2.0im) .* u; nothing)
dg!(out, u, p, t, i) = (out .= u; nothing)

const U0 = [1.0, 0.0, 0.0]
const RP = [0.04, 3.0e7, 1.0e4]
const TRIDIAG = spdiagm(-1 => ones(9), 0 => ones(10), 1 => ones(9))

function run_case(name)
    return if name == "rk"
        solve(ODEProblem(lorenz!, U0, (0.0, 1.0)), TSRK("5dp"); abstol = 1.0e-8, reltol = 1.0e-8)
    elseif name == "rk_p"
        solve(ODEProblem(lorenz_p!, U0, (0.0, 1.0), [10.0, 28.0, 8 / 3]), TSRK("5dp"))
    elseif name == "rk_oop"
        solve(ODEProblem(lorenz, U0, (0.0, 1.0)), TSRK())
    elseif name == "rk_saveat"
        solve(ODEProblem(lorenz!, U0, (0.0, 1.0)), TSRK(); saveat = 0.1, abstol = 1.0e-8)
    elseif name == "rosw"
        solve(ODEProblem(rober!, U0, (0.0, 1.0), RP), TSRosW())
    elseif name == "bdf"
        solve(ODEProblem(rober!, U0, (0.0, 1.0), RP), TSImplicit("bdf"))
    elseif name == "bdf_sparse"
        f = ODEFunction(heat!; jac_prototype = TRIDIAG)
        solve(ODEProblem(f, collect(1.0:10.0), (0.0, 1.0)), TSImplicit("bdf"))
    elseif name == "arkimex"
        solve(SplitODEProblem(stiff!, nonstiff!, [1.0, 2.0], (0.0, 1.0)), TSARKIMEX())
    elseif name == "dae"
        prob = DAEProblem(
            rober_dae!, [-0.04, 0.04, 0.0], U0, (0.0, 1.0), RP;
            differential_vars = [true, true, false],
        )
        solve(prob, TSDAE("bdf"))
    elseif name == "integ"
        integ = init(ODEProblem(lorenz!, U0, (0.0, 1.0)), TSRK("5dp"))
        step!(integ)
        step!(integ, 0.1, true)
        integ((integ.tprev + integ.t) / 2)
        solve!(integ)
    elseif name == "callback"
        cb = CallbackSet(
            ContinuousCallback((u, t, integ) -> u[1] - 5.0, integ -> (integ.u[2] += 0.1)),
            DiscreteCallback((u, t, integ) -> t > 0.5, integ -> nothing),
        )
        solve(ODEProblem(lorenz!, U0, (0.0, 1.0)), TSRK("5dp"); callback = cb, saveat = 0.1)
    elseif name == "f32"
        prob = ODEProblem(decay!, Float32[1.0, 2.0], (0.0f0, 1.0f0), [-1.0f0])
        solve(prob, TSRK("5dp"); dt = 0.01f0)
    elseif name == "complex"
        solve(ODEProblem(cdecay!, ComplexF64[1.0, 2.0im], (0.0, 1.0)), TSRK("5dp"))
    elseif name == "adjoint"
        PETScDiffEq._discrete_adjoint(
            ODEProblem(decay!, [1.0, 2.0], (0.0, 1.0), [-1.0]), TSRK("4"), PETScAdjoint();
            t = collect(0.0:0.1:1.0), dgdu_discrete = dg!, dt = 0.01, adaptive = false,
        )
    else
        error("unknown case $name")
    end
end

name = only(ARGS)
t_first = @elapsed run_case(name)
t_second = @elapsed run_case(name)
fmt(t) = lpad(string(round(t; digits = 3)), 9)
println(rpad(name, 12), fmt(t_load), fmt(t_first), fmt(t_second))
