# PETScDiffEq.jl

PETScDiffEq.jl contains bindings for the [PETSc](https://petsc.org) TS time integrators
to allow them to be used with the SciML common interface. PETSc's linear and nonlinear
solvers are already reachable from SciML through LinearSolve.jl and NonlinearSolve.jl;
this package covers the third layer, TS.

## Installation

```julia
using Pkg
Pkg.add("PETScDiffEq")
```

## Common API Usage

This library adds the common interface to PETSc's TS solvers, documented in the
[DifferentialEquations.jl documentation](https://docs.sciml.ai/DiffEqDocs/stable/).
Following the Lorenz example from
[the ODE tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/ode_example/):

```julia
using PETScDiffEq, SciMLBase

function lorenz(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = SciMLBase.ODEProblem(lorenz, u0, tspan)
sol = SciMLBase.solve(prob, TSRK("5dp"); dt = 0.01, abstol = 1e-8, reltol = 1e-8)
```

`dt` sets the first step. An adaptive solve can leave it out, and the first step is then
Hairer and Wanner's estimate as OrdinaryDiffEq uses it, taken with the `abstol` and `reltol`
keywords (SciML's defaults, `abstol = 1e-6` and `reltol = 1e-3`, when not given). DAE and mass-matrix
problems start from a small step instead. A solve PETSc steps at a fixed size needs `dt`:
`adaptive = false`, `-ts_adapt_type none`, the fixed-step families (`TSImplicit` and `TSDAE`
other than `bdf`, `TSIRK`, `TSMPRK`), and subtypes registered without an embedded error
estimate. `TSGeneric` needs it too, since which of its types adapt is not known here.

## Solver Options

The options available in `solve` are documented
[at the common solver options page](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/).
This package supports `dt`, `adaptive`, `dtmin`, `force_dtmin`, `dtmax`, `reltol` and
`abstol` (either may be a vector of per-component tolerances), `saveat`, `save_everystep`,
`save_start`, `save_end`, `save_on`, `save_idxs`, `dense`, `callback`, `tstops`,
`d_discontinuities`, `unstable_check` and `isoutofdomain`. Keywords it cannot
honour emit a warning rather than being silently dropped.

Saving follows OrdinaryDiffEq: a `saveat` keeps only its own points, adding `t0` or `tf`
only when it names them or `save_start` or `save_end` asks, and `save_everystep = true`
saves every step alongside it. `dtmin` is a floor the solve keeps: once PETSc proposes a
smaller step, the solve ends there with `ReturnCode.DtLessThanMin`. A step shortened to
land on a stop or the final time does not count, and a fixed-step solve has no floor, as in
OrdinaryDiffEq. With `force_dtmin = true` the solve goes on at `dtmin` instead, and the floor
wins over a smaller `dtmax`. `d_discontinuities` are stepped onto, and, as SciML defines them,
the step after each starts one ULP past it, so the right-hand side there sees the new regime
when written as `if t > t_d`. `isoutofdomain(u, p, t)` is asked after each step of an adaptive solve, and a
step that leaves the domain is taken again at a fifth of its size, as OrdinaryDiffEq takes it;
one that cannot be made small enough ends the solve with `Unstable`, or `DtLessThanMin` at
`dtmin`.

A solve that stops short of the final time says why in its retcode: `Unstable` when the
state stops being finite, PETSc hits an overflow, or `unstable_check(dt, u, p, t)`
returns true, which is asked before each step with the step about to be taken, as
OrdinaryDiffEq asks it, `ConvergenceFailure` when a nonlinear
solve fails, `DtLessThanMin` as above, `MaxIters` when `maxiters` steps are taken, and
`Failure` for a zero pivot, with a warning, or another step PETSc cannot take. Where
`petsc_options` asks PETSc to raise, with `-ksp_error_if_not_converged`,
`-snes_error_if_not_converged` or `-ts_error_if_step_fails`, it raises instead.

A state between step ends, for `saveat`, `integrator(t)` or a `ContinuousCallback`, comes
from PETSc's own interpolant for `TSRK("5dp")`, `TSRosW("ra34pw2")`, `TSARKIMEX("4")` and
`"5"`, `TSImplicit("bdf")` and `TSDAE("bdf")`, and from the cubic Hermite interpolant dense
output uses for everything else, `TSGeneric` and a type `petsc_options` changes included.
With a mass matrix or a `DAEProblem` only PETSc's is available, and a type that has none
raises an `ArgumentError` when such a state is needed.

`ODEProblem`, `SplitODEProblem` and `DAEProblem` are supported, in place or out of
place, along with
`ODEFunction`'s `jac`, `jac_prototype` and `mass_matrix`. Supply a `jac_prototype` for
anything sparse: without one the Jacobian is dense and forces a dense factorization.

Without a `jac`, the implicit algorithms build the Jacobian with ForwardDiff, as
OrdinaryDiffEq does, and colour a sparse `jac_prototype`, so a tridiagonal problem costs
one dual evaluation of `f` per Jacobian rather than one evaluation per state. Passing
ADTypes' `AutoFiniteDiff()` as the algorithm's `autodiff` leaves the Jacobian to PETSc's
finite differences, coloured by a sparse prototype too. On a badly scaled stiff problem such as
Robertson's, those are far enough off that the solve reports success with an answer
wrong in its first digit, so keep them for a right-hand side ForwardDiff cannot run.

`DiscreteCallback`, `ContinuousCallback`, `VectorContinuousCallback` and `CallbackSet`
all work, as does the integrator interface through `init`, `step!`, `solve!`, `reinit!`
and `terminate!`.

## Adjoint sensitivities

With SciMLSensitivity loaded, `PETScAdjoint` computes gradients with PETSc's own discrete
adjoint, `TSAdjointSolve`, through `adjoint_sensitivities`:

```julia
using PETScDiffEq, SciMLBase, SciMLSensitivity

function f!(du, u, p, t)
    du[1] = -p[1] * u[1] + p[2] * u[1] * u[2]
    du[2] = p[3] * u[1] - p[4] * u[2]^2
end
function jac!(J, u, p, t)
    J[1, 1] = -p[1] + p[2] * u[2]
    J[1, 2] = p[2] * u[1]
    J[2, 1] = p[3]
    J[2, 2] = -2 * p[4] * u[2]
end
function paramjac!(pJ, u, p, t)
    fill!(pJ, 0.0)
    pJ[1, 1] = -u[1]
    pJ[1, 2] = u[1] * u[2]
    pJ[2, 3] = u[1]
    pJ[2, 4] = -u[2]^2
end
prob = ODEProblem(
    ODEFunction(f!; jac = jac!, paramjac = paramjac!),
    [1.0, 0.5], (0.0, 1.0), [0.7, 0.3, 0.4, 0.2],
)
sol = solve(prob, TSRK("4"); dt = 0.01, adaptive = false)

# The cost is the sum of |u(t)|^2 / 2 over these times.
ts = 0.0:0.1:1.0
dg!(out, u, p, t, i) = (out .= u)
du0, dp = adjoint_sensitivities(
    sol, TSRK("4"); sensealg = PETScAdjoint(),
    t = ts, dgdu_discrete = dg!, dt = 0.01, adaptive = false,
)
```

It works with `TSRK` of any subtype, `TSImplicit("beuler")` and `TSImplicit("cn")`. PETSc
has no adjoint for `TSRosW`, `TSIRK`, `TSMPRK` or BDF, and `TSARKIMEX` and the general
theta method are refused as well.

The gradient is that of the solution PETSc computes at these steps. It agrees with finite
differences of the same fixed-step `solve`, and it differs from a continuous adjoint such as
`GaussAdjoint` by the discretization error, which shrinks at the method's order. Because
`PETScAdjoint` runs the forward solve again and takes only the problem from `sol`, the
keywords that set the steps, `dt`, `adaptive`, `abstol`, `reltol`, `dtmin`, `dtmax` and
`maxiters`, have to be passed to `adjoint_sensitivities` exactly as they were to `solve`.

With fixed steps every cost time must be a time the solve steps to, since PETSc's adjoint
has no derivative of interpolation. An adaptive solve can take costs only at the ends of
`tspan`, and its gradient holds the accepted step sizes fixed rather than differentiating
the step-size controller.

`TSImplicit` solves transposed linear systems with the Krylov solver its Newton steps use,
by default GMRES with ILU(0) stopping at a relative residual of 1e-5, so the gradient can be
off by up to about that tolerance while the forward states are far more accurate. With
Crank-Nicolson and default options, the gradient's relative error was 4e-8 for a 2-D heat
equation on a 7 by 7 grid, 2.5e-6 on a 24 by 24 grid and 1.1e-5 for advection-diffusion on
a 16 by 16 grid, while the forward states were within 8e-12, 5e-10 and 1e-9 of a direct
solve. A 1-D heat equation with 50 unknowns gave the same gradient as a direct solve to
2e-15, since ILU(0) of a tridiagonal matrix is exact. Passing
`PETScAdjoint(petsc_options = ["-ksp_type", "preonly", "-pc_type", "lu"])` removed the
difference in every case. For a problem too large to factor, tighten `-ksp_rtol` instead;
`1e-10` brought the two larger grids to 1e-11 and 5e-11.

Callbacks, `tstops`, integral costs, mass matrices, `DAEProblem` and `SplitODEProblem` are
refused, and so is differentiating `solve` with a reverse-mode AD package. Passing
`sensealg = PETScAdjoint()` to `solve` itself does nothing.

`jac` and `paramjac` go into the gradient unchecked, so a wrong entry gives a wrong
gradient. Compare them against central differences of `f!` at a representative point
before relying on the result:

```julia
function jacobian_errors(f!, jac!, paramjac!, u, p, t; h = 1.0e-6)
    n, m = length(u), length(p)
    J, pJ = zeros(n, n), zeros(n, m)
    jac!(J, u, p, t)
    paramjac!(pJ, u, p, t)
    fp, fm = similar(u), similar(u)
    for j in 1:n
        e = zeros(n)
        e[j] = h
        f!(fp, u + e, p, t)
        f!(fm, u - e, p, t)
        J[:, j] .-= (fp - fm) / 2h
    end
    for k in 1:m
        e = zeros(m)
        e[k] = h
        f!(fp, u, p + e, t)
        f!(fm, u, p - e, t)
        pJ[:, k] .-= (fp - fm) / 2h
    end
    return maximum(abs, J), maximum(abs, pJ)
end
jacobian_errors(f!, jac!, paramjac!, [1.0, 0.5], [0.7, 0.3, 0.4, 0.2], 0.3)
```

Both errors should be near round-off, around 1e-10 here; a wrong entry shows up at its own
size.

## Limitations

Every solve runs on `MPI.COMM_SELF`, so this package is serial. PETSc TS is built for
large distributed problems, and reaching it from the SciML interface is what this package
is for; use OrdinaryDiffEq.jl for serial problems where it applies.

Solves run in Float64, PETSc's double build: a `Float32` or whole-number state is
converted, and the solution comes back in Float64.

On 32-bit Julia, use Julia 1.10, or add `PETSc_jll = "~3.22"` to your own compat: PETSc_jll
3.25 has no 32-bit builds, and newer Julia versions would otherwise resolve it.

Solves from several threads, such as an `EnsembleThreads` ensemble, are safe but run one
at a time: PETSc's options and MPI are shared by the whole process.

Finish or terminate every integrator you start. One dropped part way is released by a
finalizer, and if that finalizer runs at process exit, after MPI has shut down, PETSc's
own object finalizers print an MPI warning and the process exits non-zero.

## Solvers

```@docs
TSRK
TSRosW
TSImplicit
TSIRK
TSARKIMEX
TSDAE
TSMPRK
TSGeneric
PETScIntegrator
PETScAdjoint
```
