# PETScDiffEq.jl

[![CI](https://github.com/SciML/PETScDiffEq.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SciML/PETScDiffEq.jl/actions/workflows/CI.yml)

This package contains bindings for the [PETSc](https://petsc.org) TS time integrators to
allow them to be used with the SciML common interface. PETSc's linear and nonlinear
solvers are already reachable from SciML through LinearSolve.jl and NonlinearSolve.jl;
this package covers the third layer, TS. For more information on using the solvers from
this package, see the
[DifferentialEquations.jl documentation](https://docs.sciml.ai/DiffEqDocs/stable/).

## Installation

```julia
using Pkg
Pkg.add("PETScDiffEq")
```

## Common API Usage

This library adds the common interface to PETSc's TS solvers.
[See the DifferentialEquations.jl documentation for details on the interface](https://docs.sciml.ai/DiffEqDocs/stable/).
Following the Lorenz example from
[the ODE tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/ode_example/), we can
solve this using `TSRK` via the following:

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

`dt` is required and sets the first step.

## Solvers

- `TSRK(subtype)`, explicit Runge-Kutta
- `TSRosW(subtype)`, linearly implicit Rosenbrock-W
- `TSImplicit(subtype)`, backward Euler, Crank-Nicolson, theta and BDF
- `TSIRK(nstages)`, Gauss-Legendre implicit Runge-Kutta of order `2 * nstages`
- `TSARKIMEX(subtype)`, additive Runge-Kutta IMEX, for a `SplitODEProblem`
- `TSDAE(subtype)`, the same implicit methods applied to a `DAEProblem`
- `TSGeneric(ts_type)`, a pass-through to any other PETSc `TSType` by name

Each has a docstring covering its subtypes, whether it adapts and what it requires, so
`?TSRosW` at the REPL is the reference. Every solver takes `petsc_options`, a vector of
command-line style tokens passed to PETSc for that solve, which are parsed after the
options this package sets and so take precedence.

## Solver Options

The options available in `solve` are documented
[at the common solver options page](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/).
This package supports `dt`, `adaptive`, `dtmin`, `dtmax`, `reltol` and `abstol` (either
may be a vector of per-component tolerances), `saveat`, `save_everystep`, `save_start`,
`save_end`, `save_idxs`, `dense`, `callback` and `tstops`. Keywords it cannot honour emit
a warning rather than being silently dropped.

`ODEProblem`, `SplitODEProblem` and `DAEProblem` are supported, in place or out of
place, along with
`ODEFunction`'s `jac`, `jac_prototype` and `mass_matrix`. Supply a `jac_prototype` for
anything sparse: without one the Jacobian is dense and forces a dense factorization.

`DiscreteCallback`, `ContinuousCallback`, `VectorContinuousCallback` and `CallbackSet`
all work, as does the integrator interface through `init`, `step!`, `solve!`, `reinit!`
and `terminate!`.

## Limitations

Every solve runs on `MPI.COMM_SELF`, so this package is serial. PETSc TS is built for
large distributed problems, and reaching it from the SciML interface is what this package
is for; use OrdinaryDiffEq.jl for serial problems where it applies.

Finish or terminate every integrator you start. One dropped part way is released by a
finalizer, and if that finalizer runs at process exit, after MPI has shut down, PETSc's
own object finalizers print an MPI warning and the process exits non-zero.

## License

MIT. See [LICENSE](LICENSE).
