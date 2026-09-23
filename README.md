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

`dt` sets the first step. An adaptive solve can leave it out, and the first step is then
Hairer and Wanner's estimate as OrdinaryDiffEq uses it, taken with the `abstol` and `reltol`
keywords (SciML's defaults, `abstol = 1e-6` and `reltol = 1e-3`, when not given). DAE and mass-matrix
problems start from a small step instead. A solve PETSc steps at a fixed size needs `dt`:
`adaptive = false`, `-ts_adapt_type none`, the fixed-step families (`TSImplicit` and `TSDAE`
other than `bdf`, `TSIRK`, `TSMPRK`), and subtypes registered without an embedded error
estimate. `TSGeneric` needs it too, since which of its types adapt is not known here.

## Solvers

- `TSRK(subtype)`, explicit Runge-Kutta
- `TSRosW(subtype)`, linearly implicit Rosenbrock-W
- `TSImplicit(subtype; order)`, backward Euler, Crank-Nicolson, theta and BDF
- `TSIRK(nstages)`, Gauss-Legendre implicit Runge-Kutta of order `2 * nstages`
- `TSARKIMEX(subtype)`, additive Runge-Kutta IMEX, for a `SplitODEProblem`
- `TSDAE(subtype)`, the same implicit methods applied to a `DAEProblem`
- `TSGeneric(ts_type)`, a pass-through to any other PETSc `TSType` by name

Each has a docstring covering its subtypes, whether it adapts and what it requires, so
`?TSRosW` at the REPL is the reference. One default worth knowing: PETSc's BDF is order 2,
so pass `TSImplicit("bdf"; order = 5)` when comparing against a higher-order method. Every solver takes `petsc_options`, a vector of
command-line style tokens passed to PETSc for that solve, which are parsed after the
options this package sets and so take precedence.

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
one dual evaluation of `f` per Jacobian rather than one evaluation per state. The
prototype has to hold every entry the Jacobian can have: one it leaves out is left out of
the Jacobian, which then costs Newton iterations. Passing
`autodiff = PETScDiffEq.AutoFiniteDiff()` to the algorithm leaves the Jacobian to PETSc's
finite differences, coloured by a sparse prototype too. On a badly scaled stiff problem such as
Robertson's, those are far enough off that the solve reports success with an answer
wrong in its first digit, so keep them for a right-hand side ForwardDiff cannot run.

`DiscreteCallback`, `ContinuousCallback`, `VectorContinuousCallback` and `CallbackSet`
all work, as does the integrator interface through `init`, `step!`, `solve!`, `reinit!`
and `terminate!`.

## Number types

A `Float64` state runs in PETSc's double-precision build and a `ComplexF64` state in its
double-precision complex build. A `Float32` or `ComplexF32` state runs in the matching
single-precision build when the span is `Float32` as well, following OrdinaryDiffEq's advice
to give a single-precision problem a `Float32` span, and in the double-precision build when
the span is `Float64`, with the saved states given back in single precision. DiffEqBase
promotes the span to the type of `dt`, so a single-precision solve takes `dt = 0.01f0`
rather than `dt = 0.01`, and it makes a whole-number span `Float64`. Any other real state,
whole numbers included, is solved in `Float64` and comes back in it. Where PETSc.jl has not
loaded the single-precision build, as with a library set with `PETSc.set_library!`, a
single-precision state runs in the double-precision one, and a problem whose build is not
loaded at all is refused with an `ArgumentError` that names it.

Times are in the type of the clock PETSc steps on: `Float32` for a single-precision state
with a `Float32` span and `Float64` otherwise. That covers `sol.t` and the integrator's `t`
and `dt`, and the integrator's `u` is in the type PETSc steps. OrdinaryDiffEq gives the
times in the span's type whatever the state; here a `Float64` state with a `Float32` span is
stepped on PETSc's `Float64` clock, and rounding its times to `Float32` would give states
close together the same time.

Single precision has seven digits, which bounds the clock as well as the state. The first
step, and the step after a stop, are at least four ulps of `t`, since PETSc cannot take a
step whose stages have no room between its ends; a single-precision solve is stepped by
this package's integrator, which lands on each stop and on the final time itself, where
PETSc's own landing would refuse a step under about `1e-6` before a final time below 1.
Tolerances finer than single precision can resolve, about `1e-7`, are accepted but buy
nothing past its rounding, and can drive PETSc's adaptive step below what the clock can
tell apart, which ends the solve with a failed retcode. PETSc's single build also takes the
norms its Newton and Krylov iterations stop on in single precision, and a vector whose
entries are all below about `1e-19` in size has a norm of zero there. An implicit solve of
such a state can then stop without moving it and still report success, so this package
warns when a solve starts on such a state, or fails on one; a state that decays there from
above is within any coarser tolerance. Rescale the problem, or give it a `Float64` span.

With a complex state, times, `dt`, `saveat`, `tstops` and the tolerances stay real, and
PETSc's error norms take each component's modulus. A `ContinuousCallback`'s condition has
to return a real number, such as `real(u[1]) - 0.5`, since a root is a sign change. The
implicit methods' Newton iteration needs a holomorphic `f`, one that does not go through
`conj`, `abs`, `real` or `imag` of the state. ForwardDiff takes no complex numbers, so
without a `jac` the Jacobian is differentiated along the real parts of the state, which for
a holomorphic `f` is its complex Jacobian, and a sparse `jac_prototype` is coloured as for a
real state. A check at the start compares that derivative with the one along the imaginary
parts and refuses an `f` that is not holomorphic; it is best effort, and can miss a term too
small to show near the initial state. `AutoFiniteDiff()` and a hand-written `jac` are not
checked. The explicit methods take any `f`.

## Adjoint sensitivities

With SciMLSensitivity loaded, `adjoint_sensitivities(sol, alg; sensealg = PETScAdjoint(), ...)`
runs PETSc's own discrete adjoint for `TSRK`, `TSImplicit("beuler")` and `TSImplicit("cn")`.
The keywords that set the steps have to be repeated from `solve`. It runs in PETSc's double
real build, so a `Float32` problem is differentiated in `Float64` and its gradients come
back as `Float32`, and a complex one is refused. `?PETScAdjoint` and the documentation cover
what it needs, what it refuses and how to check `jac` and `paramjac`.

## Limitations

Every solve runs on `MPI.COMM_SELF`, so this package is serial. PETSc TS is built for
large distributed problems, and reaching it from the SciML interface is what this package
is for; use OrdinaryDiffEq.jl for serial problems where it applies.

On 32-bit Julia, use Julia 1.10, or add `PETSc_jll = "~3.22"` to your own compat: PETSc_jll
3.25 has no 32-bit builds, and newer Julia versions would otherwise resolve it.

Solves from several threads, such as an `EnsembleThreads` ensemble, are safe but run one
at a time: PETSc's options and MPI are shared by the whole process.

Finish or terminate every integrator you start. One dropped part way is released by a
finalizer, and if that finalizer runs at process exit, after MPI has shut down, PETSc's
own object finalizers print an MPI warning and the process exits non-zero.

## License

MIT. See [LICENSE](LICENSE).
