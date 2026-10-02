# DAE initialization

A `DAEProblem`, or an `ODEProblem` whose mass matrix has zero rows and columns, has to start
where its algebraic equations hold. The `initializealg` keyword picks how, with the algorithms
OrdinaryDiffEq takes, which come from DiffEqBase:

- `CheckInit()`, the default, evaluates the residual at `u0`, and at `du0` for a
  `DAEProblem`, and throws a `CheckInitFailureError` when its RMS norm exceeds `abstol`,
  taken per component when `abstol` is a vector.
- `BrownFullBasicInit()` keeps the differential variables and solves for the algebraic
  ones, and for a `DAEProblem` also for the derivatives of the differential ones, which
  needs `differential_vars`. Its own `abstol`, `1e-10` unless given, decides whether to
  solve.
- `ShampineCollocationInit(initdt)` takes one backward Euler step and starts from where it
  lands. The step is `initdt` as given. Without one it runs toward `tf`, and for an
  `ODEProblem` it is OrdinaryDiffEq's: `dt / 5`, at most `dtmax`, when `dt` is given, and a
  thousandth of the span otherwise. For a `DAEProblem` it is a tenth of `dtmax`, the span by
  default, at `t0 = 0`, and the smaller of that and `|t0| / 1000` elsewhere. OrdinaryDiffEq
  ignores `initdt` for a `DAEProblem`, and steps differently there when `t0` is negative or
  the span runs backward.
- `NoInit()` starts from `u0` as given.

The two that solve use PETSc's SNES with the Jacobian the solve itself uses: the problem's
`jac`, the ForwardDiff one or PETSc's finite differences, sparse under a `jac_prototype`. So
they take no `nlsolve`, and when SNES fails the solve returns at `t0` with
`ReturnCode.InitialFailure`. A start that already passes the check is left untouched, and
`reinit!` initializes again unless given `reinit_dae = false`. They do not run on a
communicator other than `MPI.COMM_SELF`, where `CheckInit()` still checks the whole state.

The default is `CheckInit()` even for a problem carrying ModelingToolkit's initialization
data, which OrdinaryDiffEq would solve with `OverrideInit()`; this package does not solve that
system and refuses `OverrideInit()` on such a problem.

`initialize_dae!(integrator, initializealg)` runs the same on the integrator's current state
and time, with the `initializealg` the solve was given unless another is passed, and writes
the result into PETSc. It takes `du0` from the problem for a `DAEProblem`, the current
`abstol` and, for `ShampineCollocationInit()` on an `ODEProblem`, the current `dt / 5`, as
OrdinaryDiffEq's does. When SNES fails the integrator finishes where it is with
`ReturnCode.InitialFailure`, and on an `ODEProblem` without a singular mass matrix it does
nothing. After a callback's `affect!` runs without calling
`derivative_discontinuity!(integrator, false)`, or its `initialize` calls
`derivative_discontinuity!(integrator, true)`, the integrator is initialized again with the
callback's `initializealg`, or the solve's when the callback has none, as OrdinaryDiffEq does.
So with the default a callback has to leave the algebraic equations satisfied or the solve
throws `CheckInitFailureError`, and under `BrownFullBasicInit()` the algebraic variables are
solved for again. On a `DAEProblem`, whose derivative PETSc keeps to itself, `CheckInit()`
after a callback takes a state the callback left alone as consistent and checks a changed one
against the problem's `du0`, which is right at `t0` only.
