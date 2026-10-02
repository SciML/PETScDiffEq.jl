# DAE initialization

A `DAEProblem`, or an `ODEProblem` whose mass matrix has zero rows and columns, has to start
where its algebraic equations hold. The `initializealg` keyword picks how, with the algorithms
OrdinaryDiffEq takes, which come from DiffEqBase:

- `CheckInit()`, the default for a problem without initialization data, evaluates the
  residual at `u0`, and at `du0` for a `DAEProblem`, and throws a `CheckInitFailureError`
  when its RMS norm exceeds `abstol`, taken per component when `abstol` is a vector.
- `OverrideInit()`, the default for a problem that carries initialization data, as
  ModelingToolkit's do, solves the problem's own initialization system and takes the state
  and the parameters it gives. On a problem without such data it does nothing.
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
`reinit!` initializes again unless given `reinit_dae = false`.

On a communicator other than `MPI.COMM_SELF` they run wherever they run serially, and SNES
solves the whole state on the communicator, each rank its own rows, with the distributed
Jacobian of the solve: the problem's `jac` filling this rank's rows of the sparse
`jac_prototype`, or PETSc's colouring of that prototype, and with a `dm` the DM's colouring,
whether or not the problem has a `jac`. Its linear solves are GMRES with block Jacobi, one
ILU(0) block on each rank, to a relative tolerance of the square root of the precision's
spacing, and the solve's `petsc_options` do not reach them. A row `BrownFullBasicInit()` does
not solve keeps its variable there, so it takes a mass matrix whose zero columns are its zero
rows, as a `Diagonal` one's are, and refuses a sparse one where they differ. When SNES fails
the solve returns at `t0` with `ReturnCode.InitialFailure` on every rank, and an `f` or `jac`
that throws on some ranks makes every rank throw, as in a solve.

`OverrideInit()` goes through SciMLBase's `get_initial_values`, as OrdinaryDiffEq's does, and
PETSc's SNES solves the initialization system with a finite-difference Jacobian, to the
solve's `abstol` unless `OverrideInit(; abstol)` gives its own, so no solver package has to
be loaded. SNES takes a `NonlinearProblem`, and a `NonlinearLeastSquaresProblem` with as many
equations as unknowns. An `SCCNonlinearProblem`, which ModelingToolkit builds for a fully
determined system, goes block by block through SCCNonlinearSolve, which ModelingToolkit
loads: SNES solves the nonlinear blocks and LinearSolve the linear ones. A system with more
or fewer equations than unknowns, which ModelingToolkit warns about, is refused, as is any
other kind of initialization problem; `OverrideInit(; nlsolve = alg)` hands the system to
that solver instead, for those a least-squares one from NonlinearSolve. When the solver
fails the solve returns at `t0` with `ReturnCode.InitialFailure`.

The parameters the initialization gives are the ones the solve uses, and the ones in
`sol.prob.p` and `integrator.p`. It runs where OrdinaryDiffEq runs it: at `init` and `solve`,
on an `ODEProblem` without a mass matrix too, in `reinit!`, in `initialize_dae!`, and after
a callback on a `DAEProblem` or a mass-matrix problem. The problem's hooks are given the
problem at `init`, `solve` and `reinit!` and the integrator afterwards. No check follows it,
as none does in OrdinaryDiffEq. It does not run on a communicator other than
`MPI.COMM_SELF`, and `PETScAdjoint` does not differentiate it; `initializealg = CheckInit()`
starts both from the values as given.

`initialize_dae!(integrator, initializealg)` runs the same on the integrator's current state
and time, with the `initializealg` the solve was given unless another is passed, and writes
the result into PETSc. It takes `du0` from the problem for a `DAEProblem`, the current
`abstol` and, for `ShampineCollocationInit()` on an `ODEProblem`, the current `dt / 5`, as
OrdinaryDiffEq's does. When SNES fails the integrator finishes where it is with
`ReturnCode.InitialFailure`, and on an `ODEProblem` without a singular mass matrix only
`OverrideInit()` does anything. After a callback's `affect!` runs without calling
`derivative_discontinuity!(integrator, false)`, or its `initialize` calls
`derivative_discontinuity!(integrator, true)`, the integrator is initialized again with the
callback's `initializealg`, or the solve's when the callback has none, as OrdinaryDiffEq does.
So with the default a callback has to leave the algebraic equations satisfied or the solve
throws `CheckInitFailureError`, and under `BrownFullBasicInit()` the algebraic variables are
solved for again. On a `DAEProblem`, whose derivative PETSc keeps to itself, `CheckInit()`
after a callback takes a state the callback left alone as consistent and checks a changed one
against the problem's `du0`, which is right at `t0` only.
