# Solver Options

The options available in `solve` are documented
[at the common solver options page](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/).
This package supports `dt`, `adaptive`, `dtmin`, `force_dtmin`, `dtmax`, `reltol` and
`abstol` (either may be a vector of per-component tolerances), `saveat`, `save_everystep`,
`save_start`, `save_end`, `save_on`, `save_idxs`, `save_discretes`, `dense`, `callback`,
`tstops`, `d_discontinuities`, `unstable_check`, `isoutofdomain`, `timeseries_errors`,
`dense_errors`, `verbose` and `initializealg`. The warning a solve that ends early gives is logged at the
`instability` level of a `DEVerbosity`, so `verbose = DEVerbosity(SciMLLogging.None())`
silences it, as do `SciMLLogging.None()` and `false`. Keywords it cannot honour emit a
warning rather than being silently dropped.

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
`dtmin`. An adaptive step whose error estimate is NaN or infinite, as when an explicit
method's right-hand side returns NaN or the state overflows, is taken again smaller the same
way, and both kinds of retry count in `stats.nreject`. An adaptive implicit step whose Newton
or linear solve fails, as when its right-hand side returns NaN, or whose Newton matrix has a
zero pivot, is taken again at PETSc's `-ts_adapt_scale_solve_failed` share of its size, a
quarter by default, as many times as `maxiters` allows, and counts in
`stats.nnonlinconvfail` rather than `stats.nreject`, as OrdinaryDiffEq counts a failed Newton
solve. A method that runs no Newton iteration, which is `TSRosW` or any type given
`-snes_type ksponly`, counts these failures in `stats.nreject` instead, as OrdinaryDiffEq's
Rosenbrock methods do. A zero pivot in a Newton method still counts in
`stats.nnonlinconvfail`, where OrdinaryDiffEq counts it in `stats.nreject`. A fixed-step
solve ends at its first failed Newton or linear solve with `ConvergenceFailure`, as
OrdinaryDiffEq's Newton-based methods do with `adaptive = false`.

`stats.nsolve` counts the linear solves of the steps: one for each `KSPSolve` PETSc runs on
the TS's Krylov solver, whatever number of Krylov iterations it takes, so 20 fixed steps of
the four-stage `TSRosW("ra34pw2")` give 80 and an explicit method gives 0. It is -1 where
that count could miss a solve or cannot be taken: with a `-snes_type` other than `newtonls`,
`newtontr`, `ksponly` or `ksptransposeonly`, with a nonlinear preconditioner, and under
`-snes_ksp_ew`, whose tolerances PETSc sets in the hook the count uses. `stats.ncondition`
counts the calls of the callbacks' conditions, those of a `ContinuousCallback`'s root search
included. An integrator's running `sol.stats` holds both as the steps go, and they start
again at `reinit!`. `stats.nw` stays -1: PETSc counts no matrices `shift * M - J`, builds
them itself under `AutoFiniteDiff()` and in `TSIRK`, and also asks the package's Jacobian
for `J` alone. `stats.nfpiter` and `stats.nfpconvfail` stay -1 as well.

A solve that stops short of the final time says why in its retcode: `Unstable` when the
state stops being finite, a step overflows, turns NaN or fails its Newton or linear solve at
every size tried, with a warning, an adaptive step is too small to move `t`, or `unstable_check(dt, u, p, t)`
returns true, which is asked before each step with the step about to be taken, as
OrdinaryDiffEq asks it, `ConvergenceFailure` when a fixed-step nonlinear
solve fails, `DtLessThanMin` as above, `MaxIters` after `maxiters` step attempts, accepted,
rejected or failed, as OrdinaryDiffEq counts them, and
`Failure` for a zero pivot in a fixed-step solve, with a warning, or another step PETSc cannot
take. Where
`petsc_options` asks PETSc to raise, with `-ksp_error_if_not_converged`,
`-snes_error_if_not_converged` or `-ts_error_if_step_fails`, it raises instead.

A state between step ends, for `saveat`, `integrator(t)` or a `ContinuousCallback`, comes
from PETSc's own interpolant for `TSRK("5dp")`, `TSRosW("ra34pw2")`, `TSARKIMEX("4")` and
`"5"`, `TSImplicit("bdf")` and `TSDAE("bdf")`, and from the cubic Hermite interpolant dense
output uses for everything else, `TSGeneric` and a type `petsc_options` changes included.
With a mass matrix or a `DAEProblem` only PETSc's is available, and a type that has none
raises an `ArgumentError` when such a state is needed. `integrator(t, Val{1})` is the slope
of the cubic Hermite interpolant through the step's ends, the interpolant's own derivative
where the package interpolates itself, and it raises with a mass matrix or a `DAEProblem`.
`integrator(t; idxs)`, a vector of times and `integrator(out, t)` take the forms
OrdinaryDiffEq's integrator does.

`ODEProblem`, `SplitODEProblem`, `DAEProblem`, `DynamicalODEProblem` and
`SecondOrderODEProblem` are supported, in place or out of place, along with
`ODEFunction`'s `jac`, `jac_prototype` and `mass_matrix`. Supply a `jac_prototype` for
anything sparse: without one the Jacobian is dense and forces a dense factorization.

`TSARKIMEX` integrates the `f1` of a `SplitODEProblem` implicitly and its `f2` explicitly.
Every other algorithm solves it as the `ODEProblem` of `f1 + f2` and returns that problem's
states bit for bit. `stats.nf` then counts the evaluations of the sum and `stats.nf2` stays
zero, as with an OrdinaryDiffEq method that is not IMEX. A `SplitFunction`'s own `jac` and
`jac_prototype` are those of `f1`, and `f2` carries its own as an `ODEFunction`. The Jacobian
of the sum is the sum of the two `jac`s when both parts have one; a `jac` on one part alone
is not used, and the Jacobian is built as for an `ODEProblem` without one. Its pattern is the
union of the two `jac_prototype`s when both are sparse, and it is dense when either part has
none. The `mass_matrix` of the `SplitFunction` applies to the sum. An operator as either
part is refused, as it is with `TSARKIMEX`.

Without a `jac`, the implicit algorithms build the Jacobian with ForwardDiff, as
OrdinaryDiffEq does, and colour a sparse `jac_prototype`, so a tridiagonal problem costs
one dual evaluation of `f` per Jacobian rather than one evaluation per state. The
prototype has to hold every entry the Jacobian can have: one it leaves out is left out of
the Jacobian, which then costs Newton iterations. Passing
`autodiff = PETScDiffEq.AutoFiniteDiff()` to the algorithm leaves the Jacobian to PETSc's
finite differences, coloured by a sparse prototype too. On a badly scaled stiff problem such as
Robertson's, those are far enough off that the solve reports success with an answer
wrong in its first digit, so keep them for a right-hand side ForwardDiff cannot run. They are
the default with a `dm` or on a communicator, where `autodiff = PETScDiffEq.AutoForwardDiff()`
asks for the exact Jacobian: with a DMDA as the `dm` it is seeded by the DM's colouring, and
on a communicator by the colouring of a sparse `jac_prototype`.

`DiscreteCallback`, `ContinuousCallback`, `VectorContinuousCallback` and `CallbackSet`
all work, as does the integrator interface through `init`, `step!`, `solve!`, `reinit!`,
`terminate!` and `initialize_dae!`. After a step the running solution's retcode is `Success`, as
OrdinaryDiffEq's is. `check_error` gives `Success` while the integrator can go on and the
retcode it stopped with after that. In a callback's `finalize` it gives the retcode passed to
`terminate!`, and `Success` for a solve that ended any other way, where OrdinaryDiffEq's
also gives `MaxIters` or `Unstable`. `check_error!` and `postamble!` work as SciMLBase
defines them, `postamble!` finishing the integrator where it is. Like `terminate!`, it saves
the point it stops at whenever the final time would be saved; under a `saveat` that does not
name that point, OrdinaryDiffEq saves it only with `save_end = true` or when nothing is saved
yet. Unlike OrdinaryDiffEq's, a finished integrator cannot step again. `auto_dt_reset!` takes
the step `init` would take from the current state, and `reinit!` does the same with
`reset_dt = true`, or keeps the proposed step with `reset_dt = false`. A fixed-step
integrator goes on at its fixed size through both, as OrdinaryDiffEq's does, and only `dt`
shows the estimate. `get_proposed_dt` is signed, negative on a reversed span, and
`set_proposed_dt!` also takes another integrator whose proposed step it copies. `set_abstol!`
and `set_reltol!` hold for the steps after, until `reinit!` goes back to the tolerances `init`
was given, where OrdinaryDiffEq's `reinit!` keeps them. With `erase_sol = false`, a solution
saved under `save_idxs` without dense output stays without it when the new `saveat` would
otherwise turn it on, since a partial state gives no derivative to interpolate with.
`change_t_via_interpolation!` with `Val{true}` drops what was saved past the new time, and
saves the new end when `save_everystep` asks for every step. `resize!`, `deleteat!` and
`addat!` raise an `ArgumentError`, since PETSc sizes its vectors and solvers when the
integrator is made.

A callback that names timeseries partitions in `saved_clock_partitions`, as the callbacks
ModelingToolkit builds for discrete variables do, has those parameters saved where
OrdinaryDiffEq saves them, so `sol.ps` and `sol(t; idxs)` give a discrete parameter's
history. A callback with `save_positions[2]` has its partitions saved at the start, unless
it sets `initialize_save_discretes = false` or `init` is given `initialize_save = false`,
and after each `affect!`, a `VectorContinuousCallback` saving the partitions of the events
that fired and a `ContinuousCallback` none after an `affect!` that clears the derivative
discontinuity. One with a `finalize` has them saved at the end when the end point is added
there, as under `save_everystep = false`. `save_discretes = false` leaves out the saves
after an `affect!` and keeps the others, and an integrator's `opts.save_discretes` can
change between steps. `save_on = false` does not affect them. `reinit!` starts the history
again, where OrdinaryDiffEq's keeps the earlier values, and goes on from it with
`erase_sol = false`. On a communicator or with a `dm` each rank saves from its own
parameters. A `DAEProblem`'s solution has no storage for them in SciMLBase, so none are
saved there.
