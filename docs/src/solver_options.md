# Solver Options

The options available in `solve` are documented
[at the common solver options page](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/).
This package supports `dt`, `adaptive`, `dtmin`, `force_dtmin`, `dtmax`, `maxiters`, `reltol`
and `abstol` (either may be a vector of per-component tolerances), `saveat`, `save_everystep`,
`save_start`, `save_end`, `save_on`, `save_idxs`, `save_discretes`, `dense`, `callback`,
`tstops`, `d_discontinuities`, `unstable_check`, `isoutofdomain`, `calculate_error`,
`timeseries_errors`, `dense_errors`, `verbose` and `initializealg`. The warning a solve that
ends early gives is logged at the `instability` level of a `DEVerbosity`, so
`verbose = DEVerbosity(SciMLLogging.None())` silences it, as do `SciMLLogging.None()` and
`false`. Keywords it cannot honour emit a warning rather than being silently dropped.
`tstops` can also be a function `tstops(p, tspan)`, called once the problem is initialized
and again at each `reinit!`, as in OrdinaryDiffEq.

Saving follows OrdinaryDiffEq: a `saveat` keeps only its own points, adding `t0` or `tf`
only when it names them or `save_start` or `save_end` asks, and `save_everystep = true`
saves every step alongside it. A span with `tspan[1] == tspan[2]` takes no step: the
callbacks' `initialize` and `finalize` run, the state at `t0` is saved when `save_start` or
`save_end` asks, as OrdinaryDiffEq saves it, and the solve ends with `ReturnCode.Success`.
`PETScAdjoint` gives the gradient of the costs at `t0` there.
`dtmin` is a floor the solve keeps: once PETSc proposes a
smaller step, the solve ends there with `ReturnCode.DtLessThanMin`. A step shortened to
land on a stop or the final time does not count, and a fixed-step solve has no floor, as in
OrdinaryDiffEq. With `force_dtmin = true` the solve goes on at `dtmin` instead, and the floor
wins over a smaller `dtmax`. `dtmax` bounds every step of an adaptive solve, a first step
from `dt` and one from `set_proposed_dt!` included, as in OrdinaryDiffEq, while a fixed-step
solve steps at `dt`. PETSc's rule for landing on a stop or the final time is kept: a step
that would leave less than 1% of itself before one is stretched onto it, where
OrdinaryDiffEq takes the short step after it. `TSGeneric("glle")` picks its steps with its
own controller, which ignores `dtmax`. A NaN `dt`, `dtmin` or `dtmax`, an infinite `dt` and
a zero `dtmax` throw an `ArgumentError`, where OrdinaryDiffEq ends the solve with `DtNaN` or
`Unstable`, or takes an infinite `dt` as the whole span. `d_discontinuities` are stepped onto, and, as SciML defines them,
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
included. An integrator's running `sol.stats`, also `integrator.stats` as in OrdinaryDiffEq,
holds both as the steps go, and they start again at `reinit!`. `stats.nw` stays -1: PETSc
counts no matrices `shift * M - J`, builds
them itself under `AutoFiniteDiff()` and in `TSIRK`, and also asks the package's Jacobian
for `J` alone. `stats.nfpiter` and `stats.nfpconvfail` stay -1 as well.

A solve that stops short of the final time says why in its retcode: `Unstable` when the
state stops being finite, a step overflows, turns NaN or fails its Newton or linear solve at
every size tried, with a warning, an adaptive step is too small to move `t`, an adaptive
`TSAlpha2` step loses its velocity change to the position's floating point spacing, or
`unstable_check(dt, u, p, t)`
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
The exceptions are `TSRK`'s `"5f"`, `"5bs"`, `"6vr"`, `"7vr"` and `"8vr"`, whose interpolant in
PETSc is linear and on which the cubic Hermite loses three to five digits at tight
tolerances: their states come from a Hermite interpolant through the step's ends and one
point inside the step, two for `"7vr"` and `"8vr"`, which the method reaches with a step of
its own from the step's start. The points cost from 6 calls of `f` for `"5f"` to 26 for
`"8vr"`, once in each step a state is taken from, and dense output stays the cubic Hermite,
so for these `sol(t)` between saved points is less accurate than `saveat`.
With a mass matrix or a `DAEProblem` only PETSc's is available, and a type that has none
raises an `ArgumentError` when such a state is needed. A `ContinuousCallback` looks for a sign
change at `interp_points` times across every step, so with such a type it raises at the first
step even when it would never fire; with `interp_points = 0` it raises only once it fires, and
with `rootfind = NoRootFind` it fires at the step's end and never raises. PETSc's interpolant for
`TSARKIMEX`'s `"1bee"`, `"l2"` and the DIRK tableaux whose first stage is implicit, such as
`"8614a"`, starts from that stage rather than from the step's start, so those count as having
none, whether a subtype, `-ts_arkimex_type` or `-ts_dirk_type` picks them. With these,
`-ts_exact_final_time interpolate`, which takes the final state from that interpolant, raises an
`ArgumentError` on any problem. `integrator(t, Val{1})` is the slope
of the cubic Hermite interpolant through the step's ends, the interpolant's own derivative
where the package interpolates itself, and it raises with a mass matrix or a `DAEProblem`.
Past the step's end, where DiffEqCallbacks' `PositiveDomain` and `GeneralDomain` look,
`integrator(t)` extrapolates that cubic Hermite interpolant, as OrdinaryDiffEq extrapolates
its own, and it raises there with a mass matrix, a `DAEProblem` or a finished integrator.
`change_t_via_interpolation!` stays inside the step.
On a `DAEProblem`, `get_du` is the derivative its initialization or its last step paired with
the state, the slope at the step's end for `TSDAE("bdf")` and the step's chord for the other
subtypes, and it raises an `ArgumentError` once the state has changed since. With a mass
matrix `get_du` solves `M du = f(u)` at the state. Where `M` has a zero row, or is a sparse
one split over a communicator, it is instead the slope `TSImplicit("bdf")` or `"beuler"`
solved for at its last step's end, and it raises an `ArgumentError` with any other type or
once the state has changed since.
`integrator(t; idxs)`, a vector of times and `integrator(out, t)` take the forms
OrdinaryDiffEq's integrator does.

`ODEProblem`, `SplitODEProblem`, `DAEProblem`, `DynamicalODEProblem` and
`SecondOrderODEProblem` are supported, in place or out of place except for a `DAEProblem`,
whose residual has to be in place, along with `ODEFunction`'s `jac`, `jac_prototype` and
`mass_matrix`. A `mass_matrix` that is a multiple of `I`, as a `UniformScaling` or a
`ScalarOperator`, or a constant SciMLOperator such as a `MatrixOperator`, is taken as its
matrix. An operator that `SciMLOperators.isconstant` calls not constant, such as one with an
update function, is refused, and so is a matrix-free `FunctionOperator`. Supply a
`jac_prototype` for anything sparse: without one the Jacobian is dense and forces a dense
factorization.

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

A dense Jacobian, which a serial solve without a sparse `jac_prototype` or a `dm` has, is
factored directly, with `-ksp_type preonly -pc_type lu`. PETSc's own default, GMRES with
ILU(0), does the same factorization, since ILU(0) of a dense matrix is its full LU, and runs
a Krylov iteration on top. `TSRosW`, and any solve whose SNES is `ksponly`, keeps GMRES,
whose residual norm is what notices a NaN right-hand side so that a smaller step is tried.
A sparse `jac_prototype` in a serial solve without a `dm` is factored with LU in its natural
ordering when no entry lies more than 4 places from the diagonal, since the fill then stays
inside the band: on a 1-D Brusselator with u and v interleaved this is 1.4 to 2.7 times as
fast as GMRES with ILU(0). `TSRosW` and a `ksponly` solve wrap that LU in a Richardson
iteration, whose residual norm catches a NaN as GMRES's does. A wider sparse pattern keeps
GMRES with ILU(0), ordered by reverse Cuthill-McKee, since a direct factorization of a 2-D
grid fills in: on a 2-D Brusselator with 2048 states PETSc's LU took 5 to 9 times as long.
DM and distributed Jacobians keep GMRES with ILU(0), block Jacobi in parallel. A `-ksp_type`,
`-pc_type`, `-snes_mf` or `-snes_mf_operator`, or any other `-ksp_` or `-pc_` option but a
viewer, monitor or converged reason, in `petsc_options` replaces this choice.

PETSc's Newton solver forms a new Jacobian at every iteration. An adaptive BDF solve, through
`TSImplicit("bdf")` or `TSDAE`, sets `-snes_lag_jacobian 5`: one Jacobian per nonlinear
solve, formed again only after five iterations, since BDF's extrapolated guess starts close
and a failed solve is taken again with a smaller step. It halves the Jacobians on HIRES and
Robertson's problem at the same steps and errors. Other methods, and BDF at a fixed step,
keep a Jacobian per iteration: on Van der Pol and the Oregonator the lag cost
`TSARKIMEX`'s stages, which start from the previous stage, more steps, and it made a
fixed-step backward Euler solve fail. A `-snes_lag_jacobian` in `petsc_options` replaces it.

`DiscreteCallback`, `ContinuousCallback`, `VectorContinuousCallback` and `CallbackSet`
all work, as does the integrator interface through `init`, `step!`, `solve!`, `reinit!`,
`terminate!` and `initialize_dae!`. The exception is PETSc's `"glle"`, which PETSc runs only
as a whole solve, so `init` refuses it and so does a solve with a callback or `tstops`; see
[`TSGeneric`](@ref). After a step the running solution's retcode is `Success`, as
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
was given, where OrdinaryDiffEq's `reinit!` keeps them. The other options assigned on
`opts`, such as `maxiters`, `dtmax` or `save_start`, carry through `reinit!` as in
OrdinaryDiffEq. With `erase_sol = false`, a solution
saved under `save_idxs` without dense output stays without it when the new `saveat` would
otherwise turn it on, since a partial state gives no derivative to interpolate with.
`change_t_via_interpolation!` with `Val{true}` drops what was saved past the new time, and
saves the new end when `save_everystep` asks for every step. `resize!`, `deleteat!` and
`addat!` raise an `ArgumentError`, since PETSc sizes its vectors and solvers when the
integrator is made. So does a `reinit!` whose `t0` and `tf` run the other way in time from
the span the integrator was made for; make a new integrator for that direction.

A callback that names timeseries partitions in `saved_clock_partitions`, as the callbacks
ModelingToolkit builds for discrete variables do, has those parameters saved where
OrdinaryDiffEq saves them, so `sol.ps` and `sol(t; idxs)` give a discrete parameter's
history. A callback with `save_positions[2]` has its partitions saved at the start, unless
it sets `initialize_save_discretes = false` or `init` is given `initialize_save = false`,
and after each `affect!`, a `VectorContinuousCallback` saving the partitions of the events
that fired and a `ContinuousCallback` none after an `affect!` that clears the derivative
discontinuity. One with a `finalize` has them saved at the end when the end point is added
there, as under `save_everystep = false`. `save_discretes = false` disables the initial
save, saves after an `affect!`, and the finalizer's end-point save. An integrator's
`opts.save_discretes` can change between steps. `save_on = false` does not affect them.
`reinit!` starts the history again, where OrdinaryDiffEq keeps earlier values, and
continues it with `erase_sol = false`. On a communicator or with a `dm` each rank saves from its own
parameters. A `DAEProblem`'s solution has no storage for them in SciMLBase, so none are
saved there.
