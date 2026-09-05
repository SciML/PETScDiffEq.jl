# PETScDiffEq.jl

Common interface bindings for the [PETSc](https://petsc.org) TS time integrators,
so PETSc's time steppers can be called through the standard SciML `solve`
interface alongside OrdinaryDiffEq.jl, Sundials.jl and ODEInterfaceDiffEq.jl.

This package is under construction. Five algorithm types are implemented and
verified against their documented order:

- `PETSc.TSRK`, explicit Runge-Kutta (`"3bs"` order 3, `"5dp"` order 5)
- `PETSc.TSRosW`, linearly implicit Rosenbrock-W (`"ra34pw2"` order 3)
- `PETSc.TSImplicit`, fully implicit `"beuler"` (order 1), `"cn"` (order 2),
  `"theta"` (order 2 at the default `theta = 0.5`), and `"bdf"`
- `PETSc.TSARKIMEX`, additive Runge-Kutta IMEX (`"3"` order 3), for a
  `SplitODEProblem` `u' = f1(u,p,t) + f2(u,p,t)` with `f1` stiff/implicit and
  `f2` non-stiff/explicit
- `PETSc.TSGeneric`, a pass-through to any other PETSc TS type by name.
  Only `"euler"` (order 1) and `"alpha"` (order 2) are tested here. Pass
  `explicit = true` for a type that registers an RHS function rather than an
  IFunction. A TS type that needs its own subtype call, `"glee"` and `"glle"`
  among them, will not work through this pass-through: PETSc integrates
  nothing and the solve returns the initial condition.

```julia
using PETSc, PETScDiffEq, SciMLBase

f!(du, u, p, t) = (du[1] = -u[1]; nothing)
prob = SciMLBase.ODEProblem(f!, [1.0], (0.0, 1.0))
sol = SciMLBase.solve(prob, TSRK("5dp"); dt = 0.05)
sol = SciMLBase.solve(prob, TSRosW("ra34pw2"); dt = 0.05)
sol = SciMLBase.solve(prob, TSImplicit("bdf"); dt = 0.05)
sol = SciMLBase.solve(prob, TSImplicit("theta", 0.3); dt = 0.05)

f1!(du, u, p, t) = (du[1] = -50.0 * u[1]; nothing)
f2!(du, u, p, t) = (du[1] = 1.0; nothing)
split_prob = SciMLBase.SplitODEProblem(f1!, f2!, [1.0], (0.0, 0.1))
sol = SciMLBase.solve(split_prob, TSARKIMEX("3"); dt = 0.01)

sol = SciMLBase.solve(prob, TSGeneric("euler"; explicit = true); dt = 0.05)
sol = SciMLBase.solve(prob, TSGeneric("alpha"); dt = 0.05)

jac!(J, u, p, t) = (J[1, 1] = -1.0; nothing)
jac_prob = SciMLBase.ODEProblem(SciMLBase.ODEFunction(f!; jac = jac!), [1.0], (0.0, 1.0))
sol = SciMLBase.solve(jac_prob, TSImplicit("bdf"); dt = 0.05)

using SparseArrays
sparse_jac!(J, u, p, t) = (J[1, 1] = -1.0; nothing)
proto = sparse([1], [1], [1.0], 1, 1)
sparse_jac_prob = SciMLBase.ODEProblem(
    SciMLBase.ODEFunction(f!; jac = sparse_jac!, jac_prototype = proto), [1.0], (0.0, 1.0),
)
sol = SciMLBase.solve(sparse_jac_prob, TSImplicit("bdf"); dt = 0.05)
```

An `ODEFunction`'s `jac` is used automatically by every implicit algorithm
(`TSRosW`, `TSImplicit`, `TSARKIMEX`'s implicit part, and a non-explicit
`TSGeneric`). `TSRK` and an explicit `TSGeneric` ignore `jac`, since they
never form one. On a `SplitODEProblem` the `jac` is the Jacobian of `f1`, the
implicit part, following SciMLBase's `SplitFunction` convention; give it either
on the `SplitFunction` itself or on an `ODEFunction` wrapping `f1`, and
`TSARKIMEX` uses it for its implicit stages. Without it PETSc differences `f1`.

A `SparseMatrixCSC` `jac_prototype` is used as the PETSc matrix's sparsity
pattern, so `jac!` only has to fill the pattern's own nonzero entries; the
`shift*I` term always touches the diagonal, so this package extends the
declared pattern with the full diagonal itself before handing it to PETSc.
Any other `jac_prototype` (including none) falls back to a dense matrix.

**Supply a `jac_prototype` for anything sparse.** A dense matrix forces a dense
factorization, so on a tridiagonal problem the dense path loses to no Jacobian
at all above roughly `n = 40`, while the sparse path keeps winning and pulls
further ahead with size. Measured on a 1-D heat equation, `TSImplicit("bdf")`,
fixed `dt`, against the same solve with no `jac`:

| n | no `jac` | dense `jac` | sparse `jac` |
|---|---|---|---|
| 100 | 8.0 ms | 22.3 ms | **3.0 ms** |
| 400 | 62.5 ms | 381.4 ms | **11.4 ms** |

The dense path is for small or genuinely dense systems.

A plain `ODEProblem` passed to `TSARKIMEX` is treated as fully implicit, with
the explicit part left at zero, matching PETSc's own default. `dt` is
required and sets the first step. Pass `adaptive = false` for fixed steps,
and `reltol`/`abstol` to set the controller's tolerances.

Only `TSRK`, `TSRosW`, `TSARKIMEX` and `TSImplicit("bdf")` carry an embedded
error estimate, so only those adapt. `"beuler"`, `"cn"`, `"theta"` and
PETSc types such as `"alpha"` and `"euler"` have none: they step at the `dt`
you give and ignore `reltol`/`abstol` entirely. Passing tolerances to one of
those emits a warning rather than quietly doing nothing.
Real-valued, forward-time `ODEProblem`s are accepted, and `SplitODEProblem`
only by `TSARKIMEX`. An out-of-place `f(u, p, t)` is wrapped into the in-place
form PETSc needs, so it costs one array per evaluation and the saved states
come back as plain `Vector{Float64}` whatever `u0` was; an out-of-place `jac`
is wrapped the same way, and one that returns an entry the `jac_prototype`
does not declare is rejected rather than silently misplacing later entries. Without an `ODEFunction` `jac` (see above), every implicit
solve relies on PETSc's own fallback, so pass `["-snes_fd"]` in
`petsc_options` on problems where that fallback is not enough.

## Mass matrices

An `ODEFunction` `mass_matrix` is honoured by the implicit algorithms: the
residual becomes `M*u' - f` and the Jacobian `shift*M - J`, which is PETSc's
own `IFunction`/`IJacobian` form. A singular `M` therefore gives an index-1
DAE rather than an ODE.

```julia
M = [2.0 0.0; 0.0 1.0]
prob = SciMLBase.ODEProblem(
    SciMLBase.ODEFunction(f!; mass_matrix = M), [1.0, 1.0], (0.0, 1.0),
)
sol = SciMLBase.solve(prob, TSImplicit("bdf"); dt = 0.005)
```

An explicit algorithm cannot apply `M` without inverting it, so `TSRK` and an
explicit `TSGeneric` reject a non-identity mass matrix rather than silently
integrating `u' = f`. A mass matrix on a `SplitODEProblem` is also rejected.

## Integrator interface

`init`, `step!`, `solve!` and `done` work, and drive PETSc one `TSStep` at a
time. `step!(integ)` takes one step; `step!(integ, dt)` steps until `integ.t`
has advanced by `dt`; `solve!` runs to the end and returns the same
`ODESolution` that `solve` would. `integ.t`, `integ.u`, `integ.tprev`,
`integ.uprev` and `integ.dt` are readable between steps.

```julia
integ = SciMLBase.init(prob, TSRK("5dp"); dt = 0.1)
SciMLBase.step!(integ)
integ.t, integ.u
sol = SciMLBase.solve!(integ)
```

`saveat`, `save_everystep`, `save_start` and `save_end` behave as they do for
`solve`, interpolating with `TSInterpolate` inside the step just taken.

`DiscreteCallback`, `ContinuousCallback` and a `CallbackSet` of them work,
through both `solve` and the integrator: the condition is checked after every
step, `affect!` may change `integ.u` or call `terminate!`, and the changed
state is written back into PETSc's solution vector with `TSRestartStep`, so a
multistep method drops the history it took before the jump.

`ContinuousCallback` locates its event by bisecting `TSInterpolate` inside the
step that crossed, rolls the integrator back to the root, applies `affect!` or
`affect_neg!` according to the crossing direction, and restarts the stepper
there. `rootfind`, `interp_points`, `abstol`, `repeat_nudge` and
`save_positions` are honoured. On a bouncing ball the first bounce is located
to machine precision and each flight is the restitution coefficient times the
one before it. `VectorContinuousCallback` is rejected, since it needs one root
per component.

`tstops` makes the integration land on the given times exactly, through
`solve` and the integrator alike, and `add_tstop!`, `has_tstop`, `first_tstop`
and `pop_tstop!` manage them while stepping, so `PresetTimeCallback` from
DiffEqCallbacks works. The step that reaches a stop is shortened to land on it
and the previous step size resumes afterwards, so stops may sit closer together
than `dt`. A stop beyond the problem's `tspan` is refused, since this package
cannot integrate past it.

`reinit!(integ, u0; t0, tf, erase_sol, saveat, tstops, reinit_callbacks,
initialize_save)` restarts an integrator. It rebuilds the PETSc objects from
the keywords given to `init`, so the restarted solve is identical to a fresh
one, `dt` starts again at the value given to `init`, and it works on an
integrator that has already finished.
PETSc resources are released when the integration finishes, or on
`terminate!(integ)`, which stops early with `ReturnCode.Terminated`. Finish
or terminate every integrator you start: one dropped part-way is released by
a finalizer, and if that finalizer only runs at process exit, after MPI has
shut down, PETSc.jl's own object finalizers print an MPI warning and the
process exits non-zero.

## Saving

`saveat`, `save_everystep`, `save_start` and `save_end` behave as they do
elsewhere in SciML. `saveat` values that fall inside a step are produced with
`TSInterpolate`, so they carry the integrator's own order rather than a linear
fallback between stored steps.

```julia
sol = SciMLBase.solve(prob, TSRK("5dp"); dt = 0.1, saveat = [0.3, 0.7])
sol.t    # [0.0, 0.3, 0.7, 1.0]
```

`dtmin` and `dtmax` bound the controller's step size. `sol.stats` reports
`nf`, `naccept`, `nreject`, `nnonliniter` and `nnonlinconvfail`; the remaining
`DEStats` fields stay at the `-1` "unknown" sentinel. Keywords this package
cannot honour, `save_idxs` and `isoutofdomain` among them, emit a warning
rather than being silently dropped.

`sol(t)` interpolates. By default, when every step is saved and no `saveat`
is given, the solution carries a cubic Hermite interpolant built from the
saved states and one extra `f` evaluation per saved point, counted in
`sol.stats.nf`. This is how Sundials.jl does it too, since PETSc's own
`TSInterpolate` only covers the step in flight. Pass `dense = false` to skip
the extra evaluations and fall back to linear interpolation, or `dense = true`
to get Hermite through `saveat` points. Dense output is not available with a
mass matrix.

## Performance

Measured on a 1-D heat equation discretised to `n = 200`, `reltol = 1e-10`, with the
same sparse analytic Jacobian given to both sides:

| tspan | PETSc `TSRosW` | OrdinaryDiffEq `Rodas5P` |
|---|---|---|
| 0.1 | 0.39 ms | 0.20 ms |
| 10 | 8.6 ms | 0.50 ms |
| 100 | 82 ms | 1.6 ms |

PETSc costs roughly 100 microseconds per step here against a few microseconds for
OrdinaryDiffEq, and because that cost is per step rather than per solve it does not
amortise: the gap widens as the integration lengthens. Most of it is not the
right-hand-side callback, which is about 2 microseconds, nor the choice of `KSPType`
or `PCType`, which barely moves it.

This is PETSc used well outside the regime it is built for. PETSc TS exists for
large distributed problems, and this package is serial only, so the comparison above
is not evidence about PETSc's integrators. Use OrdinaryDiffEq.jl for serial problems
of this size. What this package is for is reaching PETSc's time steppers from the
SciML interface, and comparing schemes that have no Julia implementation.

Within the package, supply an analytic Jacobian and a `jac_prototype`: on the problem
above that is 11.7x faster per step than letting PETSc form the Jacobian by finite
differences (92 microseconds against 1067).

## Scope

PETSc's linear (KSP) and nonlinear (SNES) solvers are already reachable from
SciML through LinearSolve.jl and NonlinearSolve.jl. This package covers the
third layer, TS. Every dedicated PETSc TS family (`rk`, `rosw`, `beuler`,
`cn`, `theta`, `bdf`, `arkimex`) is covered above, and `TSGeneric` reaches
any other named type, `mprk` included, though only `"euler"` and `"alpha"`
have been run through this package's own convergence tests.

Not yet implemented: `VectorContinuousCallback`, `save_idxs`,
and MPI; every solve currently runs on
`MPI.COMM_SELF`. `TSIRK` still fails even with a `jac_prototype`
supplied: PETSc factorizes its coupled-stage Jacobian as a `seqkaij`
(Kronecker AIJ) matrix, not the plain AIJ this package builds, so it needs
its own dedicated setup. `TSGeneric("radau5")` fails for a different,
unexamined reason.

## License

MIT. See [LICENSE](LICENSE).
