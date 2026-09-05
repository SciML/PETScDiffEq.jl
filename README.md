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
`TSGeneric`) when the problem is not split. `TSRK` and an explicit
`TSGeneric` ignore `jac`, since they never form one.

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
Only in-place, real-valued, forward-time
`ODEProblem`s are accepted, and `SplitODEProblem` is accepted only by
`TSARKIMEX`. Without an `ODEFunction` `jac` (see above), every implicit
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

The integrator saves every accepted step and does not take `saveat` yet.
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
cannot honour, `tstops`, `save_idxs`, `callback`, `dense` and
`isoutofdomain` among them, emit a warning rather than being silently
dropped.

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

Not yet implemented: `saveat`, callbacks and `reinit!` through the integrator
interface, continuous/dense output (`sol(t)` falls back to linear interpolation between
saved points), `tstops`, `save_idxs`, an analytic Jacobian for
`SplitODEProblem`, and MPI — every solve currently runs on `MPI.COMM_SELF`. `TSIRK` still fails even with a `jac_prototype`
supplied: PETSc factorizes its coupled-stage Jacobian as a `seqkaij`
(Kronecker AIJ) matrix, not the plain AIJ this package builds, so it needs
its own dedicated setup. `TSGeneric("radau5")` fails for a different,
unexamined reason.

## License

MIT. See [LICENSE](LICENSE).
