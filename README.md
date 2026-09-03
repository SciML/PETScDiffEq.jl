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
Any other `jac_prototype` (including none) falls back to a dense matrix,
which is a real win on small and medium systems but is `O(n^2)` to fill.

A plain `ODEProblem` passed to `TSARKIMEX` is treated as fully implicit, with
the explicit part left at zero, matching PETSc's own default. `dt` is
required and sets the first step. Stepping is adaptive by default under
PETSc's own controller, so `dt` is not held fixed; pass `adaptive = false`
for fixed steps, and `reltol`/`abstol` to set the controller's tolerances.
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

## Saving

`saveat`, `save_everystep`, `save_start` and `save_end` behave as they do
elsewhere in SciML. `saveat` values that fall inside a step are produced with
`TSInterpolate`, so they carry the integrator's own order rather than a linear
fallback between stored steps.

```julia
sol = SciMLBase.solve(prob, TSRK("5dp"); dt = 0.1, saveat = [0.3, 0.7])
sol.t    # [0.0, 0.3, 0.7, 1.0]
```

`sol.stats` reports `nf`, `naccept`, `nreject`, `nnonliniter` and
`nnonlinconvfail`; the remaining `DEStats` fields stay at the `-1` "unknown"
sentinel. Keywords this package cannot honour (`tstops`, `save_idxs`,
`d_discontinuities`, `callback`, `dense`) emit a warning rather than being
silently dropped.

## Scope

PETSc's linear (KSP) and nonlinear (SNES) solvers are already reachable from
SciML through LinearSolve.jl and NonlinearSolve.jl. This package covers the
third layer, TS. Every dedicated PETSc TS family (`rk`, `rosw`, `beuler`,
`cn`, `theta`, `bdf`, `arkimex`) is covered above, and `TSGeneric` reaches
any other named type, `mprk` included, though only `"euler"` and `"alpha"`
have been run through this package's own convergence tests.

Not yet implemented: the `init`/`step!`/`solve!` integrator interface,
continuous/dense output (`sol(t)` falls back to linear interpolation between
saved points), `tstops`, `save_idxs`, callbacks, an analytic Jacobian for
`SplitODEProblem`, and MPI — every solve currently runs on `MPI.COMM_SELF`. `TSIRK` still fails even with a `jac_prototype`
supplied: PETSc factorizes its coupled-stage Jacobian as a `seqkaij`
(Kronecker AIJ) matrix, not the plain AIJ this package builds, so it needs
its own dedicated setup. `TSGeneric("radau5")` fails for a different,
unexamined reason.

## License

MIT. See [LICENSE](LICENSE).
