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
- `PETSc.TSGeneric`, a pass-through to any other PETSc TS type by name
  (`"euler"` order 1, `"alpha"` order 2 tested), for types without a
  dedicated wrapper such as `"glle"` and `"glee"`. Pass `explicit = true` for
  a type that registers an RHS function rather than an IFunction.

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
```

An `ODEFunction`'s `jac` is used automatically by every implicit algorithm
(`TSRosW`, `TSImplicit`, `TSARKIMEX`'s implicit part, and a non-explicit
`TSGeneric`) when the problem is not split; it is stored as a dense matrix,
so this is a win on small and medium systems and does not yet help the
sparse case `jac_prototype` is for. `TSRK` and an explicit `TSGeneric`
ignore `jac`, since they never form one.

A plain `ODEProblem` passed to `TSARKIMEX` is treated as fully implicit, with
the explicit part left at zero, matching PETSc's own default. `dt` is
required. `reltol`/`abstol` enable PETSc's adaptive step controller; without
them the step size is fixed. Only in-place, real-valued, forward-time
`ODEProblem`s are accepted, and `SplitODEProblem` is accepted only by
`TSARKIMEX`. Without an `ODEFunction` `jac` (see above), every implicit
solve relies on PETSc's own fallback, so pass `["-snes_fd"]` in
`petsc_options` on problems where that fallback is not enough.

## Scope

PETSc's linear (KSP) and nonlinear (SNES) solvers are already reachable from
SciML through LinearSolve.jl and NonlinearSolve.jl. This package covers the
third layer, TS. Every dedicated PETSc TS family (`rk`, `rosw`, `beuler`,
`cn`, `theta`, `bdf`, `arkimex`) is covered above, and `TSGeneric` reaches
any other named type, `mprk` included, though only `"euler"` and `"alpha"`
have been run through this package's own convergence tests.

Not yet implemented: the `init`/`step!`/`solve!` integrator interface;
a sparse `jac_prototype` (the analytic Jacobian above is always dense);
an analytic Jacobian for `SplitODEProblem`; and MPI — every solve currently
runs on `MPI.COMM_SELF`. The next target is the sparse case, which is what
TSIRK and TSGeneric("radau5") need to become available through this
interface at all.

## License

MIT. See [LICENSE](LICENSE).
