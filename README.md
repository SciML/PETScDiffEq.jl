# PETScDiffEq.jl

Common interface bindings for the [PETSc](https://petsc.org) TS time integrators,
so PETSc's time steppers can be called through the standard SciML `solve`
interface alongside OrdinaryDiffEq.jl, Sundials.jl and ODEInterfaceDiffEq.jl.

This package is under construction. Four algorithm types are implemented and
verified against their documented order:

- `PETSc.TSRK`, explicit Runge-Kutta (`"3bs"` order 3, `"5dp"` order 5)
- `PETSc.TSRosW`, linearly implicit Rosenbrock-W (`"ra34pw2"` order 3)
- `PETSc.TSImplicit`, fully implicit `"beuler"` (order 1), `"cn"` (order 2),
  `"theta"` (order 2 at the default `theta = 0.5`), and `"bdf"`
- `PETSc.TSARKIMEX`, additive Runge-Kutta IMEX (`"3"` order 3), for a
  `SplitODEProblem` `u' = f1(u,p,t) + f2(u,p,t)` with `f1` stiff/implicit and
  `f2` non-stiff/explicit

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
```

A plain `ODEProblem` passed to `TSARKIMEX` is treated as fully implicit, with
the explicit part left at zero, matching PETSc's own default. `dt` is
required. `reltol`/`abstol` enable PETSc's adaptive step controller; without
them the step size is fixed. Only in-place, real-valued, forward-time
`ODEProblem`s are accepted, and `SplitODEProblem` is accepted only by
`TSARKIMEX`. `TSRosW`, `TSImplicit`, and `TSARKIMEX` have no analytic
Jacobian yet (see below), so pass `["-snes_fd"]` in `petsc_options` on
problems where PETSc's built-in coloring fallback is not enough.

## Scope

PETSc's linear (KSP) and nonlinear (SNES) solvers are already reachable from
SciML through LinearSolve.jl and NonlinearSolve.jl. This package covers the
third layer, TS, which provides `arkimex`, `rosw`, `irk`, `dirk`, `bdf`,
`radau5`, `glee` and the `mprk` multirate methods.

Not yet implemented: `irk`, `dirk`, `radau5`, `glee` and `mprk`; the
`init`/`step!`/`solve!` integrator interface; analytic Jacobians; and MPI.
The next target is analytic Jacobians, which is what work-precision
benchmarking against OrdinaryDiffEq.jl on stiff problems requires.

## Relation to PETSc.jl

Hendrik Ranocha drafted a SciML interface for PETSc TS as a package extension
in [JuliaParallel/PETSc.jl#242](https://github.com/JuliaParallel/PETSc.jl/pull/242).
This package is a separate implementation, written at his suggestion. See
[SciML/OrdinaryDiffEq.jl#4451](https://github.com/SciML/OrdinaryDiffEq.jl/issues/4451)
for the discussion.

## License

MIT. See [LICENSE](LICENSE).
