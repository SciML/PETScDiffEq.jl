# PETScDiffEq.jl

Common interface bindings for the [PETSc](https://petsc.org) TS time integrators,
so PETSc's time steppers can be called through the standard SciML `solve`
interface alongside OrdinaryDiffEq.jl, Sundials.jl and ODEInterfaceDiffEq.jl.

This package is under construction. `PETSc.TSRK`, the explicit Runge-Kutta
family, is implemented and verified against the documented order of its
`"3bs"` and `"5dp"` subtypes:

```julia
using PETSc, PETScDiffEq, SciMLBase

f!(du, u, p, t) = (du[1] = -u[1]; nothing)
prob = SciMLBase.ODEProblem(f!, [1.0], (0.0, 1.0))
sol = SciMLBase.solve(prob, TSRK("5dp"); dt = 0.05)
```

`dt` is required. `reltol`/`abstol` enable PETSc's adaptive step controller;
without them the step size is fixed. Only in-place, real-valued, forward-time
`ODEProblem`s are accepted.

## Scope

PETSc's linear (KSP) and nonlinear (SNES) solvers are already reachable from
SciML through LinearSolve.jl and NonlinearSolve.jl. This package covers the
third layer, TS, which provides `arkimex`, `rosw`, `irk`, `dirk`, `bdf`,
`radau5`, `glee` and the `mprk` multirate methods.

Not yet implemented: every TS type besides `rk`, the `init`/`step!`/`solve!`
integrator interface, `SplitODEProblem`, analytic Jacobians, and MPI. The
next target is analytic Jacobians, which is what work-precision benchmarking
against OrdinaryDiffEq.jl on stiff problems requires.

## Relation to PETSc.jl

Hendrik Ranocha drafted a SciML interface for PETSc TS as a package extension
in [JuliaParallel/PETSc.jl#242](https://github.com/JuliaParallel/PETSc.jl/pull/242).
This package is a separate implementation, written at his suggestion. See
[SciML/OrdinaryDiffEq.jl#4451](https://github.com/SciML/OrdinaryDiffEq.jl/issues/4451)
for the discussion.

## License

MIT. See [LICENSE](LICENSE).
