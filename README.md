# PETScDiffEq.jl

Common interface bindings for the [PETSc](https://petsc.org) TS time integrators,
so PETSc's time steppers can be called through the standard SciML `solve`
interface alongside OrdinaryDiffEq.jl, Sundials.jl and ODEInterfaceDiffEq.jl.

This package is under construction and has no working solver interface yet.

## Scope

PETSc's linear (KSP) and nonlinear (SNES) solvers are already reachable from
SciML through LinearSolve.jl and NonlinearSolve.jl. This package covers the
third layer, TS, which provides `arkimex`, `rosw`, `irk`, `dirk`, `bdf`,
`radau5`, `glee` and the `mprk` multirate methods.

The first target is serial `ODEProblem` and `SplitODEProblem` support with
analytic Jacobians, which is what work-precision benchmarking against
OrdinaryDiffEq.jl requires.

## Relation to PETSc.jl

The interface work started as a package extension in
[JuliaParallel/PETSc.jl#242](https://github.com/JuliaParallel/PETSc.jl/pull/242)
by Hendrik Ranocha, which this package builds on. See
[SciML/OrdinaryDiffEq.jl#4451](https://github.com/SciML/OrdinaryDiffEq.jl/issues/4451)
for the discussion.

## License

MIT. See [LICENSE](LICENSE).
