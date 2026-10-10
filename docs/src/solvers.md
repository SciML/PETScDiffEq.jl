# Solvers

- [`TSRK(subtype)`](@ref TSRK), explicit Runge-Kutta
- [`TSRosW(subtype)`](@ref TSRosW), linearly implicit Rosenbrock-W
- [`TSImplicit(subtype; order)`](@ref TSImplicit), backward Euler, Crank-Nicolson, theta and BDF
- [`TSIRK(nstages)`](@ref TSIRK), Gauss-Legendre implicit Runge-Kutta of order `2 * nstages`
- [`TSARKIMEX(subtype)`](@ref TSARKIMEX), additive Runge-Kutta IMEX, for a `SplitODEProblem`
- [`TSDAE(subtype)`](@ref TSDAE), the same implicit methods applied to a `DAEProblem`
- [`TSMPRK(slow, subtype)`](@ref TSMPRK), multirate partitioned Runge-Kutta
- [`TSBasicSymplectic(subtype)`](@ref TSBasicSymplectic), symplectic splitting methods for a `DynamicalODEProblem` or
  `SecondOrderODEProblem`
- [`TSAlpha2()`](@ref TSAlpha2), generalized-alpha for a `SecondOrderODEProblem`
- [`TSGeneric(ts_type)`](@ref TSGeneric), a pass-through to any other PETSc `TSType` by name

Each has a docstring covering its subtypes, whether it adapts and what it requires, so
`?TSRosW` at the REPL is the reference. One default worth knowing: PETSc's BDF is order 2,
so pass `TSImplicit("bdf"; order = 5)` when comparing against a higher-order method. Every solver takes `petsc_options`, a vector of
command-line style tokens passed to PETSc for that solve, which are parsed after the
options this package sets and so take precedence.

Only `TSARKIMEX` treats the two parts of a `SplitODEProblem` differently. The other solvers
that take an `ODEProblem` take a `SplitODEProblem` as the sum of its parts.
