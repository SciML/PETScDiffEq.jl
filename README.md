# PETScDiffEq.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/PETScDiffEq/stable/)

[![CI](https://github.com/SciML/PETScDiffEq.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SciML/PETScDiffEq.jl/actions/workflows/CI.yml)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

PETScDiffEq.jl wraps the time integrators of [PETSc](https://petsc.org)'s TS for the SciML
common interface, serially or distributed over MPI. The documentation is at
[docs.sciml.ai/PETScDiffEq](https://docs.sciml.ai/PETScDiffEq/stable/).

## Installation

```julia
using Pkg
Pkg.add("PETScDiffEq")
```

## Example

```julia
using PETScDiffEq, SciMLBase

function lorenz(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
prob = SciMLBase.ODEProblem(lorenz, [1.0, 0.0, 0.0], (0.0, 100.0))
sol = SciMLBase.solve(prob, TSRK("5dp"); abstol = 1e-8, reltol = 1e-8)
```

## Solvers

- `TSRK`, explicit Runge-Kutta
- `TSRosW`, Rosenbrock-W
- `TSImplicit`, backward Euler, Crank-Nicolson, theta and BDF
- `TSIRK`, Gauss-Legendre implicit Runge-Kutta
- `TSARKIMEX`, additive Runge-Kutta IMEX
- `TSDAE`, implicit methods for a `DAEProblem`
- `TSBasicSymplectic` and `TSAlpha2`, for second-order problems
- `TSGeneric`, any other PETSc `TSType` by name

Every solver takes `petsc_options`, PETSc command-line options for that solve.

## Documentation

- [Solver options](https://docs.sciml.ai/PETScDiffEq/stable/solver_options/)
- [Solvers](https://docs.sciml.ai/PETScDiffEq/stable/solvers/)
- [Second-order and partitioned problems](https://docs.sciml.ai/PETScDiffEq/stable/second_order/)
- [DAE initialization](https://docs.sciml.ai/PETScDiffEq/stable/dae_initialization/)
- [Number types](https://docs.sciml.ai/PETScDiffEq/stable/number_types/)
- [Adjoint sensitivities](https://docs.sciml.ai/PETScDiffEq/stable/adjoint/)
- [MPI](https://docs.sciml.ai/PETScDiffEq/stable/mpi/)
- [Limitations](https://docs.sciml.ai/PETScDiffEq/stable/limitations/)

## License

MIT. See [LICENSE](LICENSE).
