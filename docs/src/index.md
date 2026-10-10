# PETScDiffEq.jl

PETScDiffEq.jl contains bindings for the [PETSc](https://petsc.org) TS time integrators
to allow them to be used with the SciML common interface. PETSc's linear and nonlinear
solvers are already reachable from SciML through LinearSolve.jl and NonlinearSolve.jl;
this package covers the third layer, TS.

## Installation

```julia
using Pkg
Pkg.add("PETScDiffEq")
```

Precompiling the package runs a few small solves through PETSc, so that the first `solve`
of a session compiles much less. To precompile without them:

```julia
using PETScDiffEq, Preferences
set_preferences!(PETScDiffEq, "precompile_workload" => false; force = true)
```

Precompiling under `mpiexec` or `srun` skips these solves, and later sessions reuse that
build until the package or one of its dependencies changes, so load the package once
without the launcher before the first parallel run. A build made under a launcher says so
when the package is next loaded outside one, once per session and never on the ranks of a
parallel run. To rebuild it with the solves, in a session started without the launcher:

```julia
using PETScDiffEq
Base.compilecache(Base.PkgId(PETScDiffEq))
```

Sessions started after that load the new build. The note is an `@info` record, so a logger
that drops `Info` hides it, and a build made with `precompile_workload` set to `false` never
gives it.

## Common API Usage

This library adds the common interface to PETSc's TS solvers, documented in the
[DifferentialEquations.jl documentation](https://docs.sciml.ai/DiffEqDocs/stable/).
Following the Lorenz example from
[the ODE tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/ode_example/):

```julia
using PETScDiffEq, SciMLBase

function lorenz(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = SciMLBase.ODEProblem(lorenz, u0, tspan)
sol = SciMLBase.solve(prob, TSRK("5dp"); dt = 0.01, abstol = 1e-8, reltol = 1e-8)
```

`dt` sets the first step. An adaptive solve can leave it out, and the first step is then
Hairer and Wanner's estimate as OrdinaryDiffEq uses it, taken with the `abstol` and `reltol`
keywords (SciML's defaults, `abstol = 1e-6` and `reltol = 1e-3`, when not given). DAE and mass-matrix
problems start from a small step instead. A solve PETSc steps at a fixed size needs `dt`:
`adaptive = false`, `-ts_adapt_type none`, the fixed-step families (`TSImplicit` and `TSDAE`
other than `bdf`, `TSIRK`, `TSMPRK`), and subtypes registered without an embedded error
estimate. What counts is the type PETSc runs after `petsc_options`: `TSImplicit("beuler",
["-ts_type", "bdf"])` runs without `dt` and `TSImplicit("bdf", ["-ts_type", "beuler"])` needs
it. A `TSGeneric` naming a type no other constructor covers needs it too, since whether that
type adapts is not known here.

`u0` is a vector or an array of any shape, of real or complex numbers. An array-shaped
state, such as the matrix of a 2-D grid, is stepped as PETSc's flat vector `vec(u0)` and
reshaped where it meets your code, without a copy: `f`, `jac`, `isoutofdomain`,
`unstable_check` and callbacks get arrays of `u0`'s size, and `sol.u`, `sol(t)`,
`integrator.u`, `integrator.uprev`, `get_du`, `set_u!` and `reinit!` take and return them.
The Jacobian a `jac` fills, a `jac_prototype` and a mass matrix act on `vec(u)`, so each is
`length(u0)` square, `save_idxs` and `TSMPRK`'s components are linear indices into the
state, and a tolerance for each component is an array of `u0`'s size or a vector of its
length. `set_u!` and `reinit!` also take a state of another shape with as many entries and
read it in linear order. An array type other than `Array`, such as a `Transpose`, is stepped
and given back as an `Array` of its size. The solve is the one of the same problem written
on `vec(u0)`, step for step. This covers an `ODEProblem` and a `SplitODEProblem` on
`MPI.COMM_SELF`; [Limitations](@ref) lists what still needs `vec(u0)`.

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```
