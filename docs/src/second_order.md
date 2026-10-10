# Second-order and partitioned problems

A `SecondOrderODEProblem`, `u'' = f(u', u, p, t)`, and a `DynamicalODEProblem`, whose state is
a velocity `v` and a position `u` with `v' = f1(v, u, p, t)` and `u' = f2(v, u, p, t)`, have
two PETSc integrators of their own. Every other serial algorithm solves them as the
first-order system `[v; u]' = [f1; f2]`, as OrdinaryDiffEq's general methods do.

`TSBasicSymplectic` steps the velocity and the position in turn with `f1` and `f2`, so, as for
OrdinaryDiffEq's symplectic methods, `f1` must not depend on `v` nor `f2` on `u`. Its
subtypes `"sieuler"`, `"velverlet"`, `"3"` and `"4"` are of order 1 to 4 and step at a fixed
`dt`, and their energy error stays bounded over long times where a non-symplectic method's
grows:

```julia
using PETScDiffEq, SciMLBase

pendulum!(ddu, du, u, p, t) = (ddu .= -sin.(u); nothing)
prob = SecondOrderODEProblem(pendulum!, [0.0], [2.0], (0.0, 1000.0))
sol = solve(prob, TSBasicSymplectic("4"); dt = 0.1)
energy(s) = s.x[1][1]^2 / 2 - cos(s.x[2][1])
maximum(abs(energy(s) - energy(sol.u[1])) for s in sol.u)  # 6.4e-6, as large at t = 1000 as at t = 500
```

`f1` is given the time of the positions it is evaluated at, so a force that depends on `t`
keeps each method's order; PETSc's own time for that update runs ahead of the positions.
`"velverlet"` then gives OrdinaryDiffEq's `VelocityVerlet` answer to round-off.

`TSAlpha2` is PETSc's implicit generalized-alpha method of order 2, for stiff second-order
systems such as structural dynamics. `radius`, from 0 to 1, is its spectral radius at an
infinite step: below 1 it damps the frequencies the step cannot resolve, which radius 1, the
default, carries on undamped. It adapts on PETSc's error estimate with scalar `abstol` and
`reltol`, or steps at `dt` with `adaptive = false`. Its Jacobian, from a `jac` or from
`autodiff` as for the other implicit families, is that of the first-order system, `2n` by
`2n`, and a sparse `jac_prototype` of that system keeps the `n` by `n` matrix PETSc factors
sparse, whether the Jacobian comes from a `jac` or is coloured, as for `TSImplicit`.

```julia
K, C = [1.0e6 0.0; 0.0 1.0], [200.0 0.0; 0.0 0.02]
spring!(ddu, du, u, p, t) = (ddu .= -K * u .- C * du; nothing)
prob = SecondOrderODEProblem(spring!, [0.0, 0.0], [1.0, 1.0], (0.0, 5.0))
sol = solve(prob, TSAlpha2(; radius = 0.5); dt = 0.1, adaptive = false)
```

The states come back as OrdinaryDiffEq returns them: `sol.u[i]`, `sol(t)`, `integrator.u` and
`get_du(integrator)` are `ArrayPartition(v, u)`, so `sol.u[i].x[2]` is the position, while
`save_idxs` indexes the flat `[v; u]` and saves plain vectors. Between steps `sol(t)` is the
cubic Hermite interpolant of the velocity and the position, which is also what OrdinaryDiffEq
gives for `VelocityVerlet`. `stats.nf` counts evaluations of `f1`, or of the whole system,
and `stats.nf2` those of `f2` alone.

These problems take no mass matrix. They run distributed over a `comm` as the [MPI](@ref) section
describes, though not with a `dm`. Both algorithms take a reversed `tspan`. PETSc only steps
forward, so `TSAlpha2` then integrates `w(s) = u(-s)`, whose velocity is `-u'`, and gives back
`u'`: the states, `jac` and callbacks are those of the problem as written. `PETScAdjoint`
differentiates them through the first-order form with `TSRK`, `TSARKIMEX` or `TSImplicit`'s
`"beuler"`, `"cn"` or `"theta"`, on a `comm` as well; PETSc has no adjoint for
`TSBasicSymplectic` or `TSAlpha2`.
