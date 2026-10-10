# Adjoint sensitivities

With SciMLSensitivity loaded, `PETScAdjoint` computes gradients with PETSc's own discrete
adjoint, `TSAdjointSolve`, through `adjoint_sensitivities`:

```julia
using PETScDiffEq, SciMLBase, SciMLSensitivity

function f!(du, u, p, t)
    du[1] = -p[1] * u[1] + p[2] * u[1] * u[2]
    du[2] = p[3] * u[1] - p[4] * u[2]^2
end
function jac!(J, u, p, t)
    J[1, 1] = -p[1] + p[2] * u[2]
    J[1, 2] = p[2] * u[1]
    J[2, 1] = p[3]
    J[2, 2] = -2 * p[4] * u[2]
end
function paramjac!(pJ, u, p, t)
    fill!(pJ, 0.0)
    pJ[1, 1] = -u[1]
    pJ[1, 2] = u[1] * u[2]
    pJ[2, 3] = u[1]
    pJ[2, 4] = -u[2]^2
end
prob = ODEProblem(
    ODEFunction(f!; jac = jac!, paramjac = paramjac!),
    [1.0, 0.5], (0.0, 1.0), [0.7, 0.3, 0.4, 0.2],
)
sol = solve(prob, TSRK("4"); dt = 0.01, adaptive = false)

# The cost is the sum of |u(t)|^2 / 2 over these times.
ts = 0.0:0.1:1.0
dg!(out, u, p, t, i) = (out .= u)
du0, dp = adjoint_sensitivities(
    sol, TSRK("4"); sensealg = PETScAdjoint(),
    t = ts, dgdu_discrete = dg!, dt = 0.01, adaptive = false,
)
```

It works with `TSRK` of any subtype, `TSImplicit("beuler")`, `TSImplicit("cn")`,
`TSImplicit("theta")` with its `theta`, in its midpoint form or with `-ts_theta_endpoint`,
and `TSARKIMEX`. PETSc has no adjoint for `TSRosW`, `TSIRK`, `TSMPRK`, BDF,
`TSBasicSymplectic` or `TSAlpha2`. A `DynamicalODEProblem` or
`SecondOrderODEProblem` is differentiated on its flat `[v; u]`, on `MPI.COMM_SELF` only, with
the costs handed `ArrayPartition(v, u)` states as `solve` saves them and `du0` returned as
one.

It runs in PETSc's double real build. A `Float32` problem is differentiated there in
`Float64`, so `jac`, `paramjac` and the cost functions are handed `Float64` states, and the
gradients come back as `Float32` where `u0` and `p` are. A complex state is refused.

The gradient is that of the solution PETSc computes at these steps. It agrees with finite
differences of the same fixed-step `solve`, and it differs from a continuous adjoint such as
`GaussAdjoint` by the discretization error, which shrinks at the method's order. Because
`PETScAdjoint` runs the forward solve again and takes only the problem from `sol`, the
keywords that set the steps, `dt`, `adaptive`, `abstol`, `reltol`, `dtmin`, `dtmax` and
`maxiters`, have to be passed to `adjoint_sensitivities` exactly as they were to `solve`.

With fixed steps every cost time must be a time the solve steps to, since PETSc's adjoint
has no derivative of interpolation. An adaptive solve takes cost times anywhere in `tspan`.
`PETScAdjoint` gives them to PETSc as its time span, `TSSetTimeSpan`, and the step-size
controller ends a step on each one, so a loss summed over data times needs only the
tolerances repeated:

```julia
data_t = [0.25, 0.5, 0.75, 1.0]
sol = solve(prob, TSRK("5dp"); abstol = 1e-8, reltol = 1e-8, saveat = data_t)
du0, dp = adjoint_sensitivities(
    sol, TSRK("5dp"); sensealg = PETScAdjoint(),
    t = data_t, dgdu_discrete = dg!, abstol = 1e-8, reltol = 1e-8,
)
```

The gradient holds the accepted step sizes fixed rather than differentiating the step-size
controller: it agreed to within 2e-9 with central differences of a solve repeating those
steps at fixed size, for `TSRK` and `TSARKIMEX` alike. The steps are those of the adjoint's
own forward solve. `solve` with the same tolerances does not end a step on its `saveat`
times but interpolates to them, so the states it saves differ from the ones the cost is
taken on by the error of the two solves; at tolerances of 1e-5 the two sums of
`|u|^2 / 2` over eleven times differed by 1.2e-7 of their value. `tstops` do not make
`solve` take the adjoint's steps either, since `solve` steps to them in its own loop.

`TSARKIMEX` is the one stiff family `PETScAdjoint` takes that has an error estimate, so the
one stiff method whose adaptive solve it differentiates. It takes a `SplitODEProblem` as
well, on `MPI.COMM_SELF`: the implicit part's `jac` and `paramjac` are the problem's, which
a `SplitODEProblem` takes from `f1`, and the explicit part's are those of `f2`'s own
`ODEFunction`, each built by automatic differentiation when missing. Two things are
refused, because PETSc's ARKIMEX adjoint cannot do them. It has no quadrature, so an
integral cost stops with "No method adjointintegral". And with `-ts_arkimex_fully_implicit`
on a `SplitODEProblem` the solve takes `f2` implicitly while the adjoint still takes it
explicitly, which put the gradient 24% off in the test problem. With any other algorithm,
which solves a `SplitODEProblem` as the sum of its parts, `PETScAdjoint` refuses the problem;
give it the `ODEProblem` of the sum.

An integral cost, the integral of `g(u, p, t)` over `tspan`, goes through PETSc's quadrature
`TS`, which sums it with the method's own stages: `dt * b[i] * g` at each stage of a `TSRK`,
`dt * g` at the end of each backward Euler step, the trapezoidal sum for Crank-Nicolson, and
for the theta method `dt * g` at its stage, at `t + theta * dt`, or
`dt * ((1 - theta) * g(t) + theta * g(t + dt))` in its endpoint form. Its gradient is exact
for that sum and differs from `QuadratureAdjoint`'s or `InterpolatingAdjoint`'s by the
discretization error, again shrinking at the method's order. Give `g`, whose derivatives are
then taken with ForwardDiff or the algorithm's `autodiff`, or
`dgdu_continuous(out, u, p, t)` with `dgdp_continuous(out, u, p, t)`; with `dgdu_continuous`
alone the direct dependence on `p` is taken as zero, as SciMLSensitivity takes it. Discrete
and integral costs can be given together, and the gradients add:

```julia
g(u, p, t) = sum(abs2, u) / 2 + p[2] * u[1] * u[2]
du0, dp = adjoint_sensitivities(
    sol, TSRK("4"); sensealg = PETScAdjoint(), g, dt = 0.01, adaptive = false,
)
```

`TSImplicit` and `TSARKIMEX` solve transposed linear systems with the Krylov solver their
Newton steps use, by default GMRES with ILU(0) stopping at a relative residual of 1e-5, so
the gradient can be off by up to about that tolerance while the forward states are far more
accurate. With Crank-Nicolson and default options, the gradient's relative error was 4e-8
for a 2-D heat equation on a 7 by 7 grid, 2.5e-6 on a 24 by 24 grid and 1.1e-5 for
advection-diffusion on a 16 by 16 grid, while the forward states were within 8e-12, 5e-10
and 1e-9 of a direct solve. A 1-D heat equation with 50 unknowns gave the same gradient as a
direct solve to 2e-15, since ILU(0) of a tridiagonal matrix is exact. Passing
`PETScAdjoint(petsc_options = ["-ksp_type", "preonly", "-pc_type", "lu"])` removed the
difference in every case. For a problem too large to factor, tighten `-ksp_rtol` instead;
`1e-10` brought the two larger grids to 1e-11 and 5e-11.

It runs distributed over a `comm` and on a DMDA as well, needing a `jac` and `paramjac` that
fill each rank's rows, as the [MPI](@ref) section describes. Callbacks, `tstops`, mass
matrices and `DAEProblem` are refused. Passing `sensealg = PETScAdjoint()` to `solve` itself
does nothing until a reverse-mode AD package differentiates that `solve`, which the last
section of this page describes.

`jac` and `paramjac` go into the gradient unchecked, so a wrong entry gives a wrong
gradient. Compare them against central differences of `f!` at a representative point
before relying on the result:

```julia
function jacobian_errors(f!, jac!, paramjac!, u, p, t; h = 1.0e-6)
    n, m = length(u), length(p)
    J, pJ = zeros(n, n), zeros(n, m)
    jac!(J, u, p, t)
    paramjac!(pJ, u, p, t)
    fp, fm = similar(u), similar(u)
    for j in 1:n
        e = zeros(n)
        e[j] = h
        f!(fp, u + e, p, t)
        f!(fm, u - e, p, t)
        J[:, j] .-= (fp - fm) / 2h
    end
    for k in 1:m
        e = zeros(m)
        e[k] = h
        f!(fp, u, p + e, t)
        f!(fm, u, p - e, t)
        pJ[:, k] .-= (fp - fm) / 2h
    end
    return maximum(abs, J), maximum(abs, pJ)
end
jacobian_errors(f!, jac!, paramjac!, [1.0, 0.5], [0.7, 0.3, 0.4, 0.2], 0.3)
```

Both errors should be near round-off, around 1e-10 here; a wrong entry shows up at its own
size.

## Differentiating `solve`

`PETScAdjoint` is also the `sensealg` of `solve` itself for Zygote, and for other
reverse-mode packages that take their rules from ChainRules, so a fitting loss needs no cost
functions:

```julia
using Zygote

data_t = 0.0:0.1:1.0
truth = remake(prob; p = [0.77, 0.27, 0.48, 0.16])
data = Array(solve(truth, TSRK("5dp"); abstol = 1e-10, reltol = 1e-10, saveat = data_t))
function loss(p)
    sol = solve(
        prob, TSRK("5dp"); p, saveat = data_t, abstol = 1e-8, reltol = 1e-8,
        sensealg = PETScAdjoint(),
    )
    return sum(abs2, Array(sol) .- data)
end
Zygote.gradient(loss, [0.7, 0.3, 0.4, 0.2])
```

`solve` runs as it does without the `sensealg` and returns the same solution. When the
gradient is taken, the times `solve` saved become the cost times, the loss's derivative with
respect to each saved state becomes `dgdu_discrete`, and the adjoint described above runs
with the keywords `solve` was given: its own forward solve, then PETSc's adjoint. A gradient
took 3.1 times as long as the solve with `TSRK("5dp")` and 2.7 times with `TSARKIMEX("3")`
in the test problem. Gradients come back for `u0` and `p`. The loss can be written on
`Array(sol)`, `sol.u[i]`, `sol[:, i]` or `sol[i, j]`, with `saveat`, `save_start`,
`save_end`, `save_everystep` and `save_idxs` as `solve` takes them, and saved times it does
not depend on are left out of the adjoint's run.

With fixed steps the saved times have to be step ends, and the gradient is that of the loss
as `solve` computes it: it agreed with central differences of the loss to 3e-10 for
`TSRK("4")`, Crank-Nicolson, backward Euler, the theta method and `TSARKIMEX("3")` on a
`SplitODEProblem`.

For an adaptive solve the two runs differ. `solve` interpolates to its `saveat` times and
the adjoint's forward solve ends a step on each, so the gradient pairs the loss's derivative
at the first states with the sensitivities of the second. In the test problem, at tolerances
of 1e-4, 1e-6, 1e-8 and 1e-10, the two sets of states were at most 2.1e-6, 3.1e-7, 9.8e-9
and 2.0e-10 apart over eleven save times for `TSRK("5dp")`, and 1.2e-4, 5.6e-7, 1.1e-9 and
2.9e-12 for `TSARKIMEX("3")`. The gradient of the fitting loss was then within 3.5e-6,
3.6e-7, 1.1e-8 and 3.0e-11 of a reference, relative to its norm, for `TSRK("5dp")`, and
within 4.4e-4, 1.3e-5, 1.5e-7 and 1.6e-9 for `TSARKIMEX("3")`, the reference being
ForwardDiff through `Tsit5` at tolerances of 1e-13. Central differences of the same loss
were as far from that reference, 3.8e-6, 4.1e-7, 1.7e-8 and 3.7e-9 for `TSRK("5dp")` and
1.1e-3, 1.5e-5, 1.5e-7 and 1.5e-9 for `TSARKIMEX("3")`, so the mismatch stays at the size of
the solve's own error. The saved states could be taken from the adjoint's stepping instead,
by having `solve` end a step on each `saveat` time, but a differentiated `solve` would then
return other states than a plain one, so `solve` is left as it is.

Without `saveat` an adaptive solve saves its own steps, whose times move with `p`. The
gradient holds those times fixed, as it holds the step sizes. A loss on the last state is
unaffected, but for a loss summed over every step the gradient is that of the states at
those times, 3e-9 from the reference, while central differences of the loss, which let the
times move, were 5% away. Give `saveat` for a loss over several times.

A loss that calls `sol(t)` or reads `u0` or `p` from `sol.prob` is refused, since only the
saved states carry a derivative. A `Float32` problem, a `DynamicalODEProblem` or
`SecondOrderODEProblem`, a `comm` other than `MPI.COMM_SELF` and a `dm` are refused here
though `adjoint_sensitivities` takes them, and so are Enzyme, ReverseDiff, Tracker and
Mooncake, none of which this has been verified with. Everything `adjoint_sensitivities`
refuses is refused as well.
