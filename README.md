# PETScDiffEq.jl

[![CI](https://github.com/SciML/PETScDiffEq.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SciML/PETScDiffEq.jl/actions/workflows/CI.yml)

This package contains bindings for the [PETSc](https://petsc.org) TS time integrators to
allow them to be used with the SciML common interface. PETSc's linear and nonlinear
solvers are already reachable from SciML through LinearSolve.jl and NonlinearSolve.jl;
this package covers the third layer, TS. For more information on using the solvers from
this package, see the
[DifferentialEquations.jl documentation](https://docs.sciml.ai/DiffEqDocs/stable/).

## Installation

```julia
using Pkg
Pkg.add("PETScDiffEq")
```

## Common API Usage

This library adds the common interface to PETSc's TS solvers.
[See the DifferentialEquations.jl documentation for details on the interface](https://docs.sciml.ai/DiffEqDocs/stable/).
Following the Lorenz example from
[the ODE tutorial](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/ode_example/), we can
solve this using `TSRK` via the following:

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
estimate. `TSGeneric` needs it too, since which of its types adapt is not known here.

## Solvers

- `TSRK(subtype)`, explicit Runge-Kutta
- `TSRosW(subtype)`, linearly implicit Rosenbrock-W
- `TSImplicit(subtype; order)`, backward Euler, Crank-Nicolson, theta and BDF
- `TSIRK(nstages)`, Gauss-Legendre implicit Runge-Kutta of order `2 * nstages`
- `TSARKIMEX(subtype)`, additive Runge-Kutta IMEX, for a `SplitODEProblem`
- `TSDAE(subtype)`, the same implicit methods applied to a `DAEProblem`
- `TSGeneric(ts_type)`, a pass-through to any other PETSc `TSType` by name

Each has a docstring covering its subtypes, whether it adapts and what it requires, so
`?TSRosW` at the REPL is the reference. One default worth knowing: PETSc's BDF is order 2,
so pass `TSImplicit("bdf"; order = 5)` when comparing against a higher-order method. Every solver takes `petsc_options`, a vector of
command-line style tokens passed to PETSc for that solve, which are parsed after the
options this package sets and so take precedence.

## Solver Options

The options available in `solve` are documented
[at the common solver options page](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/).
This package supports `dt`, `adaptive`, `dtmin`, `force_dtmin`, `dtmax`, `reltol` and
`abstol` (either may be a vector of per-component tolerances), `saveat`, `save_everystep`,
`save_start`, `save_end`, `save_on`, `save_idxs`, `dense`, `callback`, `tstops`,
`d_discontinuities`, `unstable_check` and `isoutofdomain`. Keywords it cannot
honour emit a warning rather than being silently dropped.

Saving follows OrdinaryDiffEq: a `saveat` keeps only its own points, adding `t0` or `tf`
only when it names them or `save_start` or `save_end` asks, and `save_everystep = true`
saves every step alongside it. `dtmin` is a floor the solve keeps: once PETSc proposes a
smaller step, the solve ends there with `ReturnCode.DtLessThanMin`. A step shortened to
land on a stop or the final time does not count, and a fixed-step solve has no floor, as in
OrdinaryDiffEq. With `force_dtmin = true` the solve goes on at `dtmin` instead, and the floor
wins over a smaller `dtmax`. `d_discontinuities` are stepped onto, and, as SciML defines them,
the step after each starts one ULP past it, so the right-hand side there sees the new regime
when written as `if t > t_d`. `isoutofdomain(u, p, t)` is asked after each step of an adaptive solve, and a
step that leaves the domain is taken again at a fifth of its size, as OrdinaryDiffEq takes it;
one that cannot be made small enough ends the solve with `Unstable`, or `DtLessThanMin` at
`dtmin`. An adaptive step whose error estimate is NaN or infinite, as when the
right-hand side returns NaN or the state overflows, is taken again smaller the same way, and
both kinds of retry count in `stats.nreject`. An adaptive implicit step whose Newton or
linear solve fails, or whose Newton matrix has a zero pivot, is taken again at PETSc's
`-ts_adapt_scale_solve_failed` share of its size, a quarter by default, as many times as it
takes, and counts in `stats.nnonlinconvfail` rather than `stats.nreject`, as in
OrdinaryDiffEq. A fixed-step solve ends at its first failed Newton solve with
`ConvergenceFailure`, as OrdinaryDiffEq's does with `adaptive = false`.

A solve that stops short of the final time says why in its retcode: `Unstable` when the
state stops being finite, a step overflows, turns NaN or fails its Newton solve at every size
tried, with a warning, an adaptive step is too small to move `t`, or `unstable_check(dt, u, p, t)`
returns true, which is asked before each step with the step about to be taken, as
OrdinaryDiffEq asks it, `ConvergenceFailure` when a fixed-step nonlinear
solve fails, `DtLessThanMin` as above, `MaxIters` when `maxiters` steps are taken, and
`Failure` for a zero pivot in a fixed-step solve, with a warning, or another step PETSc cannot
take. Where
`petsc_options` asks PETSc to raise, with `-ksp_error_if_not_converged`,
`-snes_error_if_not_converged` or `-ts_error_if_step_fails`, it raises instead.

A state between step ends, for `saveat`, `integrator(t)` or a `ContinuousCallback`, comes
from PETSc's own interpolant for `TSRK("5dp")`, `TSRosW("ra34pw2")`, `TSARKIMEX("4")` and
`"5"`, `TSImplicit("bdf")` and `TSDAE("bdf")`, and from the cubic Hermite interpolant dense
output uses for everything else, `TSGeneric` and a type `petsc_options` changes included.
With a mass matrix or a `DAEProblem` only PETSc's is available, and a type that has none
raises an `ArgumentError` when such a state is needed.

`ODEProblem`, `SplitODEProblem` and `DAEProblem` are supported, in place or out of
place, along with
`ODEFunction`'s `jac`, `jac_prototype` and `mass_matrix`. Supply a `jac_prototype` for
anything sparse: without one the Jacobian is dense and forces a dense factorization.

Without a `jac`, the implicit algorithms build the Jacobian with ForwardDiff, as
OrdinaryDiffEq does, and colour a sparse `jac_prototype`, so a tridiagonal problem costs
one dual evaluation of `f` per Jacobian rather than one evaluation per state. The
prototype has to hold every entry the Jacobian can have: one it leaves out is left out of
the Jacobian, which then costs Newton iterations. Passing
`autodiff = PETScDiffEq.AutoFiniteDiff()` to the algorithm leaves the Jacobian to PETSc's
finite differences, coloured by a sparse prototype too. On a badly scaled stiff problem such as
Robertson's, those are far enough off that the solve reports success with an answer
wrong in its first digit, so keep them for a right-hand side ForwardDiff cannot run.

`DiscreteCallback`, `ContinuousCallback`, `VectorContinuousCallback` and `CallbackSet`
all work, as does the integrator interface through `init`, `step!`, `solve!`, `reinit!`
and `terminate!`.

## Number types

A `Float64` state runs in PETSc's double-precision build and a `ComplexF64` state in its
double-precision complex build. A `Float32` or `ComplexF32` state runs in the matching
single-precision build when the span is `Float32` as well, following OrdinaryDiffEq's advice
to give a single-precision problem a `Float32` span, and in the double-precision build when
the span is `Float64`, with the saved states given back in single precision. DiffEqBase
promotes the span to the type of `dt`, so a single-precision solve takes `dt = 0.01f0`
rather than `dt = 0.01`, and it makes a whole-number span `Float64`. Any other real state,
whole numbers included, is solved in `Float64` and comes back in it. Where PETSc.jl has not
loaded the single-precision build, as with a library set with `PETSc.set_library!`, a
single-precision state runs in the double-precision one, and a problem whose build is not
loaded at all is refused with an `ArgumentError` that names it. On 32-bit x86 a
single-precision state always runs in the double-precision build: there PETSc_jll's
single-precision builds end BDF and ARKIMEX solves in failure at stops that its
double-precision build takes.

Times are in the type of the clock PETSc steps on: `Float32` for a single-precision state
with a `Float32` span and `Float64` otherwise. That covers `sol.t` and the integrator's `t`
and `dt`, and the integrator's `u` is in the type PETSc steps. OrdinaryDiffEq gives the
times in the span's type whatever the state; here a `Float64` state with a `Float32` span is
stepped on PETSc's `Float64` clock, and rounding its times to `Float32` would give states
close together the same time.

Single precision has seven digits, which bounds the clock as well as the state. The first
step, and the step after a stop, are at least four ulps of `t`, since PETSc cannot take a
step whose stages have no room between its ends; a single-precision solve is stepped by
this package's integrator, which lands on each stop and on the final time itself, where
PETSc's own landing would refuse a step under about `1e-6` before a final time below 1.
Tolerances finer than single precision can resolve, about `1e-7`, are accepted but buy
nothing past its rounding, and can drive PETSc's adaptive step below what the clock can
tell apart, which ends the solve with a failed retcode. PETSc's single build also takes the
norms its Newton and Krylov iterations stop on in single precision, and a vector whose
entries are all below about `1e-19` in size has a norm of zero there. An implicit solve of
such a state can then stop without moving it and still report success, so this package
warns when a solve starts on such a state, or fails on one; a state that decays there from
above is within any coarser tolerance. Rescale the problem, or give it a `Float64` span.

With a complex state, times, `dt`, `saveat`, `tstops` and the tolerances stay real, and
PETSc's error norms take each component's modulus; a tolerance given as a complex number
with a zero imaginary part is taken as its real part. A `ContinuousCallback`'s condition has
to return a real number, such as `real(u[1]) - 0.5`, since a root is a sign change. The
implicit methods' Newton iteration needs a holomorphic `f`, one that does not go through
`conj`, `abs`, `real` or `imag` of the state. ForwardDiff takes no complex numbers, so
without a `jac` the Jacobian is differentiated along the real parts of the state, which for
a holomorphic `f` is its complex Jacobian, and a sparse `jac_prototype`, or the pattern a
sparse backend is given, is coloured as for a real state. A check at the start compares the
derivatives along the real and the imaginary parts and refuses an `f` that is not
holomorphic; it is best effort, and can miss a term too small to show near the initial
state. `AutoFiniteDiff()` and a hand-written `jac` are not
checked. The explicit methods take any `f`.

## Adjoint sensitivities

With SciMLSensitivity loaded, `adjoint_sensitivities(sol, alg; sensealg = PETScAdjoint(), ...)`
runs PETSc's own discrete adjoint for `TSRK`, `TSImplicit("beuler")` and `TSImplicit("cn")`.
The keywords that set the steps have to be repeated from `solve`. It runs in PETSc's double
real build, so a `Float32` problem is differentiated in `Float64` and its gradients come
back as `Float32`, and a complex one is refused. `?PETScAdjoint` and the documentation cover
what it needs, what it refuses and how to check `jac` and `paramjac`.

## MPI

`TSRK`, `TSRosW`, `TSImplicit`, `TSIRK`, `TSDAE`, `TSARKIMEX` and
`TSGeneric(ts_type; explicit = true)` take a `comm` keyword. With a communicator
other than the default `MPI.COMM_SELF` the solve runs distributed over it: every rank of
`comm` calls `solve` with the same arguments, and `u0` is the block of the state that rank
owns, the blocks following each other in rank order. Each rank's `sol.u` holds its own rows,
and `sol.t` is the same on every rank.

`f(du, u, p, t)` sees only its rank's rows, so it fetches what it needs from the other ranks
itself, and it has to be collective: the package calls it the same number of times, in the
same order, on every rank. A 1-D heat equation with eight rows on each rank:

```julia
using MPI, PETScDiffEq, SciMLBase

MPI.Init()
comm = MPI.COMM_WORLD
rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
dx = 1 / (8nranks + 1)
x = (8rank .+ (1:8)) .* dx

function heat!(du, u, p, t)
    left = rank == 0 ? MPI.PROC_NULL : rank - 1
    right = rank == nranks - 1 ? MPI.PROC_NULL : rank + 1
    gl, gr = zeros(1), zeros(1)
    MPI.Sendrecv!(u[1:1], gr, comm; dest = left, source = right)
    MPI.Sendrecv!(u[end:end], gl, comm; dest = right, source = left)
    for i in eachindex(u)
        l = i == 1 ? gl[1] : u[i - 1]
        r = i == length(u) ? gr[1] : u[i + 1]
        du[i] = (l - 2u[i] + r) / dx^2
    end
end

sol = solve(ODEProblem(heat!, sinpi.(x), (0.0, 0.1)), TSRK("5dp"; comm))
```

Run it with the `mpiexec` MPI.jl provides, `MPI.mpiexec()`, as in `mpiexec -n 4 julia heat.jl`.
PETSc starts up collectively over `MPI.COMM_WORLD`, so under `mpiexec` a rank that solves on
its own, on `MPI.COMM_SELF` too, hangs unless an earlier solve or `PETSc.initialize` has
started PETSc on every rank.

`saveat`, `tstops`, `d_discontinuities`, a fixed `dt` and dense output work as in a serial
solve. Vector `abstol` and `reltol`, `save_idxs` and `p` are per rank.
`unstable_check` and `isoutofdomain` are asked on each rank's rows, and `true` on any rank
counts on all of them. When `f`, `jac` or one of those checks throws on some ranks, those
ranks go on with NaN until the ranks next agree, at the end of the step or when its nonlinear
solve fails, and then every rank throws, so an `f` that throws has to do so after its own
communication.

Callbacks and the integrator interface run distributed too, as long as every rank makes the
same calls with the same arguments in the same order: `init`, `step!`, `solve!`, `reinit!`,
`terminate!`, `set_u!`, `add_tstop!`, `add_saveat!`, `savevalues!`,
`change_t_via_interpolation!`, `set_proposed_dt!`, `integrator(t)` and `get_du` are all
collective, and the last two can call `f`. `integrator.u` holds the rank's own rows, and so
does the state given to `set_u!` or `reinit!`. `set_proposed_dt!` takes the smallest step any
rank proposes.

A callback's condition should give the same value on every rank, which usually means it
reduces over the ranks itself, for instance with `MPI.Allreduce`. The package reduces the
conditions as well, so ranks whose conditions disagree still stay together: a
`DiscreteCallback` fires when its condition is `true` on any rank, and a `ContinuousCallback`
or `VectorContinuousCallback` fires at the earliest event any rank finds, with that rank's
crossing. The affect then runs on every rank whatever its own condition gave, so it has to be
collective as well: an affect that calls `terminate!` has to call it on every rank. A condition,
affect, `initialize` or `finalize` that throws on some ranks makes every rank throw, as `f`
does, so an affect that throws has to do so after its own communication.

The implicit algorithms build their Jacobian as a distributed PETSc matrix whose pattern
comes from the problem's `jac_prototype`, which then holds this rank's rows only: it is
`length(u0)` by the length of the whole state, with global column indices. A `jac` fills
those rows, and is collective like `f`. For the heat equation above:

```julia
using SparseArrays

N = 8nranks
rows = 8rank .+ (1:8)
near(i) = max(1, i - 1):min(N, i + 1)
proto = sparse(
    [k for (k, i) in enumerate(rows) for _ in near(i)], [j for i in rows for j in near(i)],
    1.0, 8, N,
)
function heat_jac!(J, u, p, t)
    for (k, i) in enumerate(rows), j in near(i)
        J[k, j] = (i == j ? -2 : 1) / dx^2
    end
end

f = ODEFunction(heat!; jac = heat_jac!, jac_prototype = proto)
sol = solve(ODEProblem(f, sinpi.(x), (0.0, 0.1)), TSImplicit("bdf"; comm))
```

Without a `jac`, `autodiff` defaults to `AutoFiniteDiff()` on such a `comm`: PETSc colours the
prototype's pattern and differences `f`, calling it the same number of times on every rank.
ForwardDiff and the other `autodiff` backends are refused there, since the number of times
they call `f` differs between ranks. So are a `jac` or colouring without a sparse prototype,
a mass matrix other than a `Diagonal` of this rank's entries, and `TSIRK` without a `jac`.
`TSIRK` also needs each rank to hold PETSc's own share of the state, split evenly with the
first ranks taking one row more, since PETSc lays out its stage vector that way.

PETSc solves the linear systems of a distributed solve with GMRES and block Jacobi, one
ILU(0) block on each rank, to a relative tolerance of 1e-5, so such a solve agrees with a
serial one to that accuracy rather than to round-off. Options such as `-ksp_rtol` or
`-sub_pc_type` in `petsc_options` change that solver.

A distributed solve refuses, with an `ArgumentError`, `TSMPRK`, an implicit `TSGeneric` and
`PETScAdjoint`. Solving from several threads at once, as `EnsembleThreads` does, is not
refused, but nothing then keeps the ranks' solves in the same order, which they need.

## Limitations

Only the algorithms named under MPI run distributed so far; every other solve runs on
`MPI.COMM_SELF`. PETSc TS is built for large distributed problems, and reaching it
from the SciML interface is what this package is for; use OrdinaryDiffEq.jl for serial
problems where it applies.

On 32-bit Julia, use Julia 1.10, or add `PETSc_jll = "~3.22"` to your own compat: PETSc_jll
3.25 has no 32-bit builds, and newer Julia versions would otherwise resolve it.

Solves from several threads, such as an `EnsembleThreads` ensemble, are safe but run one
at a time: PETSc's options and MPI are shared by the whole process.

Finish or terminate every integrator you start. One dropped part way is released by a
finalizer, and if that finalizer runs at process exit, after MPI has shut down, PETSc's
own object finalizers print an MPI warning and the process exits non-zero.

## License

MIT. See [LICENSE](LICENSE).
