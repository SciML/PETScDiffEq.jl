# MPI

Every algorithm takes a `comm` keyword. With a communicator
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
counts on all of them. A step that turns NaN or overflows on any rank's rows is taken again
smaller on every rank, and a fixed-step solve stops at the first state that is not finite on
some rank, as in a serial solve. When `f`, `jac` or one of those checks throws on some ranks,
those ranks go on with NaN until the ranks next agree, at the end of the step or when its
nonlinear solve fails, and then every rank throws rather than retrying the step, so an `f`
that throws has to do so after its own communication.

Callbacks and the integrator interface run distributed too, as long as every rank makes the
same calls with the same arguments in the same order: `init`, `step!`, `solve!`, `reinit!`,
`terminate!`, `set_u!`, `initialize_dae!`, `add_tstop!`, `add_saveat!`, `savevalues!`,
`change_t_via_interpolation!`, `set_proposed_dt!`, `set_abstol!`, `set_reltol!`,
`postamble!`, `auto_dt_reset!`, `integrator(t)` and `get_du` are all collective.
`integrator.u` holds the rank's own rows, and so
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
a dense mass matrix, and `TSIRK` without a `jac`.

A mass matrix is a `Diagonal` of this rank's entries, or a sparse matrix holding this rank's
rows with global column indices, as the prototype does, such as a finite element mass
matrix. PETSc assembles a sparse one into a distributed matrix, and its pattern joins the
prototype's in the Jacobian `a*M - J` of the implicit solve. A rank whose rows of the mass
matrix are the identity can leave it as `I`, whatever the other ranks give.
`TSIRK` also needs each rank to hold PETSc's own share of the state, split evenly with the
first ranks taking one row more, since PETSc lays out its stage vector that way.

PETSc solves the linear systems of a distributed solve with GMRES and block Jacobi, one
ILU(0) block on each rank, to a relative tolerance of 1e-5, so such a solve agrees with a
serial one to that accuracy rather than to round-off. Options such as `-ksp_rtol` or
`-sub_pc_type` in `petsc_options` change that solver.

`TSMPRK`'s `slow` and `medium` index the rank's own rows, and either may be empty on some
ranks as long as some rank names a slow row and, for `"2a23"` and `"2a33"`, a medium one. An
implicit `TSGeneric` runs distributed for `"beuler"`, `"cn"`, `"theta"`, `"bdf"`, `"rosw"`,
`"arkimex"`, `"irk"`, `"alpha"` and `"dirk"`, as `TSImplicit` does. Other implicit types are
refused: `"glle"`'s step control follows the round-off of the distributed linear solve, so it
takes other steps than a serial solve and ends with another error, larger or smaller.

`PETScAdjoint` runs distributed too, for `TSRK`, `TSARKIMEX` on an `ODEProblem` and
`TSImplicit`'s `"beuler"`, `"cn"` and `"theta"`. It needs the problem's `jac`, filling this
rank's rows of a sparse prototype as above, and when there are parameters a `paramjac`
filling this rank's rows, since automatic differentiation would call `f` a different number
of times on each rank; both are collective like `f`. An explicit method takes such a `jac`
in its own solve too, and ignores it there. `dgdu_discrete` gets this rank's rows of the
state and writes their gradient, and `dgdp_discrete` gives this rank's share of the cost's
direct derivative with respect to `p`, which the ranks add up. `du0` comes back as this
rank's rows and `dp` as the whole gradient, the same on every rank. The cost times,
`no_start`, the length of `p` and whether `dgdp_discrete` is given have to agree across the
ranks. A `jac`, `paramjac`, cost function or `f` that throws on some ranks makes every rank
throw, as in a solve. The transposed linear solves of `TSImplicit` and `TSARKIMEX` use the
solver above; `["-ksp_type", "preonly", "-pc_type", "redundant"]` in `petsc_options` solves
them directly.

A PETSc DM can do the halo exchange instead. Build a DMDA with PETSc.jl and pass it as `dm`,
which every algorithm that takes `comm` takes as well. The solve then runs on the DM's
communicator, which `comm` may name too but not contradict, and `u0` is the block of the grid
this rank owns, in the DM's order. `f(du, u, p, t)` gets `u` ghosted: before every call to `f`,
including the package's own calls for the first step size, dense output, callbacks, `get_du`
and the Jacobian, the state is scattered into a local vector from `DMGetLocalVector` with
`DMGlobalToLocalBegin` and `DMGlobalToLocalEnd`, so `u` also holds the neighbouring ranks'
points within the stencil width. `du` is the owned block. `PETScDiffEq.reshape_local_array(x, dm)`
views either one by grid point in global numbering, as `x[c, i]` on a 1-D grid and `x[c, i, j]`
on a 2-D one, where `c` is the degree of freedom at the point. It is PETSc.jl's
`reshape_local_array`, which PETSc.jl 0.4 calls `reshapelocalarray`. With `DM_BOUNDARY_GHOSTED`
the ghost points past the edge of the grid read zero, so this heat equation is zero at both
ends:

```julia
using MPI, PETScDiffEq, SciMLBase
using PETScDiffEq: PETSc, LibPETSc

MPI.Init()
petsclib = PETSc.getlib(; PetscScalar = Float64)
PETSc.initialize(petsclib)
N = 64
dx = 1 / (N + 1)
da = PETSc.DMDA(petsclib, MPI.COMM_WORLD, (LibPETSc.DM_BOUNDARY_GHOSTED,), (N,), 1, 1)

function heat!(du, u, da, t)
    U = PETScDiffEq.reshape_local_array(u, da)
    D = PETScDiffEq.reshape_local_array(du, da)
    for i in axes(D, 2)
        D[1, i] = (U[1, i - 1] - 2U[1, i] + U[1, i + 1]) / dx^2
    end
end

xs, _, _, xm = LibPETSc.DMDAGetCorners(petsclib, da)
prob = ODEProblem(heat!, sinpi.((xs .+ (1:xm)) .* dx), (0.0, 0.1), da)
sol = solve(prob, TSRK("5dp"; dm = da))
sol_bdf = solve(prob, TSImplicit("bdf"; dm = da))
```

With a `dm` the implicit algorithms need no `jac_prototype`. Their Jacobian is the DM's own
matrix from `DMCreateMatrix`, whose pattern comes from the DM's stencil. Without a `jac` PETSc
fills it by colouring it and differencing `f`, so `autodiff` defaults to `AutoFiniteDiff()`
and the other backends are refused; the stencil has to cover every point `f` reads. Colouring
calls `f` once per colour at every Jacobian, and the colours grow with the stencil width and
the degrees of freedom at a point, so a `jac(J, u, p, t)` can fill the matrix instead. It gets
`u` ghosted, as `f` does, and `J` is the DM's matrix as a PETSc.jl `Mat`, zeroed before the
call and assembled after it, so `jac` writes ``df/du`` into it through PETSc's matrix API,
each rank its own rows: `set_stencil_values!(J, rows, cols, vals)` writes a block by grid
index through `MatSetValuesStencil`, in the numbering `reshape_local_array` uses, and
`J[i, j] = v` writes one entry by global index through `MatSetValues`, for those who have the
global numbering; `LibPETSc` has the rest. A column at a ghost point past the edge of a
`DM_BOUNDARY_GHOSTED` grid is dropped, having no global entry, and a set and an add cannot
follow each other without a `PETSc.assemble!(J)` between them. The package turns the matrix
into PETSc's `shift * M - J` itself. A `DAEProblem`'s `jac(J, du, u, p, gamma, t)` gets `u`
ghosted and `du` owned, and writes PETSc's whole `dG/du + gamma dG/du'`. `jac` runs on every
rank at every Jacobian, so anything collective in it has to be called on all of them in the
same order, and it has to be in place: an out-of-place one is refused, as is a
`jac_prototype`, since the DM gives the pattern. An explicit method takes a `jac` and ignores
it in its own solve, for `PETScAdjoint`. `autodiff` is ignored with a `jac`. The heat
equation above with its Jacobian, whose columns past the ends of the grid are dropped:

```julia
function heat_jac!(J, u, da, t)
    for i in (xs + 1):(xs + xm)
        set_stencil_values!(J, (1, i), ((1, i - 1), (1, i), (1, i + 1)), (1, -2, 1) ./ dx^2)
    end
end

fn = ODEFunction(heat!; jac = heat_jac!)
sol_jac = solve(ODEProblem(fn, sinpi.((xs .+ (1:xm)) .* dx), (0.0, 0.1), da), TSImplicit("bdf"; dm = da))
```

The rest works as it does without a DM: `TSRK`, `TSRosW`, `TSImplicit`, `TSDAE`,
`TSARKIMEX` and `TSGeneric(ts_type; explicit = true)`, `saveat`, dense output, callbacks and
the integrator interface, a `Diagonal` mass matrix, a `SplitODEProblem`, whose `f2` gets `u`
ghosted as `f` does, and a `DAEProblem`, whose residual `f(r, du, u, p, t)` gets `u` ghosted
and `du` owned. Everything else the package calls, such as a callback, `unstable_check` or
`isoutofdomain`, sees the owned block. The TS works on a copy of the DM from `DMClone`, so the
DM itself stays free for further solves. A DMDA on `MPI.COMM_SELF`, or on a single rank, gives
a serial solve. Only a DMDA is taken so far.

`PETScAdjoint` runs with a `dm` too, for the methods and discrete costs it takes on a `comm`,
and on a DM of a single rank for integral costs and a `SplitODEProblem` with `TSARKIMEX` as
well. It needs the problem's `jac`, filling the DM's matrix as above, and when there are
parameters a `paramjac(pJ, u, p, t)` that gets `u` ghosted and fills this rank's rows of `pJ`
in the DM's order, a column per entry of `p`, which
`PETScDiffEq.reshape_local_array(view(pJ, :, k), dm)` indexes by grid point; a
`SplitODEProblem` needs `f2`'s as well. Neither is built when missing: automatic
differentiation would not see the ghosted array, and colouring only approximates the
Jacobian, which moved the gradient by up to 3.5e-9 in a reaction-diffusion test and which
PETSc's `TSARKIMEX` adjoint cannot use at all. The rest follows the distributed adjoint above:
the cost functions get this rank's block of the state as `solve` saves it, not the ghosted
array, `du0` comes back as that block and `dp` whole on every rank, and a function that
throws on some ranks makes every rank throw. On 1-D and 2-D DMDAs of 1 to 3 ranks the
gradient agreed with the comm-mode and serial adjoints of the same discretization to 2e-15
and with central differences of the same fixed-step solve to 1.4e-9.

A solve with a `dm` refuses `TSIRK`, `TSMPRK` and an implicit `TSGeneric` with an
`ArgumentError`. A distributed solve, with a `dm` or without, is refused inside
`Threads.@threads` on more than one thread, as `EnsembleThreads` runs its trajectories:
nothing there keeps the ranks' solves in the same order, and ranks taking them in different
orders run different solves as one and can return wrong results without an error.
Distributed solves running at once from `Threads.@spawn` tasks are not refused, so the caller
has to keep them in the same order on every rank. An ensemble of distributed solves runs with
`EnsembleSerial()`.
