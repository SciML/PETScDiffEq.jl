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
    gl, gr = zeros(eltype(u), 1), zeros(eltype(u), 1)
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
those rows, and is collective like `f`. A `SplitODEProblem` under an algorithm other than
`TSARKIMEX`, which solves the sum of its parts, needs such a prototype on both parts, since
the pattern is their union, and a rank runs both parts even when the first throws. For the
heat equation above:

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
`autodiff = AutoForwardDiff()`, alone or in an `AutoSparse`, has ForwardDiff differentiate `f`
instead. At the start rank 0 gathers every rank's rows of the prototype, colours the columns of
the whole pattern and sends the colours back, and each rank seeds its own block of the state
with them. Every rank then calls `f` once per chunk of colours, and the dual numbers `f` sends
in its halo exchange carry the other ranks' share of each derivative, so `f`'s buffers have to
take `u`'s element type, as `zeros(eltype(u), 1)` above does. With `zeros(1)` the rank that
receives a dual number throws in the middle of the exchange and the ranks hang, which is one
reason it is not the default. On 1 to 3 ranks, including one holding no rows, the Jacobians of
the heat equation and of a Brusselator matched those of a serial ForwardDiff solve and of a
hand-written `jac` bit for bit, and the solves matched the `jac`'s exactly. Nor is it reliably
cheaper on a stiff problem: on twelve Robertson cells coupled by diffusion, with `"bdf"` at
`abstol = 1e-10` and `reltol = 1e-6`, it took at most the colouring's Newton iterations to
t = 1e5 with a third of the calls to `f`, but to t = 1e11 it took 31% fewer on one rank and 47%
more on three. Other backends are refused there, since `f` would have to carry their derivatives
through its own communication. So are a `jac`, PETSc's colouring or ForwardDiff without a sparse
prototype, a dense mass matrix, and `TSIRK` without a `jac` or `AutoForwardDiff()`.

A mass matrix is a `Diagonal` of this rank's entries, or a sparse matrix holding this rank's
rows with global column indices, as the prototype does, such as a finite element mass
matrix. PETSc assembles a sparse one into a distributed matrix, and its pattern joins the
prototype's in the Jacobian `a*M - J` of the implicit solve. A rank whose rows of the mass
matrix are the identity can leave it as `I`, whatever the other ranks give.
`TSIRK` takes any split of the state. PETSc lays out its stage vector by its own even split,
the first ranks taking one row more, so on any other split the solve runs on that one and
moves the state to and from each rank's block around every call to `f` and `jac`, which, like
`sol.u` and the callbacks, still see the rank's own block. On 2 and 3 ranks, one of them
holding no rows in some splits, a heat equation and a stiff reaction-diffusion problem gave the
states of the solve on PETSc's own split bit for bit from the same initial values, and the
serial solve's to 3e-15 with `-ksp_rtol 1e-14`. On up to 24,000 rows the moves, 12 to 37 a
step, took 7% to 23% of the solve's time.

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

A `DynamicalODEProblem` or `SecondOrderODEProblem` runs distributed as well, with
`TSBasicSymplectic`, `TSAlpha2` or any algorithm above on its first-order form. Each rank's
`ArrayPartition(v, u)` holds its block of the velocity and its block of the position, each
following the other ranks' blocks of the same part in rank order, and the states it gets back,
in `sol.u`, `integrator.u`, callbacks and `unstable_check`, are such blocks too; `save_idxs` and
vector tolerances index this rank's `[v; u]`. `f1` and `f2` see only those blocks and are
collective like `f`, and a rank runs both even when the first throws. A `jac_prototype` holds
this rank's rows of the first-order system's Jacobian, its `v` rows and then its `u` rows,
with the columns a serial solve gives them, the whole velocity before the whole position, so
it is `length(u0)` by the length of the whole state, and `jac` fills it as it would those rows
of the serial Jacobian. `TSAlpha2` needs each rank's `v` and `u` to have the same
length, and builds the distributed matrix it factors, `shift_a I - shift_v df/dv - df/du`,
from the `v` rows; without a `jac` PETSc colours that matrix and differences `f`, as for the
other implicit algorithms. `autodiff = AutoForwardDiff()` builds the first-order system's
Jacobian from the prototype instead, for `TSAlpha2` and for the implicit algorithms on the
first-order form, as it does for an `ODEProblem`: the columns are coloured in the prototype's
order, and each rank seeds the entries of its own `[v; u]` that those columns stand for, so
`f1` and `f2` have to exchange dual numbers as `f` does there. It needs the sparse prototype,
and other backends are refused. A `dm` is refused for these problems. Split over 1 to 3 ranks,
unevenly on more than one, a 1-D wave equation and a
chain of particles gave the serial solve's states to 3e-15 with `TSBasicSymplectic` and fixed
steps of `TSRK`, the energy error included, and to 1.2e-9 with adaptive steps of `TSRK("5dp")`.
`TSAlpha2` and `TSImplicit("bdf")` with a `jac` agreed to 2e-10 with `["-ksp_type", "preonly",
"-pc_type", "redundant"]` in `petsc_options` and to 8e-8 with the default linear solver, and
colouring, whose differences depend on the layout, moved `TSAlpha2` by up to 3e-6. On 1 and 2
ranks the ForwardDiff Jacobians of the wave equation and of the chain matched a `jac`'s entry by
entry to 1e-12, and the solves took the `jac` solve's steps, with states within 1e-13 of its
own on the wave equation under `TSAlpha2` and `TSImplicit("bdf")` and within 1e-10 on the chain
under `TSImplicit("bdf")`.

`PETScAdjoint` runs distributed too, for `TSRK`, `TSARKIMEX` and `TSImplicit`'s `"beuler"`,
`"cn"` and `"theta"` on an `ODEProblem`, and for `TSARKIMEX` on a `SplitODEProblem`. It takes
the problem's `jac`, filling this rank's rows of a sparse prototype as above, and when there
are parameters a `paramjac` filling this rank's rows; both are collective like `f`. An
explicit method takes such a `jac` in its own solve too, and ignores it there. Without a
`jac` it builds one with ForwardDiff from the sparse prototype, coloured as above, so `f`'s
buffers have to take `u`'s element type. An implicit method needs
`autodiff = AutoForwardDiff()` for that, since the default on such a `comm`,
`AutoFiniteDiff()`, leaves the adjoint no Jacobian to multiply by; `TSRK` has no `autodiff`
and needs only the prototype. Without a `paramjac` it seeds `p` instead: every rank calls
`f` once per chunk of parameters and keeps its own rows. `u` holds plain numbers in those
calls, so a halo exchange of `u` goes through as in the solve, and only what `f` sends that
depends on `p` has to travel in a buffer that takes dual numbers. Other backends are refused
for both. On 1 and 2 ranks the gradients built this way were within 1.5e-15 relative of
those from a hand-written `jac` and `paramjac` and of the serial adjoint's, for `TSRK`,
`TSImplicit` and `TSARKIMEX`. A `SplitODEProblem` takes all of this for each part: `f1`'s
`jac`, `paramjac` and prototype are the problem's, and `f2`'s `ODEFunction` carries its own
`jac`, `paramjac` and sparse `jac_prototype` of this rank's rows, none of which
`TSARKIMEX`'s own solve reads; the prototype is needed with or without the `jac`.
`dgdu_discrete` gets this rank's rows of the state and writes their gradient, and
`dgdp_discrete` gives this rank's share of the cost's direct derivative with respect to `p`,
which the ranks add up. `du0` comes back as this rank's rows and `dp` as the whole gradient,
the same on every rank. The cost times, `no_start`, the length of `p` and whether
`dgdp_discrete`, `jac` and `paramjac` are given have to agree across the ranks. A `jac`,
`paramjac`, cost function or `f` that throws on some ranks makes every rank throw, as in a
solve. The transposed linear solves of `TSImplicit` and `TSARKIMEX` use the solver above;
`["-ksp_type", "preonly", "-pc_type", "redundant"]` in `petsc_options` solves them directly.
On 1 and 2 ranks a reaction-diffusion problem split into its diffusion and its reaction gave
the gradient of the serial split adjoint to 7e-16 and that of central differences of the
same fixed-step solve to 1.2e-9. With `f2`'s `jac` and `paramjac` built the gradient was the
same to the last digit, and with `f1`'s built it was within 8.2e-15, which is where building
them puts it on `MPI.COMM_SELF` too.

A `DynamicalODEProblem` or `SecondOrderODEProblem` is differentiated there too, with the same
methods on its first-order form. Its `jac` fills the prototype described above, this rank's
`v` rows and then its `u` rows with the columns of the whole `[v; u]`, and `paramjac` fills
the same rows. A missing one is built with ForwardDiff as for an `ODEProblem`, the `jac` from
that prototype with each rank seeding its own `[v; u]`. The cost functions get this rank's
`ArrayPartition(v, u)` and write their derivative into one, and `du0` comes back as one. On 2
ranks a damped nonlinear wave gave the serial adjoint's gradient to 3.1e-14 with each of the
six methods, and on one rank that of central differences of the same fixed-step solve to
2e-10. With `jac` and `paramjac` built the gradient stayed within 3.1e-15 of the one from
hand-written ones on the same ranks.

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
fills it by colouring it and differencing `f`, so `autodiff` defaults to `AutoFiniteDiff()`;
the stencil has to cover every point `f` reads. Colouring calls `f` once per colour at every
Jacobian, and the colours grow with the stencil width and the degrees of freedom at a point.
On a DMDA `autodiff = AutoForwardDiff()` builds the exact Jacobian instead: `f` runs on a
ghosted array of dual numbers, each entry seeded by the colour of the grid point it belongs
to, ghosts included, so one call fills up to 12 colours and a `chunksize` sets another count.
The colours are the DM's own, or those of the matrix's pattern where PETSc has none or a wrong
one for a periodic grid. On one rank, for grids in 1-D and 2-D with one and two degrees of
freedom, star and box stencils and periodic and ghosted edges, the matrix matched a `jac`'s to
7e-15 where PETSc's colouring was up to 4e-6 off, and the solves took the `jac`'s steps. `f`
has to accept dual numbers, and a DMStag, a DMPlex, a `DAEProblem`, a complex state and the
other backends are refused. A `jac(J, u, p, t)` can fill the matrix as well. It gets
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
it in its own solve, for `PETScAdjoint`. `autodiff` is ignored with a `jac`. The two `jac`s of
a `SplitODEProblem` cannot be added in the DM's matrix, so an implicit algorithm other than
`TSARKIMEX`, which solves the sum of the parts, refuses a `jac` on both; with one on a single
part or none, PETSc colours the matrix. The heat
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

A DMStag, PETSc's staggered grid, keeps values on the vertices, edges, faces and cells of its
elements, and runs through the same calls. On one, `PETScDiffEq.reshape_local_array(x, dm)`
indexes `x[loc, c, i]` on a 1-D grid, `x[loc, c, i, j]` on a 2-D one and `x[loc, c, i, j, k]`
on a 3-D one: component `c` at location `loc` of element `(i, j, k)`, where `loc` is a
`LibPETSc.DMStagStencilLocation` such as `DMSTAG_ELEMENT`, `DMSTAG_LEFT` or `DMSTAG_DOWN`, and
`c` and the elements count from 1, where PETSc counts from 0. A point on the upper side of an
element, `DMSTAG_RIGHT` say, is the lower one of the next element. `axes(x, d)` gives the
elements along axis `d` whose points `x` holds: the ghosted array takes in the ghost elements,
and on the last rank along an axis that is not periodic the owned block runs to element
`N + 1`, which holds only the points on that boundary, so `f` fills the cells up to `N` and the
vertices up to `N + 1`. Any other point throws an `ArgumentError`. A `jac` writes through
`set_stencil_values!` with points `(loc, c, i)`, `(loc, c, i, j)` or `(loc, c, i, j, k)`, which
go to `DMStagMatSetValuesStencil`. Without one PETSc colours the DM's matrix as for a DMDA, so
its stencil has to cover what `f` reads: a `DMSTAG_STENCIL_NONE` grid couples only the
components at each point, `DMSTAG_STENCIL_STAR` reaches the neighbours along each axis within
the stencil width and `DMSTAG_STENCIL_BOX`, PETSc.jl's default, the diagonal ones too. A wave
with fluxes on the vertices, zero at both ends, and pressures in the cells:

```julia
stag = PETSc.DMStag(petsclib, MPI.COMM_WORLD, (LibPETSc.DM_BOUNDARY_GHOSTED,), (N,), (1, 1), 1)
LEFT, RIGHT, CELL = LibPETSc.DMSTAG_LEFT, LibPETSc.DMSTAG_RIGHT, LibPETSc.DMSTAG_ELEMENT

function wave!(du, u, stag, t)
    U = PETScDiffEq.reshape_local_array(u, stag)
    D = PETScDiffEq.reshape_local_array(du, stag)
    for i in axes(D, 1)
        D[LEFT, 1, i] = i == 1 || i == N + 1 ? 0.0 : (U[CELL, 1, i - 1] - U[CELL, 1, i]) * N
        i <= N && (D[CELL, 1, i] = (U[LEFT, 1, i] - U[RIGHT, 1, i]) * N)
    end
end

function wave_jac!(J, u, stag, t)
    for i in owned
        1 < i <= N && set_stencil_values!(J, (LEFT, 1, i), ((CELL, 1, i - 1), (CELL, 1, i)), (N, -N))
        i <= N && set_stencil_values!(J, (CELL, 1, i), ((LEFT, 1, i), (RIGHT, 1, i)), (N, -N))
    end
end

u0 = zeros(LibPETSc.DMStagGetEntries(petsclib, stag))
U0 = PETScDiffEq.reshape_local_array(u0, stag)
owned = axes(U0, 1)
for i in owned
    i <= N && (U0[CELL, 1, i] = sinpi((i - 0.5) / N))
end
fn = ODEFunction(wave!; jac = wave_jac!)
sol_stag = solve(ODEProblem(fn, u0, (0.0, 0.5), stag), TSImplicit("bdf"; dm = stag))
```

On `MPI.COMM_SELF` and on 1 to 3 ranks, a damped form of this wave and a 2-D one with fluxes on
the faces matched the same equations written without a DM: bit for bit with `TSRK`, and to
1.2e-14 with a `jac` and a direct linear solve, `["-ksp_type", "preonly", "-pc_type",
"redundant"]`, in `TSImplicit` and `TSRosW`, which colouring moved by up to 1.4e-14. The wave
also matched through the integrator to 1.1e-19 and with a mass matrix to 9.4e-15, on 1 to 3
ranks as a `SplitODEProblem` in `TSARKIMEX` and a `DAEProblem` in `TSDAE` to 5.7e-15 and in its
DAE initialization to 1.1e-19, and `PETScAdjoint` with a `jac` and a `paramjac` matched the
serial adjoint to 2.3e-15 relative.

A DMPlex, PETSc's unstructured mesh, runs through the same calls once it has a local section, a
`PetscSection` that gives each mesh point its degrees of freedom: on the vertices, on the
cells, or on any mix of the mesh's strata. The section has to be set up, point-major, without a
permutation and without constrained degrees of freedom. `DMPlexCreateBoxMesh` leaves a mesh
with an empty one, and a mesh without degrees of freedom, a section that is not set up and the
other layouts are refused with an `ArgumentError`. `PETScDiffEq.reshape_local_array(x, dm)`
indexes `x[c, p]`, component `c` of mesh point `p`, where `p` is PETSc's own point number,
counted from 0 as `DMPlexGetDepthStratum`, `DMPlexGetHeightStratum`, `DMPlexGetCone` and
`DMPlexGetSupport` give it, and `c` counts from 1. The ghosted `u` holds every point of this
rank's part of the mesh, and `du` only the points the rank owns, which
`checkbounds(Bool, D, c, p)` picks out. A point's neighbours are on its rank when the mesh is
distributed with an overlap of one cell. The mesh does not change during a solve, so its
connectivity is best gathered once, outside `f`. A `jac` writes through `set_stencil_values!`
with points `(c, p)`, which go to `MatSetValues` at the indices of the DM's global section. The
DM's matrix has the pattern of the DM's adjacency, which by default couples each point to the
closure of its star: enough for vertices coupled along edges, while cells coupled across their
faces need `DMSetBasicAdjacency(dm, true, false)`. PETSc refuses a `jac` entry outside the
pattern, and without a `jac` it colours the pattern. Finite-element assembly through PetscFE
and `DMPlexSNESComputeResidualFEM` is not used. Reaction-diffusion on the vertices of a
triangulated box, which `PETSc.DMPlex` builds and distributes through `DMSetFromOptions`:

```julia
dm = PETSc.DMPlex(
    petsclib, MPI.COMM_WORLD; dm_plex_dim = 2, dm_plex_simplex = true,
    dm_plex_box_faces = "16,16", dm_distribute_overlap = 1,
)
vstart, vend = LibPETSc.DMPlexGetDepthStratum(petsclib, dm, 0)
section = LibPETSc.PetscSectionCreate(petsclib, MPI.COMM_WORLD)
LibPETSc.PetscSectionSetChart(petsclib, section, LibPETSc.DMPlexGetChart(petsclib, dm)...)
for v in vstart:(vend - 1)
    LibPETSc.PetscSectionSetDof(petsclib, section, v, 1)
end
LibPETSc.PetscSectionSetUp(petsclib, section)
LibPETSc.DMSetLocalSection(petsclib, dm, section)
LibPETSc.PetscSectionDestroy(petsclib, Ref(section))

u0 = zeros(LibPETSc.VecGetLocalSize(petsclib, LibPETSc.DMCreateGlobalVector(petsclib, dm)))
U0 = PETScDiffEq.reshape_local_array(u0, dm)
owned = [v for v in vstart:(vend - 1) if checkbounds(Bool, U0, 1, v)]
along_edges(v) = [
    only(filter(!=(v), LibPETSc.DMPlexGetCone(petsclib, dm, e))) for
        e in LibPETSc.DMPlexGetSupport(petsclib, dm, v)
]
neighbours = Dict(v => along_edges(v) for v in owned)
for v in owned
    U0[1, v] = sinpi(v / 10)
end

function rd!(du, u, dm, t)
    U, D = PETScDiffEq.reshape_local_array(u, dm), PETScDiffEq.reshape_local_array(du, dm)
    for v in owned
        D[1, v] = 256 * sum(U[1, w] - U[1, v] for w in neighbours[v]) - U[1, v]^3
    end
end

function rd_jac!(J, u, dm, t)
    U = PETScDiffEq.reshape_local_array(u, dm)
    for v in owned
        ws = neighbours[v]
        set_stencil_values!(
            J, (1, v), [(1, v); [(1, w) for w in ws]],
            [-256 * length(ws) - 3U[1, v]^2; fill(256.0, length(ws))],
        )
    end
end

fn = ODEFunction(rd!; jac = rd_jac!)
sol_plex = solve(ODEProblem(fn, u0, (0.0, 0.1), dm), TSImplicit("bdf"; dm))
```

PETSc.jl 0.4's `PetscSectionCreate` fills a `Ref` it is given, `PetscSectionCreate(petsclib,
MPI.COMM_WORLD, section)` with `section = Ref{LibPETSc.PetscSection}()`, instead of returning
the section. On `MPI.COMM_SELF` and on 1 to 3 ranks, reaction-diffusion on the vertices of a
triangulated box and in the cells of a box of squares, coupled across their edges, matched the
same equations assembled without a DM from the same mesh: bit for bit with `TSRK`, to 1.2e-14
with a `jac` and a direct linear solve in `TSImplicit` and `TSRosW`, which colouring moved by up
to 9.8e-14, and exactly through the integrator. On `MPI.COMM_SELF` the vertex problem also
matched with a mass matrix exactly, as a `SplitODEProblem` in `TSARKIMEX` to 5.4e-15 and as a
`DAEProblem` in `TSDAE` to 1.1e-14, and `PETScAdjoint` with a `jac` and a `paramjac` matched the
adjoint without a DM exactly.

The rest works as it does without a DM: `TSRK`, `TSRosW`, `TSImplicit`, `TSDAE`,
`TSARKIMEX` and `TSGeneric`, explicit or implicit, `saveat`, dense output, callbacks and
the integrator interface, a `Diagonal` mass matrix, a `SplitODEProblem`, whose `f2` gets `u`
ghosted as `f` does, and a `DAEProblem`, whose residual `f(r, du, u, p, t)` gets `u` ghosted
and `du` owned. Everything else the package calls, such as a callback, `unstable_check` or
`isoutofdomain`, sees the owned block. The TS works on a copy of the DM from `DMClone`, so the
DM itself stays free for further solves. A DM on `MPI.COMM_SELF`, or on a single rank, gives a
serial solve. Only a DMDA, a DMStag or a DMPlex is taken so far, and any other DM, such as a
DMShell, is refused with an `ArgumentError`.

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

A solve with a `dm` refuses `TSIRK`, `TSMPRK` and an implicit `TSGeneric` of type `"irk"`,
or of a type refused on a `comm`, with an `ArgumentError`. A distributed solve, with a `dm`
or without, is refused off the root task
when Julia has more than one thread, whether inside `Threads.@threads` (as `EnsembleThreads`
runs its trajectories) or from a `Threads.@spawn` task: nothing there keeps the ranks' solves
in the same order, and ranks taking them in different orders run different solves as one and
can return wrong results without an error. An ensemble of distributed solves runs with
`EnsembleSerial()`.
