using MPI, PETScDiffEq, SciMLBase, SparseArrays, Test
using LinearAlgebra: Diagonal
using PETScDiffEq: PETSc, LibPETSc, PETScCompat, AutoForwardDiff, AutoFiniteDiff, DiffEqBase,
    reshape_local_array
using SciMLBase: ODEProblem, ODEFunction, DAEProblem, DAEFunction, SplitODEProblem,
    DiscreteCallback, ContinuousCallback, ReturnCode, init, solve, solve!, step!, get_du,
    terminate!

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
# PetscInitialize is collective, and the serial reference solves run on single ranks.
for S in (Float64, Float32)
    PETSc.initialize(PETSc.getlib(; PetscScalar = S))
end
const pl = PETSc.getlib(; PetscScalar = Float64)
const GHOSTED = LibPETSc.DM_BOUNDARY_GHOSTED
const thrower = nranks - 1
const TOL = (abstol = 1.0e-8, reltol = 1.0e-8)
const FIXED = (; dt = 1.0e-3, adaptive = false)
const ROUNDOFF = 5.0e-14
# Measured at 1 to 3 ranks: a dm jac solve is within 7.1e-14 of a serial bdf jac solve, 4.2e-9
# of a serial rosw one, whose parallel linear solves no Newton cleans up, and 2.0e-14 of colouring.
const JAC_SERIAL_TOL = (bdf = 5.0e-13, rosw = 2.0e-8)
const COLOUR_TOL = 1.0e-13
const INIT_GAP = 1.0e-14
# Measured at 1 and 2 ranks: a start from a `jac` is within 1.5e-14 of the coloured one.
const INIT_JAC_GAP = 1.0e-12
# Measured at 1 to 3 ranks with direct linear solves: DMStag solves are within 8.0e-15 of the
# serial ones and 1.1e-14 of colouring, steps and DAE initialization within 1.1e-19, and the
# adjoint within 1.8e-15 relative.
const STAG_SERIAL_TOL = 1.0e-13
const STAG_STEP_TOL = 1.0e-15
const STAG_ADJOINT_GAP = 1.0e-14
# Measured at 1 to 3 ranks: DMPlex solves with a jac are within 6.0e-15 of the serial ones,
# colouring within 9.8e-14 of them, and steps match exactly.
const PLEX_SERIAL_TOL = 1.0e-13
const PLEX_COLOUR_TOL = 5.0e-13
const PLEX_STEP_TOL = 1.0e-15

function uneven(n)
    counts = floor.(Int, n .* (1:nranks) ./ sum(1:nranks))
    counts[end] += n - sum(counts)
    return counts
end
owned(counts) = (lo = sum(counts[1:rank]) + 1; lo:(lo + counts[rank + 1] - 1))

same_everywhere(x) = MPI.bcast(x, 0, comm) == x
anywhere(b) = MPI.Allreduce(b, |, comm)
everywhere(b) = MPI.Allreduce(b, &, comm)
maxdiff(a, b) = maximum(maximum(abs, x - y) for (x, y) in zip(a, b))

function refs(obj)
    n = Ref{LibPETSc.PetscInt}(0)
    PETScDiffEq._check_code(
        ccall(
            PETScDiffEq._symbol(pl, :PetscObjectGetReference), LibPETSc.PetscErrorCode,
            (Ptr{Cvoid}, Ptr{LibPETSc.PetscInt}), obj, n,
        ),
    )
    return Int(n[])
end
held(dm) = PETScDiffEq._referenced(pl, dm)
function released(dm)
    n = refs(dm)
    PETScDiffEq._check_code(
        ccall(
            PETScDiffEq._symbol(pl, :DMDestroy), LibPETSc.PetscErrorCode, (Ptr{Ptr{Cvoid}},),
            Ref(dm),
        ),
    )
    return n == 1
end

function caught(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

remote(e) = e isa ErrorException && occursin("another rank", e.msg)
raised(e, what) = rank == thrower ? e isa ErrorException && occursin(what, e.msg) : remote(e)
refused(f, what) = (e = caught(f); e isa ArgumentError && occursin(what, e.msg))
refused_on_thrower(e, what) =
    rank == thrower ? e isa ArgumentError && occursin(what, e.msg) : remote(e)

function natural(u, dm, dims)
    full = zeros(dims)
    a = reshape_local_array(u, dm)
    for I in CartesianIndices(axes(a)[2:end])
        full[I] = a[1, I]
    end
    return MPI.Reduce(vec(full), +, comm; root = 0)
end
natural(sol::SciMLBase.AbstractODESolution, dm, dims) = [natural(u, dm, dims) for u in sol.u]

const N = 23
const dx = 1 / (N + 1)
const counts = uneven(N)
const rows = owned(counts)
const SPAN = (0.0, 0.1)
heat0(idx) = sinpi.(idx .* dx) .+ 0.5 .* sinpi.(3 .* idx .* dx)
eigval(k) = -4 / dx^2 * sinpi(k * dx / 2)^2
heat_exact(t) = exp(eigval(1) * t) .* sinpi.((1:N) .* dx) .+
    0.5 * exp(eigval(3) * t) .* sinpi.(3 .* (1:N) .* dx)

line_da(c; kw...) = PETSc.DMDA(pl, c, (GHOSTED,), (N,), 1, 1; kw...)
const da = line_da(comm; points_per_proc = (LibPETSc.PetscInt.(counts),))

function heat_dm!(du, u, da, t)
    U, D = reshape_local_array(u, da), reshape_local_array(du, da)
    for i in axes(D, 2)
        D[1, i] = (U[1, i - 1] - 2U[1, i] + U[1, i + 1]) / dx^2
    end
    return nothing
end

function laplacian!(du, u, left, right)
    n = length(u)
    for i in 1:n
        l = i == 1 ? left : u[i - 1]
        r = i == n ? right : u[i + 1]
        du[i] = (l - 2u[i] + r) / dx^2
    end
    return nothing
end

function halo(u)
    left = rank == 0 ? MPI.PROC_NULL : rank - 1
    right = rank == nranks - 1 ? MPI.PROC_NULL : rank + 1
    gl, gr = zeros(1), zeros(1)
    MPI.Sendrecv!(u[1:1], gr, comm; dest = left, source = right)
    MPI.Sendrecv!(u[end:end], gl, comm; dest = right, source = left)
    return gl[1], gr[1]
end

heat!(du, u, p, t) = laplacian!(du, u, halo(u)...)
heat_serial!(du, u, p, t) = laplacian!(du, u, 0.0, 0.0)

neighbours(i) = max(1, i - 1):min(N, i + 1)
function heat_proto(idx)
    I = [k for (k, i) in enumerate(idx) for _ in neighbours(i)]
    J = [j for i in idx for j in neighbours(i)]
    return sparse(I, J, ones(length(I)), length(idx), N)
end

dm_heat(f = heat_dm!; kw...) = ODEProblem(ODEFunction(f; kw...), heat0(rows), SPAN, da)
comm_heat(f = heat!; kw...) =
    ODEProblem(ODEFunction(f; jac_prototype = heat_proto(rows), kw...), heat0(rows), SPAN)
serial_heat(f = heat_serial!; kw...) =
    ODEProblem(ODEFunction(f; jac_prototype = heat_proto(1:N), kw...), heat0(1:N), SPAN)

explicit(; kw...) = TSRK("5dp"; kw...)
implicit(; kw...) = TSImplicit("bdf"; autodiff = AutoFiniteDiff(), kw...)

const NX, NY = 7, 6
const hx, hy = 1 / (NX + 1), 1 / (NY + 1)
grid0(i, j) = sinpi(i * hx) * sinpi(j * hy) + 0.5 * sinpi(3i * hx) * sinpi(2j * hy)
eig2(k, l) = -4 / hx^2 * sinpi(k * hx / 2)^2 - 4 / hy^2 * sinpi(l * hy / 2)^2
grid_exact(t) = vec(
    [
        exp(eig2(1, 1) * t) * sinpi(i * hx) * sinpi(j * hy) +
            0.5 * exp(eig2(3, 2) * t) * sinpi(3i * hx) * sinpi(2j * hy)
            for i in 1:NX, j in 1:NY
    ],
)

grid_da(processors, ppp) = PETSc.DMDA(
    pl, comm, (GHOSTED, GHOSTED), (NX, NY), 1, 1, LibPETSc.DMDA_STENCIL_STAR;
    processors, points_per_proc = ppp,
)

function grid_heat_dm!(du, u, da, t)
    U, D = reshape_local_array(u, da), reshape_local_array(du, da)
    for j in axes(D, 3), i in axes(D, 2)
        D[1, i, j] = (U[1, i - 1, j] - 2U[1, i, j] + U[1, i + 1, j]) / hx^2 +
            (U[1, i, j - 1] - 2U[1, i, j] + U[1, i, j + 1]) / hy^2
    end
    return nothing
end

function grid_u0(da)
    u = zeros(PETScDiffEq._dm_local_size(pl, da))
    a = reshape_local_array(u, da)
    for j in axes(a, 3), i in axes(a, 2)
        a[1, i, j] = grid0(i, j)
    end
    return u
end

function grid_laplacian!(du, u, below, above)
    ny = length(u) ÷ NX
    U, D = reshape(u, NX, ny), reshape(du, NX, ny)
    at(i, j) = !(1 <= i <= NX) ? 0.0 : j == 0 ? below[i] : j == ny + 1 ? above[i] : U[i, j]
    for j in 1:ny, i in 1:NX
        D[i, j] = (at(i - 1, j) - 2U[i, j] + at(i + 1, j)) / hx^2 +
            (at(i, j - 1) - 2U[i, j] + at(i, j + 1)) / hy^2
    end
    return nothing
end

function grid_halo(u)
    down = rank == 0 ? MPI.PROC_NULL : rank - 1
    up = rank == nranks - 1 ? MPI.PROC_NULL : rank + 1
    below, above = zeros(NX), zeros(NX)
    MPI.Sendrecv!(u[1:NX], above, comm; dest = down, source = up)
    MPI.Sendrecv!(u[(end - NX + 1):end], below, comm; dest = up, source = down)
    return below, above
end

grid_heat!(du, u, p, t) = grid_laplacian!(du, u, grid_halo(u)...)
grid_heat_serial!(du, u, p, t) = grid_laplacian!(du, u, zeros(NX), zeros(NX))

function grid_proto(ys)
    point(i, j) = i + (j - 1) * NX
    near(i, j) = [
        point(a, b) for (a, b) in ((i, j - 1), (i - 1, j), (i, j), (i + 1, j), (i, j + 1))
            if 1 <= a <= NX && 1 <= b <= NY
    ]
    local_rows = [point(i, j) - point(1, first(ys)) + 1 for j in ys for i in 1:NX]
    cols = [near(i, j) for j in ys for i in 1:NX]
    I = [r for (r, c) in zip(local_rows, cols) for _ in c]
    return sparse(I, reduce(vcat, cols), ones(length(I)), NX * length(ys), NX * NY)
end

function throwing(f, when)
    return function (du, u, p, t)
        f(du, u, p, t)
        rank == thrower && when(t) && error("f threw on rank $rank")
        return nothing
    end
end

# A damped wave on a DMStag: fluxes on the vertices, held at zero on the ends, pressures in the
# cells, with `p` scaling the damping of each. The serial form orders them vertex, cell,
# vertex, ..., as the ranks of the DM do in turn.
const LEFT, RIGHT, ELEM = LibPETSc.DMSTAG_LEFT, LibPETSc.DMSTAG_RIGHT, LibPETSc.DMSTAG_ELEMENT
const DOWN, UP = LibPETSc.DMSTAG_DOWN, LibPETSc.DMSTAG_UP
const NS = 17
const hs = 1 / NS
const stag = PETSc.DMStag(
    pl, comm, (GHOSTED,), (NS,), (1, 1), 1; points_per_proc = (LibPETSc.PetscInt.(uneven(NS)),),
)
stag_points(dm) = axes(reshape_local_array(zeros(PETScDiffEq._dm_local_size(pl, dm)), dm))

function wave_dm!(du, u, p, t)
    U, D = reshape_local_array(u, stag), reshape_local_array(du, stag)
    for i in axes(D, 1)
        D[LEFT, 1, i] = i == 1 || i == NS + 1 ? 0.0 :
            -(U[ELEM, 1, i] - U[ELEM, 1, i - 1]) / hs - p[1] * U[LEFT, 1, i]
        i <= NS &&
            (D[ELEM, 1, i] = -(U[RIGHT, 1, i] - U[LEFT, 1, i]) / hs - p[2] * U[ELEM, 1, i]^3)
    end
    return nothing
end

function wave!(dx, x, p, t)
    for v in 1:(NS + 1)
        dx[2v - 1] = v == 1 || v == NS + 1 ? 0.0 : -(x[2v] - x[2v - 2]) / hs - p[1] * x[2v - 1]
    end
    for i in 1:NS
        dx[2i] = -(x[2i + 1] - x[2i - 1]) / hs - p[2] * x[2i]^3
    end
    return nothing
end

function wave_jac_dm!(J, u, p, t)
    U = reshape_local_array(u, stag)
    for i in stag_points(stag)[1]
        1 < i <= NS && set_stencil_values!(
            J, (LEFT, 1, i), ((ELEM, 1, i - 1), (ELEM, 1, i), (LEFT, 1, i)),
            (1 / hs, -1 / hs, -p[1]),
        )
        i <= NS && set_stencil_values!(
            J, (ELEM, 1, i), ((LEFT, 1, i), (RIGHT, 1, i), (ELEM, 1, i)),
            (1 / hs, -1 / hs, -3p[2] * U[ELEM, 1, i]^2),
        )
    end
    return nothing
end

function wave_jac!(J, x, p, t)
    # The adjoint hands `jac` the prototype's values, so every stored entry is written.
    fill!(nonzeros(J), 0.0)
    for v in 2:NS
        J[2v - 1, 2v - 2], J[2v - 1, 2v], J[2v - 1, 2v - 1] = 1 / hs, -1 / hs, -p[1]
    end
    for i in 1:NS
        J[2i, 2i - 1], J[2i, 2i + 1], J[2i, 2i] = 1 / hs, -1 / hs, -3p[2] * x[2i]^2
    end
    return nothing
end

function wave_paramjac_dm!(pJ, u, p, t)
    U = reshape_local_array(u, stag)
    A, B = reshape_local_array(view(pJ, :, 1), stag), reshape_local_array(view(pJ, :, 2), stag)
    for i in axes(A, 1)
        A[LEFT, 1, i] = i == 1 || i == NS + 1 ? 0.0 : -U[LEFT, 1, i]
        B[LEFT, 1, i] = 0.0
        i <= NS || continue
        A[ELEM, 1, i] = 0.0
        B[ELEM, 1, i] = -U[ELEM, 1, i]^3
    end
    return nothing
end

function wave_paramjac!(pJ, x, p, t)
    fill!(pJ, 0.0)
    for v in 2:NS
        pJ[2v - 1, 1] = -x[2v - 1]
    end
    for i in 1:NS
        pJ[2i, 2] = -x[2i]^3
    end
    return nothing
end

const NW = 2NS + 1
band(r) = max(1, r - 2):min(NW, r + 2)
const wave_proto = sparse(
    [r for r in 1:NW for _ in band(r)], [c for r in 1:NW for c in band(r)],
    ones(sum(length ∘ band, 1:NW)), NW, NW,
)
pressure(i) = sinpi((i - 0.5) * hs) + 0.3 * sinpi(3 * (i - 0.5) * hs)
const wave0 = [isodd(k) ? 0.0 : pressure(k ÷ 2) for k in 1:NW]

function on_wave(x, dm)
    u = zeros(PETScDiffEq._dm_local_size(pl, dm))
    a = reshape_local_array(u, dm)
    for i in axes(a, 1)
        a[LEFT, 1, i] = x[2i - 1]
        i <= NS && (a[ELEM, 1, i] = x[2i])
    end
    return u
end

function wave_natural(u, dm)
    x = zeros(NW)
    a = reshape_local_array(u, dm)
    for i in axes(a, 1)
        x[2i - 1] = a[LEFT, 1, i]
        i <= NS && (x[2i] = a[ELEM, 1, i])
    end
    return MPI.Reduce(x, +, comm; root = 0)
end

# A 2-D form of it: x-fluxes on the left faces, y-fluxes on the bottom ones, pressures in the
# cells, numbered cells first in the serial form.
const QX, QY = 6, 5
const qx, qy = 1 / QX, 1 / QY
cell(i, j) = (j - 1) * QX + i
xface(i, j) = QX * QY + (j - 1) * (QX + 1) + i
yface(i, j) = QX * QY + (QX + 1) * QY + (j - 1) * QX + i
const NQ = QX * QY + (QX + 1) * QY + QX * (QY + 1)

function flow_dm!(du, u, dm, t)
    U, D = reshape_local_array(u, dm), reshape_local_array(du, dm)
    for j in axes(D, 2), i in axes(D, 1)
        j <= QY && (
            D[LEFT, 1, i, j] = i == 1 || i == QX + 1 ? 0.0 :
                -(U[ELEM, 1, i, j] - U[ELEM, 1, i - 1, j]) / qx - U[LEFT, 1, i, j]
        )
        i <= QX && (
            D[DOWN, 1, i, j] = j == 1 || j == QY + 1 ? 0.0 :
                -(U[ELEM, 1, i, j] - U[ELEM, 1, i, j - 1]) / qy - U[DOWN, 1, i, j]
        )
        i <= QX && j <= QY && (
            D[ELEM, 1, i, j] = -(
                (U[RIGHT, 1, i, j] - U[LEFT, 1, i, j]) / qx +
                    (U[UP, 1, i, j] - U[DOWN, 1, i, j]) / qy
            ) - U[ELEM, 1, i, j]^3
        )
    end
    return nothing
end

function flow!(dx, x, p, t)
    for j in 1:QY, i in 1:(QX + 1)
        dx[xface(i, j)] = i == 1 || i == QX + 1 ? 0.0 :
            -(x[cell(i, j)] - x[cell(i - 1, j)]) / qx - x[xface(i, j)]
    end
    for j in 1:(QY + 1), i in 1:QX
        dx[yface(i, j)] = j == 1 || j == QY + 1 ? 0.0 :
            -(x[cell(i, j)] - x[cell(i, j - 1)]) / qy - x[yface(i, j)]
    end
    for j in 1:QY, i in 1:QX
        dx[cell(i, j)] = -(
            (x[xface(i + 1, j)] - x[xface(i, j)]) / qx + (x[yface(i, j + 1)] - x[yface(i, j)]) / qy
        ) - x[cell(i, j)]^3
    end
    return nothing
end

function flow_jac_dm!(J, u, dm, t)
    U = reshape_local_array(u, dm)
    is, js = stag_points(dm)
    for j in js, i in is
        j <= QY && 1 < i <= QX && set_stencil_values!(
            J, (LEFT, 1, i, j), ((ELEM, 1, i - 1, j), (ELEM, 1, i, j), (LEFT, 1, i, j)),
            (1 / qx, -1 / qx, -1.0),
        )
        i <= QX && 1 < j <= QY && set_stencil_values!(
            J, (DOWN, 1, i, j), ((ELEM, 1, i, j - 1), (ELEM, 1, i, j), (DOWN, 1, i, j)),
            (1 / qy, -1 / qy, -1.0),
        )
        i <= QX && j <= QY && set_stencil_values!(
            J, (ELEM, 1, i, j),
            ((LEFT, 1, i, j), (RIGHT, 1, i, j), (DOWN, 1, i, j), (UP, 1, i, j), (ELEM, 1, i, j)),
            (1 / qx, -1 / qx, 1 / qy, -1 / qy, -3U[ELEM, 1, i, j]^2),
        )
    end
    return nothing
end

function flow_jac!(J, x, p, t)
    for j in 1:QY, i in 2:QX
        J[xface(i, j), cell(i - 1, j)], J[xface(i, j), cell(i, j)] = 1 / qx, -1 / qx
        J[xface(i, j), xface(i, j)] = -1.0
    end
    for j in 2:QY, i in 1:QX
        J[yface(i, j), cell(i, j - 1)], J[yface(i, j), cell(i, j)] = 1 / qy, -1 / qy
        J[yface(i, j), yface(i, j)] = -1.0
    end
    for j in 1:QY, i in 1:QX
        J[cell(i, j), xface(i, j)], J[cell(i, j), xface(i + 1, j)] = 1 / qx, -1 / qx
        J[cell(i, j), yface(i, j)], J[cell(i, j), yface(i, j + 1)] = 1 / qy, -1 / qy
        J[cell(i, j), cell(i, j)] = -3x[cell(i, j)]^2
    end
    return nothing
end

function flow_proto()
    J = spzeros(NQ, NQ)
    flow_jac!(J, ones(NQ), nothing, 0.0)
    return sparse(1:NQ, 1:NQ, ones(NQ)) + abs.(J)
end

const flow0 = [k <= QX * QY ? sinpi(mod1(k, QX) * qx) * sinpi(cld(k, QX) * qy) : 0.0 for k in 1:NQ]

function on_flow(x, dm)
    u = zeros(PETScDiffEq._dm_local_size(pl, dm))
    a = reshape_local_array(u, dm)
    for j in axes(a, 2), i in axes(a, 1)
        j <= QY && (a[LEFT, 1, i, j] = x[xface(i, j)])
        i <= QX && (a[DOWN, 1, i, j] = x[yface(i, j)])
        i <= QX && j <= QY && (a[ELEM, 1, i, j] = x[cell(i, j)])
    end
    return u
end

function flow_natural(u, dm)
    x = zeros(NQ)
    a = reshape_local_array(u, dm)
    for j in axes(a, 2), i in axes(a, 1)
        j <= QY && (x[xface(i, j)] = a[LEFT, 1, i, j])
        i <= QX && (x[yface(i, j)] = a[DOWN, 1, i, j])
        i <= QX && j <= QY && (x[cell(i, j)] = a[ELEM, 1, i, j])
    end
    return MPI.Reduce(x, +, comm; root = 0)
end


# Reaction-diffusion on a DMPlex box, coupling each vertex to its neighbours along the edges
# or each cell to its neighbours across the edges, held in the mesh's coordinate order so the
# distributed and serial meshes sum the same terms in the same order.
const PLEX_FACES = (5, 4)
const PLEX_K = 4.0

plex_call(code) = PETScDiffEq._check_code(code)
plex_sym(name) = PETScDiffEq._symbol(pl, name)

# PETSc's 64-bit builds take Int64 indices.
function plex_range(dm, name, k)
    lo, hi = Ref(0), Ref(0)
    plex_call(
        ccall(plex_sym(name), Cint, (Ptr{Cvoid}, Int64, Ptr{Int64}, Ptr{Int64}), dm.ptr, k, lo, hi),
    )
    return lo[]:(hi[] - 1)
end

function plex_adjacent(dm, p, size_name, name)
    n, q = Ref(0), Ref{Ptr{Int64}}()
    plex_call(ccall(plex_sym(size_name), Cint, (Ptr{Cvoid}, Int64, Ptr{Int64}), dm.ptr, p, n))
    plex_call(ccall(plex_sym(name), Cint, (Ptr{Cvoid}, Int64, Ptr{Ptr{Int64}}), dm.ptr, p, q))
    return copy(unsafe_wrap(Array, q[], n[]))
end
plex_cone(dm, p) = plex_adjacent(dm, p, :DMPlexGetConeSize, :DMPlexGetCone)
plex_support(dm, p) = plex_adjacent(dm, p, :DMPlexGetSupportSize, :DMPlexGetSupport)

# PETSc distributes the box as DMSetFromOptions builds it.
plex_mesh(c, simplex) = PETSc.DMPlex(
    pl, c; dm_plex_dim = 2, dm_plex_simplex = simplex ? "1" : "0",
    dm_plex_box_faces = join(PLEX_FACES, ","), dm_distribute_overlap = 1,
)

plex_set(name, s, p, k) =
    plex_call(ccall(plex_sym(name), Cint, (Ptr{Cvoid}, Int64, Int64), s, p, k))

function plex_section!(dm, points, cells)
    c, s, lo, hi = Ref{MPI.API.MPI_Comm}(), Ref{Ptr{Cvoid}}(), Ref(0), Ref(0)
    plex_call(
        ccall(plex_sym(:PetscObjectGetComm), Cint, (Ptr{Cvoid}, Ptr{MPI.API.MPI_Comm}), dm.ptr, c),
    )
    plex_call(
        ccall(plex_sym(:PetscSectionCreate), Cint, (MPI.API.MPI_Comm, Ptr{Ptr{Cvoid}}), c[], s),
    )
    plex_call(
        ccall(
            plex_sym(:DMPlexGetChart), Cint, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64}), dm.ptr, lo, hi,
        ),
    )
    plex_set(:PetscSectionSetChart, s[], lo[], hi[])
    for p in points
        plex_set(:PetscSectionSetDof, s[], p, 1)
    end
    plex_call(ccall(plex_sym(:PetscSectionSetUp), Cint, (Ptr{Cvoid},), s[]))
    plex_call(ccall(plex_sym(:DMSetLocalSection), Cint, (Ptr{Cvoid}, Ptr{Cvoid}), dm.ptr, s[]))
    plex_call(ccall(plex_sym(:PetscSectionDestroy), Cint, (Ptr{Ptr{Cvoid}},), s))
    cells && plex_call(
        ccall(plex_sym(:DMSetBasicAdjacency), Cint, (Ptr{Cvoid}, Cint, Cint), dm.ptr, 1, 0),
    )
    return nothing
end

plex_handle(name, obj) = (
    h = Ref{Ptr{Cvoid}}();
    plex_call(ccall(plex_sym(name), Cint, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), obj, h)); h[]
)

function plex_xy(dm)
    v = plex_handle(:DMGetCoordinatesLocal, dm.ptr)
    s = plex_handle(:DMGetCoordinateSection, dm.ptr)
    n, off, a = Ref(0), Ref(0), Ref{Ptr{Float64}}()
    plex_call(ccall(plex_sym(:VecGetLocalSize), Cint, (Ptr{Cvoid}, Ptr{Int64}), v, n))
    plex_call(ccall(plex_sym(:VecGetArrayRead), Cint, (Ptr{Cvoid}, Ptr{Ptr{Float64}}), v, a))
    x = copy(unsafe_wrap(Array, a[], n[]))
    plex_call(ccall(plex_sym(:VecRestoreArrayRead), Cint, (Ptr{Cvoid}, Ptr{Ptr{Float64}}), v, a))
    return Dict(
        map(plex_range(dm, :DMPlexGetDepthStratum, 0)) do p
            plex_call(
                ccall(
                    plex_sym(:PetscSectionGetOffset), Cint, (Ptr{Cvoid}, Int64, Ptr{Int64}),
                    s, p, off,
                ),
            )
            p => (x[off[] + 1], x[off[] + 2])
        end,
    )
end

plex_key(x) = round.(x; digits = 9)

# The points carrying the unknowns, where each sits, and its neighbours in coordinate order.
function plex_layout(dm, cells)
    xy = plex_xy(dm)
    if cells
        points = plex_range(dm, :DMPlexGetHeightStratum, 0)
        corners(c) = unique(v for e in plex_cone(dm, c) for v in plex_cone(dm, e))
        centre(c) = sum(v -> collect(xy[v]), corners(c)) ./ length(corners(c))
        at = Dict(c => plex_key(centre(c)) for c in points)
        across(c) = [x for e in plex_cone(dm, c) for x in plex_support(dm, e) if x != c]
        near = Dict(c => across(c) for c in points)
    else
        points = plex_range(dm, :DMPlexGetDepthStratum, 0)
        at = Dict(v => plex_key(collect(xy[v])) for v in points)
        along(v) = [only(filter(!=(v), plex_cone(dm, e))) for e in plex_support(dm, v)]
        near = Dict(v => along(v) for v in points)
    end
    return collect(points), at, Dict(p => sort(near[p]; by = q -> at[q]) for p in points)
end

function plex_problem(simplex, cells)
    dm = plex_mesh(comm, simplex)
    stratum = cells ? :DMPlexGetHeightStratum : :DMPlexGetDepthStratum
    plex_section!(dm, plex_range(dm, stratum, 0), cells)
    points, at, near = plex_layout(dm, cells)
    probe = reshape_local_array(zeros(PETScDiffEq._dm_local_size(pl, dm)), dm)
    own = [p for p in points if checkbounds(Bool, probe, 1, p)]
    serial = plex_mesh(MPI.COMM_SELF, simplex)
    spoints, sat, snear = plex_layout(serial, cells)
    PETScCompat.destroy!(serial)
    index = Dict(sat[p] => k for (k, p) in enumerate(spoints))
    near_k = [[index[sat[q]] for q in snear[p]] for p in spoints]
    n = length(spoints)
    proto = sparse(
        [k for k in 1:n for _ in 0:length(near_k[k])], reduce(vcat, [k; near_k[k]] for k in 1:n),
        1.0, n, n,
    )
    u0 = zeros(length(own))
    U0 = reshape_local_array(u0, dm)
    start(x) = sinpi(x[1]) * cospi(x[2]) + 0.5
    for p in own
        U0[1, p] = start(at[p])
    end
    return (;
        dm, own, near, slot = [index[at[p]] for p in own], u0, near_k, proto, n,
        x0 = [start(sat[p]) for p in spoints],
    )
end

function plex_natural(u, q)
    x = zeros(q.n)
    a = reshape_local_array(u, q.dm)
    for (p, k) in zip(q.own, q.slot)
        x[k] = a[1, p]
    end
    return MPI.Reduce(x, +, comm; root = 0)
end

function plex_rd_dm!(du, u, q, t)
    U, D = reshape_local_array(u, q.dm), reshape_local_array(du, q.dm)
    for p in q.own
        s = 0.0
        for x in q.near[p]
            s += U[1, x] - U[1, p]
        end
        D[1, p] = PLEX_K * s - U[1, p]^3
    end
    return nothing
end

function plex_rd_jac_dm!(J, u, q, t)
    U = reshape_local_array(u, q.dm)
    for p in q.own
        set_stencil_values!(J, (1, p), (1, p), -PLEX_K * length(q.near[p]) - 3U[1, p]^2)
        for x in q.near[p]
            set_stencil_values!(J, (1, p), (1, x), PLEX_K)
        end
    end
    return nothing
end

function plex_rd!(du, u, q, t)
    for k in eachindex(u)
        s = 0.0
        for j in q.near_k[k]
            s += u[j] - u[k]
        end
        du[k] = PLEX_K * s - u[k]^3
    end
    return nothing
end

function plex_rd_jac!(J, u, q, t)
    for k in eachindex(u)
        J[k, k] = -PLEX_K * length(q.near_k[k]) - 3u[k]^2
        for j in q.near_k[k]
            J[k, j] = PLEX_K
        end
    end
    return nothing
end


@testset "MPI DM, $nranks ranks" begin
    @testset "the algorithms take the DM's communicator" begin
        alg = TSRK(; dm = da)
        @test nranks == 1 ? alg.comm == MPI.COMM_SELF :
            MPI.Comm_compare(alg.comm, comm) == MPI.CONGRUENT
        @test TSRK(; dm = da, comm).comm == comm
        @test TSImplicit("bdf"; dm = da).autodiff isa AutoFiniteDiff
        @test TSRosW(; dm = da, comm).autodiff isa AutoFiniteDiff
        @test refused(() -> TSRK(; dm = 1), "takes a PETSc DM")
        nranks > 1 && @test refused(() -> TSRK(; dm = da, comm = MPI.COMM_SELF), "other ranks")
        integ = init(dm_heat(), TSImplicit("bdf"; dm = da); TOL...)
        ts_dm = held(PETScDiffEq._ts_dm(pl, integ.h.ts))
        @test ts_dm != da.ptr
        @test PETScDiffEq._dm_type(pl, LibPETSc.PetscDM(ts_dm, pl)) == "da"
        terminate!(integ)
        @test everywhere(released(ts_dm))
        @test everywhere(refs(da.ptr) == 1)
    end

    @testset "1-D heat, explicit and implicit" begin
        for (make, kw, err) in ((explicit, FIXED, 2.0e-12), (implicit, TOL, 1.5e-6))
            got = solve(dm_heat(), make(; dm = da, comm); TOL...)
            ref = solve(comm_heat(), make(; comm); TOL...)
            @test got.retcode == ReturnCode.Success
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
            @test same_everywhere(got.stats.nf)
            got = solve(dm_heat(), make(; dm = da); kw...)
            us = natural(got, da, N)
            if rank == 0
                serial = solve(serial_heat(), make(); kw...)
                @test got.t[end] == serial.t[end]
                @test maximum(abs, us[end] - serial.u[end]) <= ROUNDOFF
                @test maximum(abs, us[end] - heat_exact(SPAN[2])) <= err
            end
        end
        ssp(; kw...) = TSGeneric("ssp"; explicit = true, kw...)
        got = solve(dm_heat(), ssp(; dm = da, comm); FIXED...)
        ref = solve(comm_heat(), ssp(; comm); FIXED...)
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)
        for type in ("alpha", "dirk")
            generic(; kw...) = TSGeneric(type; kw...)
            got = solve(dm_heat(), generic(; dm = da, comm); FIXED...)
            ref = solve(comm_heat(), generic(; comm); FIXED...)
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
        end
    end

    @testset "ghost points past the edge of the grid read zero on every call" begin
        function scribbling!(du, u, da, t)
            heat_dm!(du, u, da, t)
            u .= NaN
            return nothing
        end
        got = solve(dm_heat(scribbling!), explicit(; dm = da, comm); FIXED...)
        ref = solve(dm_heat(), explicit(; dm = da, comm); FIXED...)
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)
    end

    @testset "an implicit solve on a periodic grid of any size" begin
        ring = PETSc.DMDA(
            pl, comm, (LibPETSc.DM_BOUNDARY_PERIODIC,), (N,), 1, 1;
            points_per_proc = (LibPETSc.PetscInt.(counts),),
        )
        wave(idx) = cospi.(2 .* idx ./ N) .+ 0.5 .* sinpi.(6 .* idx ./ N)
        got = solve(ODEProblem(heat_dm!, wave(rows), SPAN, ring), implicit(; dm = ring); TOL...)
        @test got.retcode == ReturnCode.Success
        us = natural(got, ring, N)
        if rank == 0
            ring!(du, u, p, t) = laplacian!(du, u, u[end], u[1])
            I = repeat(1:N; inner = 3)
            J = [mod1(i + d, N) for i in 1:N for d in -1:1]
            fn = ODEFunction(ring!; jac_prototype = sparse(I, J, ones(3N), N, N))
            serial = solve(ODEProblem(fn, wave(1:N), SPAN), implicit(); TOL...)
            @test got.t[end] == serial.t[end]
            @test maximum(abs, us[end] - serial.u[end]) <= ROUNDOFF
        end
        PETScCompat.destroy!(ring)
    end

    @testset "2-D heat, explicit and implicit" begin
        slab = uneven(NY)
        across = grid_da((nranks, 1), (LibPETSc.PetscInt.(uneven(NX)), nothing))
        along = grid_da((1, nranks), (nothing, LibPETSc.PetscInt.(slab)))
        ys = owned(slab)
        span = (0.0, 0.05)
        serial = ODEProblem(
            ODEFunction(grid_heat_serial!; jac_prototype = grid_proto(1:NY)),
            vec([grid0(i, j) for i in 1:NX, j in 1:NY]), span,
        )
        for (make, kw, err) in ((explicit, FIXED, 2.0e-9), (implicit, TOL, 4.0e-6))
            got = solve(
                ODEProblem(grid_heat_dm!, grid_u0(across), span, across), make(; dm = across);
                saveat = 0.01, kw...,
            )
            @test got.retcode == ReturnCode.Success
            @test got.t == collect(0.0:0.01:0.05)
            us = natural(got, across, (NX, NY))
            if rank == 0
                ref = solve(serial, make(); saveat = 0.01, kw...)
                @test maxdiff(us, ref.u) <= ROUNDOFF
                @test maxdiff(us, grid_exact.(got.t)) <= err
            end
            got = solve(
                ODEProblem(grid_heat_dm!, grid_u0(along), span, along),
                make(; dm = along, comm); TOL...,
            )
            fn = ODEFunction(grid_heat!; jac_prototype = grid_proto(ys))
            u0 = vec([grid0(i, j) for i in 1:NX, j in ys])
            ref = solve(ODEProblem(fn, u0, span), make(; comm); TOL...)
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
        end
        PETScCompat.destroy!(across)
        PETScCompat.destroy!(along)
    end

    @testset "a DMDA on MPI.COMM_SELF, one to each rank" begin
        solo = line_da(MPI.COMM_SELF)
        for make in (explicit, implicit)
            alg = make(; dm = solo)
            @test alg.comm == MPI.COMM_SELF
            got = solve(ODEProblem(heat_dm!, heat0(1:N), SPAN, solo), alg; TOL...)
            ref = solve(serial_heat(), make(); TOL...)
            @test got.t == ref.t
            @test got.u == ref.u
        end
        back = (0.01, 0.0)
        got = solve(ODEProblem(heat_dm!, heat0(1:N), back, solo), explicit(; dm = solo); TOL...)
        ref = solve(ODEProblem(heat_serial!, heat0(1:N), back), explicit(); TOL...)
        @test got.retcode == ReturnCode.Success
        @test got.t == ref.t
        @test got.u == ref.u
        nan_after(f) = (du, u, p, t) -> (f(du, u, p, t); t > 0.05 && fill!(du, NaN); nothing)
        got = @test_logs (:warn, r"floating point exception") solve(
            ODEProblem(nan_after(heat_dm!), heat0(1:N), SPAN, solo), explicit(; dm = solo),
        )
        ref = @test_logs (:warn, r"floating point exception") solve(
            ODEProblem(nan_after(heat_serial!), heat0(1:N), SPAN), explicit(),
        )
        @test got.retcode == ReturnCode.Unstable
        @test got.stats.nreject > 10
        @test got.t == ref.t
        @test got.u == ref.u
        PETScCompat.destroy!(solo)
    end

    @testset "an adaptive step that turns NaN on every rank leaves the TS's DM free" begin
        nan_after(f) = (du, u, p, t) -> (f(du, u, p, t); t > 0.05 && fill!(du, NaN); nothing)
        for (prob, alg) in (
                (dm_heat(nan_after(heat_dm!)), explicit(; dm = da, comm)),
                (comm_heat(nan_after(heat!)), explicit(; comm)),
            )
            integ = init(prob, alg)
            ts_dm = held(PETScDiffEq._ts_dm(pl, integ.h.ts))
            sol = @test_logs (:warn, r"floating point exception") solve!(integ)
            @test sol.retcode == ReturnCode.Unstable
            @test everywhere(released(ts_dm))
        end
    end

    @testset "saveat and dense output" begin
        rk3(; kw...) = TSRK("3bs"; kw...)
        for (make, kw) in ((explicit, FIXED), (implicit, TOL), (rk3, FIXED))
            got = solve(dm_heat(), make(; dm = da, comm); saveat = 0.01, TOL...)
            ref = solve(comm_heat(), make(; comm); saveat = 0.01, TOL...)
            @test got.t == ref.t == collect(0.0:0.01:0.1)
            @test everywhere(got.u == ref.u)
            times = collect(0.0105:0.01:0.0905)
            got = solve(dm_heat(), make(; dm = da); saveat = times, kw...)
            @test got.t == times
            us = natural(got, da, N)
            if rank == 0
                serial = solve(serial_heat(), make(); saveat = times, kw...)
                @test maxdiff(us, serial.u) <= ROUNDOFF
            end
        end
        got = solve(dm_heat(), explicit(; dm = da, comm); FIXED...)
        ref = solve(comm_heat(), explicit(; comm); FIXED...)
        mids = (got.t[1:(end - 1)] .+ got.t[2:end]) ./ 2
        @test everywhere([got(t) for t in mids] == [ref(t) for t in mids])
    end

    @testset "callbacks" begin
        cases = (
            (;
                callback = DiscreteCallback((u, t, i) -> t == 0.05, i -> (i.u .*= 0.5)),
                tstops = [0.05],
            ),
            (;
                callback = ContinuousCallback(
                    (u, t, i) -> MPI.Allreduce(sum(u), +, comm) - 8.0, terminate!,
                ),
            ),
        )
        rk3(; kw...) = TSRK("3bs"; kw...)
        for kw in cases, make in (explicit, rk3, (; kw...) -> TSImplicit("bdf"; kw...))
            got = solve(dm_heat(), make(; dm = da, comm); kw..., FIXED...)
            ref = solve(comm_heat(), make(; comm); kw..., FIXED...)
            @test got.retcode == ref.retcode
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
        end
        @test solve(dm_heat(), explicit(; dm = da); cases[2]..., FIXED...).retcode ==
            ReturnCode.Terminated
        seen = Int[]
        sol = solve(
            dm_heat(), explicit(; dm = da);
            unstable_check = (dt, u, p, t) -> (push!(seen, length(u)); false),
            isoutofdomain = (u, p, t) -> (push!(seen, length(u)); false), TOL...,
        )
        @test sol.retcode == ReturnCode.Success
        @test !isempty(seen) && all(==(length(rows)), seen)
    end

    @testset "the integrator interface" begin
        function walk(prob, alg)
            integ = init(prob, alg; FIXED...)
            mids, dus = Vector{Float64}[], Vector{Float64}[]
            for _ in 1:30
                step!(integ)
                push!(mids, integ((integ.tprev + integ.t) / 2))
                push!(dus, get_du(integ))
            end
            terminate!(integ)
            return mids, dus
        end
        for make in (explicit, (; kw...) -> TSImplicit("bdf"; kw...))
            got = walk(dm_heat(), make(; dm = da, comm))
            ref = walk(comm_heat(), make(; comm))
            @test everywhere(got == ref)
        end
    end

    @testset "the caller can destroy the DM once the integrator has it" begin
        function ghosted!(du, u, p, t)
            for i in eachindex(du)
                du[i] = (u[i] - 2u[i + 1] + u[i + 2]) / dx^2
            end
            return nothing
        end
        prob = ODEProblem(ghosted!, heat0(rows), SPAN)
        fresh = line_da(comm; points_per_proc = (LibPETSc.PetscInt.(counts),))
        integ = init(prob, explicit(; dm = fresh); FIXED...)
        PETScCompat.destroy!(fresh)
        got = solve!(integ)
        ref = solve(prob, explicit(; dm = da); FIXED...)
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)
    end

    @testset "a mass matrix, a SplitODEProblem and a DAEProblem" begin
        d = 1 .+ (1:N) ./ N
        scaled(f) = (du, u, p, t) -> (f(du, u, p, t); du .*= d[rows]; nothing)
        for make in ((; kw...) -> TSImplicit("bdf"; kw...), (; kw...) -> TSRosW(; kw...))
            mass = Diagonal(d[rows])
            got = solve(dm_heat(scaled(heat_dm!); mass_matrix = mass), make(; dm = da, comm); TOL...)
            ref = solve(comm_heat(scaled(heat!); mass_matrix = mass), make(; comm); TOL...)
            @test got.retcode == ReturnCode.Success
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
        end

        function decay_dm!(du, u, da, t)
            U, D = reshape_local_array(u, da), reshape_local_array(du, da)
            for i in axes(D, 2)
                D[1, i] = -U[1, i]
            end
            return nothing
        end
        decay!(du, u, p, t) = (du .= -u; nothing)
        got = solve(
            SplitODEProblem(decay_dm!, heat_dm!, heat0(rows), SPAN, da), TSARKIMEX(; dm = da, comm);
            TOL...,
        )
        n = length(rows)
        f1 = ODEFunction(decay!; jac_prototype = sparse(1:n, rows, ones(n), n, N))
        ref = solve(SplitODEProblem(f1, heat!, heat0(rows), SPAN), TSARKIMEX(; comm); TOL...)
        @test got.retcode == ReturnCode.Success
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)
        u = natural(got.u[end], da, N)
        rank == 0 && @test maximum(abs, u - exp(-SPAN[2]) .* heat_exact(SPAN[2])) <= 5.0e-9

        residual(f) = (r, du, u, p, t) -> (f(r, u, p, t); r .= du .- r; nothing)
        du0 = similar(heat0(rows))
        heat!(du0, heat0(rows), nothing, 0.0)
        got = solve(
            DAEProblem(residual(heat_dm!), du0, heat0(rows), SPAN, da), TSDAE("bdf"; dm = da, comm);
            TOL...,
        )
        fn = DAEFunction(residual(heat!); jac_prototype = heat_proto(rows))
        ref = solve(DAEProblem(fn, du0, heat0(rows), SPAN), TSDAE("bdf"; comm); TOL...)
        @test got.retcode == ReturnCode.Success
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)
    end

    @testset "BrownFullBasicInit and ShampineCollocationInit with a DM" begin
        algebraic(i) = i % 4 == 0
        m = [algebraic(i) ? 0.0 : 1.0 for i in rows]
        cubic(u, l, r) = u^3 + u - (l + r) / 2 - 0.1
        function chain_dm!(du, u, da, t)
            U, D = reshape_local_array(u, da), reshape_local_array(du, da)
            for i in axes(D, 2)
                l, c, r = U[1, i - 1], U[1, i], U[1, i + 1]
                D[1, i] = algebraic(i) ? cubic(c, l, r) : l - 2c + r
            end
            return nothing
        end
        seen = Ref(0)
        function chain_jac_dm!(J, u, da, gamma)
            seen[] += 1
            U = reshape_local_array(u, da)
            for (k, i) in enumerate(rows)
                c = U[1, i]
                vals = algebraic(i) ? [-0.5, 3c^2 + 1, -0.5] : [1.0, -2.0, 1.0]
                gamma === nothing || (vals = [0, gamma * m[k], 0] .- vals)
                set_stencil_values!(J, (1, i), [(1, i - 1), (1, i), (1, i + 1)], vals)
            end
            return nothing
        end
        function chain!(du, u, p, t)
            left, right = halo(u)
            for (k, i) in enumerate(rows)
                l = k == 1 ? left : u[k - 1]
                r = k == length(u) ? right : u[k + 1]
                du[k] = algebraic(i) ? cubic(u[k], l, r) : l - 2u[k] + r
            end
            return nothing
        end
        residual(f) = (r, du, u, p, t) -> (f(r, u, p, t); r .= m .* du .- r; nothing)
        ode_jac_dm!(J, u, da, t) = chain_jac_dm!(J, u, da, nothing)
        dae_jac_dm!(J, du, u, da, gamma, t) = chain_jac_dm!(J, u, da, gamma)
        function problems(form; jac)
            if form == :dae
                fn = jac ? DAEFunction(residual(chain_dm!); jac = dae_jac_dm!) :
                    DAEFunction(residual(chain_dm!))
                plain = DAEFunction(residual(chain!); jac_prototype = heat_proto(rows))
                dv = m .!= 0
                with_dm = DAEProblem(fn, zero(m), heat0(rows), SPAN, da; differential_vars = dv)
                without = DAEProblem(plain, zero(m), heat0(rows), SPAN; differential_vars = dv)
                return with_dm, without
            end
            fn = jac ? ODEFunction(chain_dm!; jac = ode_jac_dm!, mass_matrix = Diagonal(m)) :
                ODEFunction(chain_dm!; mass_matrix = Diagonal(m))
            return ODEProblem(fn, heat0(rows), SPAN, da),
                comm_heat(chain!; mass_matrix = Diagonal(m))
        end
        method(form; kw...) = form == :dae ? TSDAE("bdf"; kw...) : TSImplicit("bdf"; kw...)
        for form in (:mass, :dae), jac in (true, false),
                ia in (DiffEqBase.BrownFullBasicInit(), DiffEqBase.ShampineCollocationInit())
            with_dm, without = problems(form; jac)
            @test caught(() -> solve(with_dm, method(form; dm = da, comm); TOL...)) isa
                SciMLBase.CheckInitFailureError
            got = solve(with_dm, method(form; dm = da, comm); initializealg = ia, TOL...)
            ref = solve(without, method(form; comm); initializealg = ia, TOL...)
            @test got.retcode == ref.retcode == ReturnCode.Success
            @test anywhere(maximum(abs, got.u[1] - heat0(rows)) > 0.1)
            gap = jac ? INIT_JAC_GAP : INIT_GAP
            @test everywhere(maximum(abs, got.u[1] - ref.u[1]) <= gap)
            seen[] = 0
            integ = init(with_dm, method(form; dm = da, comm); initializealg = ia, TOL...)
            # Initialization fills the DM's matrix from the `jac` on a communicator too.
            @test (seen[] > 0) == jac
            SciMLBase.set_u!(integ, integ.u .+ 0.05)
            SciMLBase.initialize_dae!(integ)
            plain = init(without, method(form; comm); initializealg = ia, TOL...)
            SciMLBase.set_u!(plain, plain.u .+ 0.05)
            SciMLBase.initialize_dae!(plain)
            @test everywhere(maximum(abs, integ.u - plain.u) <= gap)
            terminate!(integ)
            terminate!(plain)
        end
        function throwing_jac!(J, u, da, t)
            rank == thrower && error("jac threw on rank $rank")
            return ode_jac_dm!(J, u, da, t)
        end
        fn = ODEFunction(chain_dm!; jac = throwing_jac!, mass_matrix = Diagonal(m))
        for ia in (DiffEqBase.BrownFullBasicInit(), DiffEqBase.ShampineCollocationInit())
            e = caught(
                () -> init(
                    ODEProblem(fn, heat0(rows), SPAN, da), TSImplicit("bdf"; dm = da, comm);
                    initializealg = ia, TOL...,
                ),
            )
            @test raised(e, "jac threw")
        end
        @test everywhere(refs(da.ptr) == 1)
    end

    @testset "TSARKIMEX takes its first stage at the step's start" begin
        a = 2 .+ (1:N) ./ N
        exact(t) = 1 ./ (a[rows] .- sin(t))
        function rhs!(du, u, da, t)
            U, D = reshape_local_array(u, da), reshape_local_array(du, da)
            for i in axes(D, 2)
                D[1, i] = U[1, i]^2 * cos(t)
            end
            return nothing
        end
        function jac!(J, u, da, t)
            U = reshape_local_array(u, da)
            for i in rows
                J[i, i] = 2 * U[1, i] * cos(t)
            end
            return nothing
        end
        tight = ["-snes_rtol", "1e-12", "-snes_atol", "1e-14", "-ksp_rtol", "1e-12"]
        for jac in (nothing, jac!), span in ((1.0, 2.0), (2.0, 1.0))
            sol = solve(
                ODEProblem(ODEFunction(rhs!; jac), exact(span[1]), span, da),
                TSARKIMEX("4", tight; dm = da, comm); dt = 0.05, adaptive = false,
            )
            err = MPI.Allreduce(maximum(abs, sol.u[end] - exact(span[2])), max, comm)
            @test err < 1.0e-7
        end
    end

    @testset "a jac fills the DM's matrix" begin
        stencil(i) = ([(1, i - 1), (1, i), (1, i + 1)], [1, -2, 1] ./ dx^2)
        njac = Ref(0)
        ghosted = Ref(true)
        function heat_jac_dm!(J, u, da, t)
            njac[] += 1
            ghosted[] &= length(u) == length(rows) + 2
            for i in rows
                set_stencil_values!(J, (1, i), stencil(i)...)
            end
            return nothing
        end
        function heat_jac_global!(J, u, da, t)
            for i in rows
                J[i, i] = -2 / dx^2
                i > 1 && (J[i, i - 1] = 1 / dx^2)
                i < N && (J[i, i + 1] = 1 / dx^2)
            end
            return nothing
        end
        heat_jac_rows(idx, scale = i -> 1.0) = function (J, u, p, t)
            for (k, i) in enumerate(idx)
                J[k, i] = -2scale(i) / dx^2
                i > 1 && (J[k, i - 1] = scale(i) / dx^2)
                i < N && (J[k, i + 1] = scale(i) / dx^2)
            end
            return nothing
        end
        bdf(; kw...) = TSImplicit("bdf"; kw...)
        rosw(; kw...) = TSRosW(; kw...)
        for (make, serial_tol) in ((bdf, JAC_SERIAL_TOL.bdf), (rosw, JAC_SERIAL_TOL.rosw))
            njac[] = 0
            got = solve(dm_heat(; jac = heat_jac_dm!), make(; dm = da, comm); saveat = 0.01, TOL...)
            @test got.retcode == ReturnCode.Success
            @test ghosted[]
            @test got.stats.njacs == njac[] > 0
            @test same_everywhere(got.stats.njacs)
            @test same_everywhere(got.stats.nf)
            ref = solve(comm_heat(; jac = heat_jac_rows(rows)), make(; comm); saveat = 0.01, TOL...)
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
            by_index = solve(
                dm_heat(; jac = heat_jac_global!), make(; dm = da, comm); saveat = 0.01, TOL...,
            )
            @test by_index.t == got.t
            @test everywhere(by_index.u == got.u)
            coloured = solve(dm_heat(), make(; dm = da, comm); saveat = 0.01, TOL...)
            @test coloured.stats.njacs == 0
            @test got.stats.nf < coloured.stats.nf
            us, cs = natural(got, da, N), natural(coloured, da, N)
            if rank == 0
                serial = solve(
                    serial_heat(; jac = heat_jac_rows(1:N)), make(); saveat = 0.01, TOL...,
                )
                @test maxdiff(us, serial.u) <= serial_tol
                @test maxdiff(us, cs) <= COLOUR_TOL
            end
        end
        with_ad = solve(
            dm_heat(; jac = heat_jac_dm!), bdf(; dm = da, comm, autodiff = AutoForwardDiff());
            TOL...,
        )
        plain = solve(dm_heat(; jac = heat_jac_dm!), bdf(; dm = da, comm); TOL...)
        @test with_ad.t == plain.t
        @test everywhere(with_ad.u == plain.u)

        slab = uneven(NY)
        along = grid_da((1, nranks), (nothing, LibPETSc.PetscInt.(slab)))
        across = grid_da((nranks, 1), (LibPETSc.PetscInt.(uneven(NX)), nothing))
        own = Ref{Any}(nothing)
        function grid_jac_dm!(J, u, da, t)
            for I in CartesianIndices(own[])
                i, j = Tuple(I)
                cols = [(1, i, j), (1, i - 1, j), (1, i + 1, j), (1, i, j - 1), (1, i, j + 1)]
                vals = [-2 / hx^2 - 2 / hy^2, 1 / hx^2, 1 / hx^2, 1 / hy^2, 1 / hy^2]
                set_stencil_values!(J, (1, i, j), cols, vals)
            end
            return nothing
        end
        point(i, j) = i + (j - 1) * NX
        function grid_jac_serial!(J, u, p, t)
            for j in 1:NY, i in 1:NX
                J[point(i, j), point(i, j)] = -2 / hx^2 - 2 / hy^2
                for (a, b, w) in ((i - 1, j, hx), (i + 1, j, hx), (i, j - 1, hy), (i, j + 1, hy))
                    1 <= a <= NX && 1 <= b <= NY && (J[point(i, j), point(a, b)] = 1 / w^2)
                end
            end
            return nothing
        end
        span = (0.0, 0.05)
        serial = ODEProblem(
            ODEFunction(
                grid_heat_serial!; jac = grid_jac_serial!, jac_prototype = grid_proto(1:NY),
            ),
            vec([grid0(i, j) for i in 1:NX, j in 1:NY]), span,
        )
        for (make, serial_tol) in ((bdf, JAC_SERIAL_TOL.bdf), (rosw, JAC_SERIAL_TOL.rosw)),
                g in (along, across)

            u0 = grid_u0(g)
            own[] = axes(reshape_local_array(u0, g))[2:end]
            got = solve(
                ODEProblem(ODEFunction(grid_heat_dm!; jac = grid_jac_dm!), u0, span, g),
                make(; dm = g); saveat = 0.01, TOL...,
            )
            @test got.retcode == ReturnCode.Success
            @test got.stats.njacs > 0
            coloured = solve(
                ODEProblem(grid_heat_dm!, u0, span, g), make(; dm = g); saveat = 0.01, TOL...,
            )
            us, cs = natural(got, g, (NX, NY)), natural(coloured, g, (NX, NY))
            if rank == 0
                ref = solve(serial, make(); saveat = 0.01, TOL...)
                @test maxdiff(us, ref.u) <= serial_tol
                @test maxdiff(us, cs) <= COLOUR_TOL
            end
        end
        PETScCompat.destroy!(along)
        PETScCompat.destroy!(across)

        d = 1 .+ (1:N) ./ N
        scaled(f) = (du, u, p, t) -> (f(du, u, p, t); du .*= d[rows]; nothing)
        function scaled_jac_dm!(J, u, da, t)
            for i in rows
                set_stencil_values!(J, (1, i), stencil(i)[1], [d[i], -2d[i], d[i]] ./ dx^2)
            end
            return nothing
        end
        mass = Diagonal(d[rows])
        for make in (bdf, rosw)
            got = solve(
                dm_heat(scaled(heat_dm!); mass_matrix = mass, jac = scaled_jac_dm!),
                make(; dm = da, comm); saveat = 0.01, TOL...,
            )
            ref = solve(
                comm_heat(scaled(heat!); mass_matrix = mass, jac = heat_jac_rows(rows, i -> d[i])),
                make(; comm); saveat = 0.01, TOL...,
            )
            @test got.retcode == ReturnCode.Success
            @test got.t == ref.t
            @test everywhere(got.u == ref.u)
        end

        residual(f) = (r, du, u, p, t) -> (f(r, u, p, t); r .= du .- r; nothing)
        function dae_jac_dm!(J, du, u, da, gamma, t)
            for i in rows
                cols, vals = stencil(i)
                set_stencil_values!(J, (1, i), cols, gamma .* [0, 1, 0] .- vals)
            end
            return nothing
        end
        function dae_jac_rows!(J, du, u, p, gamma, t)
            for (k, i) in enumerate(rows)
                J[k, i] = gamma + 2 / dx^2
                i > 1 && (J[k, i - 1] = -1 / dx^2)
                i < N && (J[k, i + 1] = -1 / dx^2)
            end
            return nothing
        end
        du0 = similar(heat0(rows))
        heat!(du0, heat0(rows), nothing, 0.0)
        got = solve(
            DAEProblem(
                DAEFunction(residual(heat_dm!); jac = dae_jac_dm!), du0, heat0(rows), SPAN, da,
            ),
            TSDAE("bdf"; dm = da, comm); saveat = 0.01, TOL...,
        )
        fn = DAEFunction(residual(heat!); jac = dae_jac_rows!, jac_prototype = heat_proto(rows))
        ref = solve(
            DAEProblem(fn, du0, heat0(rows), SPAN), TSDAE("bdf"; comm); saveat = 0.01, TOL...,
        )
        @test got.retcode == ReturnCode.Success
        @test got.stats.njacs > 0
        @test got.t == ref.t
        @test everywhere(got.u == ref.u)

        solo = line_da(MPI.COMM_SELF)
        solo_jac!(J, u, da, t) = (
            for i in 1:N
                set_stencil_values!(J, (1, i), stencil(i)...)
            end; nothing
        )
        back = (0.01, 0.0)
        got = solve(
            ODEProblem(ODEFunction(heat_dm!; jac = solo_jac!), heat0(1:N), back, solo),
            bdf(; dm = solo); TOL...,
        )
        fn = ODEFunction(heat_serial!; jac = heat_jac_rows(1:N), jac_prototype = heat_proto(1:N))
        ref = solve(ODEProblem(fn, heat0(1:N), back), bdf(); TOL...)
        @test got.retcode == ReturnCode.Success
        @test got.t == ref.t
        @test got.u == ref.u
        PETScCompat.destroy!(solo)

        function throwing_jac!(J, u, da, t)
            heat_jac_dm!(J, u, da, t)
            rank == thrower && t > 0.02 && error("jac threw on rank $rank")
            return nothing
        end
        e = caught(() -> solve(dm_heat(; jac = throwing_jac!), bdf(; dm = da); TOL...))
        @test raised(e, "jac threw")

        never(J, u, da, t) = error("an explicit solve called its jac")
        @test everywhere(
            solve(dm_heat(; jac = never), explicit(; dm = da); FIXED...).u ==
                solve(dm_heat(), explicit(; dm = da); FIXED...).u,
        )
        @test everywhere(refs(da.ptr) == 1)
    end

    @testset "f throwing on one rank raises on every rank" begin
        after = t -> t > 0.02
        dense = (; FIXED..., saveat = [0.0105], dense = true)
        for (when, alg, kw) in (
                (t -> t == 0, explicit(; dm = da), (;)),
                (after, explicit(; dm = da), (;)),
                (after, implicit(; dm = da), (;)),
                (t -> t == 0.0105, explicit(; dm = da), dense),
            )
            e = caught(() -> solve(dm_heat(throwing(heat_dm!, when)), alg; kw...))
            @test raised(e, "f threw")
        end
        @test solve(dm_heat(throwing(heat_dm!, t -> false)), implicit(; dm = da)).retcode ==
            ReturnCode.Success
        @test everywhere(refs(da.ptr) == 1)
    end

    @testset "refusals" begin
        prob = dm_heat()
        for alg in (TSIRK(2; dm = da), TSMPRK([1]; dm = da), TSGeneric("irk"; dm = da))
            @test refused(() -> solve(prob, alg; dt = 1.0e-3), "cannot run")
        end
        @test refused(
            () -> solve(
                ODEProblem(
                    ODEFunction{false}((u, p, t) -> -u; jac = (u, p, t) -> nothing), heat0(rows),
                    SPAN, da,
                ),
                implicit(; dm = da),
            ),
            "has to be in place",
        )
        @test refused(
            () -> solve(dm_heat(; jac_prototype = heat_proto(rows)), implicit(; dm = da)),
            "leave out `jac_prototype`",
        )
        @test refused(
            () -> solve(prob, TSImplicit("bdf"; dm = da, autodiff = AutoForwardDiff())),
            "cannot use `AutoForwardDiff()`",
        )
        n = length(rows)
        dense_mass = [i == j ? 2.0 : 0.0 for i in 1:n, j in 1:n]
        @test refused(
            () -> solve(dm_heat(; mass_matrix = dense_mass), implicit(; dm = da)),
            "only a `Diagonal` mass matrix",
        )
        @test refused(
            () -> solve(prob, TSImplicit("bdf", ["-ts_type", "irk"]; dm = da)),
            "`irk` cannot run with a `dm`",
        )
        @test refused(
            () -> PETScDiffEq._discrete_adjoint(
                ODEProblem((du, u, p, t) -> heat_dm!(du, u, da, t), heat0(rows), SPAN),
                TSRK("4"; dm = da), PETScAdjoint(); t = [0.1],
                dgdu_discrete = (out, u, p, t, i) -> (out .= u), dt = 0.01, adaptive = false,
            ),
            "PETScAdjoint needs the ODEFunction's `jac` with a `dm`",
        )
        shell = LibPETSc.DMShellCreate(pl, comm)
        @test refused(
            () -> solve(ODEProblem(heat_dm!, heat0(rows), SPAN, shell), explicit(; dm = shell)),
            "only a DMDA",
        )
        LibPETSc.DMDestroy(pl, shell)

        short = rank == thrower ? heat0(rows)[1:(end - 1)] : heat0(rows)
        e = caught(() -> solve(ODEProblem(heat_dm!, short, SPAN, da), explicit(; dm = da)))
        @test refused_on_thrower(e, "`u0` has $(length(short)) entries")
        e = caught(
            () -> solve(
                dm_heat(; mass_matrix = rank == thrower ? dense_mass : Diagonal(ones(n))),
                implicit(; dm = da),
            ),
        )
        @test refused_on_thrower(e, "only a `Diagonal` mass matrix")
        single = rank == thrower
        prob = ODEProblem(
            heat_dm!, single ? Float32.(heat0(rows)) : heat0(rows),
            single ? Float32.(SPAN) : SPAN, da,
        )
        e = caught(() -> solve(prob, explicit(; dm = da)))
        @test refused_on_thrower(e, "belongs to PETSc's")
        da32 = PETSc.DMDA(
            PETSc.getlib(; PetscScalar = Float32), comm, (GHOSTED,), (N,), 1, 1;
            points_per_proc = (LibPETSc.PetscInt.(counts),),
        )
        @test refused(
            () -> solve(
                ODEProblem(heat_dm!, Float32.(heat0(rows)), SPAN, da32), explicit(; dm = da32),
            ),
            "Float32 real build, but `u0` and `tspan` together run this problem in its " *
                "Float64 real one",
        )
        PETScCompat.destroy!(da32)
    end

    @testset "a DMStag: f, jac and the rest by staggered point" begin
        direct = ["-ksp_type", "preonly", "-pc_type", "redundant"]
        bdf(; kw...) = TSImplicit("bdf", direct; kw...)
        rosw(; kw...) = TSRosW("ra34pw2", direct; kw...)
        damping = [1.0, 1.0]
        u0 = on_wave(wave0, stag)
        back = wave_natural(u0, stag)
        rank == 0 && @test back == wave0
        first_cell = sum(uneven(NS)[1:rank]) + 1
        @test stag_points(stag)[1] ==
            first_cell:(first_cell + uneven(NS)[rank + 1] - (rank == nranks - 1 ? 0 : 1))
        dm_wave(; kw...) = ODEProblem(ODEFunction(wave_dm!; kw...), u0, SPAN, damping)
        serial_wave(; kw...) = ODEProblem(
            ODEFunction(wave!; jac_prototype = wave_proto, kw...), wave0, SPAN, damping,
        )

        halve = DiscreteCallback((u, t, i) -> t == 0.05, i -> (i.u .*= 0.5))
        events = (; saveat = 0.02, callback = halve, tstops = [0.05], FIXED...)
        got = solve(dm_wave(), explicit(; dm = stag); events...)
        @test got.retcode == ReturnCode.Success
        us = wave_natural.(got.u, Ref(stag))
        if rank == 0
            ref = solve(serial_wave(), explicit(); events...)
            @test got.t == ref.t
            @test us == ref.u
        end

        for make in (bdf, rosw)
            got = solve(dm_wave(; jac = wave_jac_dm!), make(; dm = stag); saveat = 0.02, TOL...)
            coloured = solve(dm_wave(), make(; dm = stag); saveat = 0.02, TOL...)
            @test got.retcode == coloured.retcode == ReturnCode.Success
            @test got.stats.njacs > 0
            @test coloured.stats.njacs == 0
            @test got.stats.nf < coloured.stats.nf
            @test same_everywhere(got.stats.nf)
            us, cs = wave_natural.(got.u, Ref(stag)), wave_natural.(coloured.u, Ref(stag))
            if rank == 0
                ref = solve(serial_wave(; jac = wave_jac!), make(); saveat = 0.02, TOL...)
                @test maxdiff(us, ref.u) <= STAG_SERIAL_TOL
                @test maxdiff(cs, us) <= COLOUR_TOL
            end
        end

        weights = 1 .+ (1:NW) ./ NW
        got = solve(
            dm_wave(; jac = wave_jac_dm!, mass_matrix = Diagonal(on_wave(weights, stag))),
            bdf(; dm = stag); saveat = 0.02, TOL...,
        )
        us = wave_natural.(got.u, Ref(stag))
        if rank == 0
            ref = solve(
                serial_wave(; jac = wave_jac!, mass_matrix = Diagonal(weights)), bdf();
                saveat = 0.02, TOL...,
            )
            @test maxdiff(us, ref.u) <= STAG_SERIAL_TOL
        end

        tight = ["-snes_rtol", "1e-13", "-snes_atol", "1e-15", direct...]
        function sink_dm!(du, u, p, t)
            U, D = reshape_local_array(u, stag), reshape_local_array(du, stag)
            for i in axes(D, 1)
                D[LEFT, 1, i] = 0.0
                i <= NS && (D[ELEM, 1, i] = -0.5 * U[ELEM, 1, i])
            end
            return nothing
        end
        sink!(dx, x, p, t) = (dx .= ifelse.(isodd.(1:NW), 0.0, -0.5 .* x); nothing)
        got = solve(
            SplitODEProblem(wave_dm!, sink_dm!, u0, SPAN, damping),
            TSARKIMEX("3", tight; dm = stag); saveat = 0.02, TOL...,
        )
        us = wave_natural.(got.u, Ref(stag))
        residual(f) = (r, du, u, p, t) -> (f(r, u, p, t); r .= du .- r; nothing)
        dwave0 = similar(wave0)
        wave!(dwave0, wave0, damping, 0.0)
        dae = solve(
            DAEProblem(residual(wave_dm!), on_wave(dwave0, stag), u0, SPAN, damping),
            TSDAE("bdf", tight; dm = stag); saveat = 0.02, TOL...,
        )
        ds = wave_natural.(dae.u, Ref(stag))
        @test got.retcode == dae.retcode == ReturnCode.Success
        if rank == 0
            stiff = ODEFunction(wave!; jac_prototype = wave_proto)
            ref = solve(
                SplitODEProblem(stiff, sink!, wave0, SPAN, damping), TSARKIMEX("3", tight);
                saveat = 0.02, TOL...,
            )
            @test maxdiff(us, ref.u) <= STAG_SERIAL_TOL
            fn = DAEFunction(residual(wave!); jac_prototype = wave_proto)
            ref = solve(
                DAEProblem(fn, dwave0, wave0, SPAN, damping), TSDAE("bdf", tight);
                saveat = 0.02, TOL...,
            )
            @test maxdiff(ds, ref.u) <= STAG_SERIAL_TOL
        end

        function walk(integ, gather)
            seen = map(1:20) do _
                step!(integ)
                gather.((integ.u, integ((integ.tprev + integ.t) / 2), get_du(integ)))
            end
            terminate!(integ)
            return seen
        end
        integ = init(dm_wave(; jac = wave_jac_dm!), bdf(; dm = stag); FIXED...)
        ts_dm = held(PETScDiffEq._ts_dm(pl, integ.h.ts))
        @test PETScDiffEq._dm_type(pl, LibPETSc.PetscDM(ts_dm, pl)) == "stag"
        seen = walk(integ, u -> wave_natural(u, stag))
        @test everywhere(released(ts_dm))
        if rank == 0
            ref = walk(init(serial_wave(; jac = wave_jac!), bdf(); FIXED...), copy)
            @test maximum(maxdiff(a, b) for (a, b) in zip(seen, ref)) <= STAG_STEP_TOL
        end

        # The flux on each end is pinned to 0.1, which the initial state misses.
        function pinned_dm!(du, u, p, t)
            wave_dm!(du, u, p, t)
            U, D = reshape_local_array(u, stag), reshape_local_array(du, stag)
            for i in intersect(axes(D, 1), (1, NS + 1))
                D[LEFT, 1, i] = 0.1 - U[LEFT, 1, i]
            end
            return nothing
        end
        function pinned!(dx, x, p, t)
            wave!(dx, x, p, t)
            dx[1], dx[NW] = 0.1 - x[1], 0.1 - x[NW]
            return nothing
        end
        m = [k in (1, NW) ? 0.0 : 1.0 for k in 1:NW]
        for ia in (DiffEqBase.BrownFullBasicInit(), DiffEqBase.ShampineCollocationInit())
            got = solve(
                ODEProblem(
                    ODEFunction(pinned_dm!; mass_matrix = Diagonal(on_wave(m, stag))), u0, SPAN,
                    damping,
                ),
                bdf(; dm = stag); initializealg = ia, TOL...,
            )
            @test got.retcode == ReturnCode.Success
            start = wave_natural(got.u[1], stag)
            if rank == 0
                fn = ODEFunction(pinned!; jac_prototype = wave_proto, mass_matrix = Diagonal(m))
                ref = solve(
                    ODEProblem(fn, wave0, SPAN, damping), bdf(); initializealg = ia, TOL...,
                )
                @test maximum(abs, start[[1, NW]] .- 0.1) <= INIT_GAP
                @test maximum(abs, start - ref.u[1]) <= INIT_GAP
            end
        end

        cost = (out, u, p, t, i) -> (out .= u; nothing)
        exact = ["-snes_rtol", "1e-13", "-snes_atol", "1e-15", "-ksp_type", "preonly"]
        for make in (
                (c; kw...) -> TSRK("4"; kw...),
                (c; kw...) -> TSImplicit(
                    "cn", [exact; "-pc_type"; c == MPI.COMM_SELF ? "lu" : "redundant"]; kw...,
                ),
            )
            fn = ODEFunction(wave_dm!; jac = wave_jac_dm!, paramjac = wave_paramjac_dm!)
            du0, dp = PETScDiffEq._discrete_adjoint(
                ODEProblem(fn, u0, (0.0, 0.05), [0.7, 1.3]), make(comm; dm = stag), PETScAdjoint();
                t = [0.0, 0.025, 0.05], dgdu_discrete = cost, dt = 1.0e-3, adaptive = false,
            )
            @test same_everywhere(dp)
            mine = vcat(wave_natural(du0, stag), vec(dp))
            if rank == 0
                fn = ODEFunction(
                    wave!; jac = wave_jac!, paramjac = wave_paramjac!, jac_prototype = wave_proto,
                )
                sdu0, sdp = PETScDiffEq._discrete_adjoint(
                    ODEProblem(fn, wave0, (0.0, 0.05), [0.7, 1.3]), make(MPI.COMM_SELF),
                    PETScAdjoint();
                    t = [0.0, 0.025, 0.05], dgdu_discrete = cost, dt = 1.0e-3, adaptive = false,
                )
                ref = vcat(sdu0, vec(sdp))
                @test maximum(abs, mine - ref) / maximum(abs, ref) <= STAG_ADJOINT_GAP
            end
        end

        for (processors, stencil) in (
                ((nranks, 1), LibPETSc.DMSTAG_STENCIL_BOX),
                ((1, nranks), LibPETSc.DMSTAG_STENCIL_STAR),
            )
            plane = PETSc.DMStag(
                pl, comm, (GHOSTED, GHOSTED), (QX, QY), (0, 1, 1), 1, stencil; processors,
            )
            span = (0.0, 0.05)
            v0 = on_flow(flow0, plane)
            got = solve(
                ODEProblem(flow_dm!, v0, span, plane), explicit(; dm = plane);
                saveat = 0.01, FIXED...,
            )
            us = flow_natural.(got.u, Ref(plane))
            if rank == 0
                ref = solve(ODEProblem(flow!, flow0, span), explicit(); saveat = 0.01, FIXED...)
                @test got.t == ref.t
                @test us == ref.u
            end
            got = solve(
                ODEProblem(ODEFunction(flow_dm!; jac = flow_jac_dm!), v0, span, plane),
                bdf(; dm = plane); saveat = 0.01, TOL...,
            )
            coloured = solve(
                ODEProblem(flow_dm!, v0, span, plane), bdf(; dm = plane); saveat = 0.01, TOL...,
            )
            @test got.retcode == coloured.retcode == ReturnCode.Success
            @test got.stats.nf < coloured.stats.nf
            us, cs = flow_natural.(got.u, Ref(plane)), flow_natural.(coloured.u, Ref(plane))
            if rank == 0
                ref = solve(
                    ODEProblem(
                        ODEFunction(flow!; jac = flow_jac!, jac_prototype = flow_proto()), flow0,
                        span,
                    ),
                    bdf(); saveat = 0.01, TOL...,
                )
                @test maxdiff(us, ref.u) <= STAG_SERIAL_TOL
                @test maxdiff(cs, us) <= COLOUR_TOL
            end
            PETScCompat.destroy!(plane)
        end
        @test everywhere(refs(stag.ptr) == 1)
    end

    @testset "a DMPlex: f and jac by mesh point" begin
        direct = ["-ksp_type", "preonly", "-pc_type", "redundant"]
        bdf(; kw...) = TSImplicit("bdf", direct; kw...)
        rosw(; kw...) = TSRosW("ra34pw2", direct; kw...)
        for (simplex, cells) in ((true, false), (false, true))
            q = plex_problem(simplex, cells)
            back = plex_natural(q.u0, q)
            rank == 0 && @test back == q.x0
            dm_rd(; kw...) = ODEProblem(ODEFunction(plex_rd_dm!; kw...), q.u0, SPAN, q)
            serial_rd(; kw...) = ODEProblem(
                ODEFunction(plex_rd!; jac_prototype = q.proto, kw...), q.x0, SPAN, q,
            )
            got = solve(dm_rd(), explicit(; dm = q.dm); saveat = 0.02, FIXED...)
            @test got.retcode == ReturnCode.Success
            us = plex_natural.(got.u, Ref(q))
            if rank == 0
                ref = solve(serial_rd(), explicit(); saveat = 0.02, FIXED...)
                @test got.t == ref.t
                @test us == ref.u
            end
            for make in (bdf, rosw)
                got = solve(
                    dm_rd(; jac = plex_rd_jac_dm!), make(; dm = q.dm); saveat = 0.02, TOL...,
                )
                coloured = solve(dm_rd(), make(; dm = q.dm); saveat = 0.02, TOL...)
                @test got.retcode == coloured.retcode == ReturnCode.Success
                @test got.stats.njacs > 0
                @test coloured.stats.njacs == 0
                @test got.stats.nf < coloured.stats.nf
                @test same_everywhere(got.stats.nf)
                us, cs = plex_natural.(got.u, Ref(q)), plex_natural.(coloured.u, Ref(q))
                if rank == 0
                    ref = solve(serial_rd(; jac = plex_rd_jac!), make(); saveat = 0.02, TOL...)
                    @test maxdiff(us, ref.u) <= PLEX_SERIAL_TOL
                    @test maxdiff(cs, us) <= PLEX_COLOUR_TOL
                end
            end
            function walk(integ, gather)
                seen = map(1:20) do _
                    step!(integ)
                    gather.((integ.u, integ((integ.tprev + integ.t) / 2), get_du(integ)))
                end
                terminate!(integ)
                return seen
            end
            integ = init(dm_rd(; jac = plex_rd_jac_dm!), bdf(; dm = q.dm); FIXED...)
            ts_dm = held(PETScDiffEq._ts_dm(pl, integ.h.ts))
            @test PETScDiffEq._dm_type(pl, LibPETSc.PetscDM(ts_dm, pl)) == "plex"
            seen = walk(integ, u -> plex_natural(u, q))
            @test everywhere(released(ts_dm))
            if rank == 0
                ref = walk(init(serial_rd(; jac = plex_rd_jac!), bdf(); FIXED...), copy)
                @test maximum(maxdiff(a, b) for (a, b) in zip(seen, ref)) <= PLEX_STEP_TOL
            end
            @test everywhere(refs(q.dm.ptr) == 1)
            PETScCompat.destroy!(q.dm)
        end
    end

    @testset "every handle is freed" begin
        @test isempty(PETScDiffEq.PARALLEL_HANDLES)
        @test all(h -> h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
    end
end

PETScCompat.destroy!(da)
PETScCompat.destroy!(stag)
