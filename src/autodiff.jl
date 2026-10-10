_check_autodiff(ad::ADTypes.AbstractADType) = ad
_check_autodiff(ad) = throw(
    ArgumentError(
        "`autodiff` takes an ADTypes backend, such as `AutoForwardDiff()` or " *
            "`AutoFiniteDiff()`; got $(repr(ad))",
    ),
)

_default_autodiff(comm, dm) =
    dm === nothing && something(comm, MPI.COMM_SELF) == MPI.COMM_SELF ? AutoForwardDiff() :
    AutoFiniteDiff()

_autodiff(alg::Union{TSRosW, TSImplicit, TSIRK, TSDAE, TSARKIMEX, TSGeneric, TSAlpha2}) =
    alg.autodiff
_autodiff(::Union{TSRK, TSMPRK, TSBasicSymplectic}) = nothing

function _petsc_differences(alg)
    ad = _autodiff(alg)
    return ad !== nothing && ADTypes.dense_ad(ad) isa AutoFiniteDiff
end

function _coloring_algorithm(backend)
    coloring = backend isa ADTypes.AutoSparse ? ADTypes.coloring_algorithm(backend) :
        ADTypes.NoColoringAlgorithm()
    coloring isa ADTypes.NoColoringAlgorithm || return coloring
    return SparseMatrixColorings.GreedyColoringAlgorithm()
end

function _with_pattern(backend, proto)
    if proto isa SparseArrays.AbstractSparseMatrix
        return ADTypes.AutoSparse(
            ADTypes.dense_ad(backend);
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(proto),
            coloring_algorithm = _coloring_algorithm(backend),
        )
    end
    backend isa ADTypes.AutoSparse &&
        ADTypes.sparsity_detector(backend) isa ADTypes.NoSparsityDetector &&
        return ADTypes.dense_ad(backend)
    return backend
end

struct Counted{F}
    f::F
    n::Base.RefValue{Int}
end

(c::Counted)(args...) = (c.n[] += 1; c.f(args...))

struct ADJacobian{F, B, P, S}
    f!::F
    backend::B
    prep::P
    du::Vector{S}
    advice::String
end

function (j::ADJacobian)(J, u, p, t)
    try
        DI.jacobian!(j.f!, j.du, J, j.prep, j.backend, u, DI.Constant(p), DI.Constant(t))
    catch e
        _dual_failure(e) && throw(_dual_error(e, j.backend, j.advice))
        rethrow()
    end
    _check_finite(J, t, j.advice) do
        j.f!(j.du, u, p, t)
        return j.du
    end
    return nothing
end

function _ad_jacobian(backend, f!, jac_prototype, u0, p, t, calls, advice)
    b = _with_pattern(backend, jac_prototype)
    g! = Counted(f!, calls)
    du = similar(u0)
    prep = DI.prepare_jacobian(g!, du, b, copy(u0), DI.Constant(p), DI.Constant(t))
    return ADJacobian(g!, b, prep, du, advice)
end

# gamma * dG/du' + dG/du is the Jacobian of v -> G(du + gamma * (v - u), v) at v = u.
function _shifted_residual!(r, v, g!, du, u, gamma, p, t, w)
    @. w = du + gamma * (v - u)
    g!(r, w, v, p, t)
    return nothing
end

struct ADDAEJacobian{G, B, P, S}
    g!::G
    backend::B
    prep::P
    r::Vector{S}
    w::Vector{S}
    advice::String
end

function (j::ADDAEJacobian)(J, du, u, p, gamma, t)
    try
        DI.jacobian!(
            _shifted_residual!, j.r, J, j.prep, j.backend, u, DI.Constant(j.g!),
            DI.Constant(du), DI.Constant(u), DI.Constant(gamma), DI.Constant(p),
            DI.Constant(t), DI.Cache(j.w),
        )
    catch e
        _dual_failure(e) && throw(_dual_error(e, j.backend, j.advice))
        rethrow()
    end
    _check_finite(J, t, j.advice) do
        j.g!(j.r, du, u, p, t)
        return j.r
    end
    return nothing
end

function _ad_dae_jacobian(backend, g!, jac_prototype, u0, p, t, calls, advice)
    b = _with_pattern(backend, jac_prototype)
    h! = Counted(g!, calls)
    r, w = similar(u0), similar(u0)
    prep = DI.prepare_jacobian(
        _shifted_residual!, r, b, copy(u0), DI.Constant(h!), DI.Constant(zero(u0)),
        DI.Constant(copy(u0)), DI.Constant(one(t)), DI.Constant(p), DI.Constant(t),
        DI.Cache(w),
    )
    return ADDAEJacobian(h!, b, prep, r, w, advice)
end

# ForwardDiff takes no complex input, so differentiate x -> [real(f); imag(f)] at
# complex(x, y). For holomorphic f that gives [A; C], and the complex Jacobian is A + iC.
function _split_parts!(out, z)
    n = length(z)
    @views out[1:n] .= real.(z)
    @views out[(n + 1):(2n)] .= imag.(z)
    return nothing
end

function _complex_rhs!(out, x, f!, y, p, t)
    u = complex.(x, y)
    du = similar(u)
    f!(du, u, p, t)
    _split_parts!(out, du)
    return nothing
end

function _complex_residual!(out, x, g!, y, du, u, gamma, p, t)
    v = complex.(x, y)
    r = similar(v)
    g!(r, @.(du + gamma * (v - u)), v, p, t)
    _split_parts!(out, r)
    return nothing
end

function _stacked_pattern(R, proto)
    P = _structure(R, SparseMatrixCSC(proto))
    fill!(P.nzval, one(R))
    return vcat(P, P)
end

function _complex_backend(R, backend, jac_prototype)
    jac_prototype isa SparseArrays.AbstractSparseMatrix &&
        return _with_pattern(backend, _stacked_pattern(R, jac_prototype))
    backend isa ADTypes.AutoSparse || return backend
    detector = ADTypes.sparsity_detector(backend)
    detector isa ADTypes.KnownJacobianSparsityDetector || return ADTypes.dense_ad(backend)
    return _with_pattern(
        backend, _stacked_pattern(R, SparseArrays.sparse(detector.jacobian_sparsity)),
    )
end

# Each column of Jr holds J's rows, then the same rows shifted by n.
function _assemble_complex!(J::SparseMatrixCSC, Jr::SparseMatrixCSC)
    for j in axes(J, 2)
        r, k = nzrange(J, j), first(nzrange(Jr, j))
        m = length(r)
        for (i, q) in enumerate(r)
            J.nzval[q] = complex(Jr.nzval[k + i - 1], Jr.nzval[k + m + i - 1])
        end
    end
    return nothing
end

function _assemble_complex!(J, Jr)
    n = size(J, 1)
    @views J .= complex.(Jr[1:n, :], Jr[(n + 1):(2n), :])
    return nothing
end

struct ADComplexJacobian{F, B, P, R, JR, S}
    f!::F
    backend::B
    prep::P
    x::Vector{R}
    y::Vector{R}
    out::Vector{R}
    Jr::JR
    du::Vector{S}
    advice::String
end

function _real_jacobian!(j::ADComplexJacobian, u, p, t)
    j.x .= real.(u)
    j.y .= imag.(u)
    try
        DI.jacobian!(
            _complex_rhs!, j.out, j.Jr, j.prep, j.backend, j.x, DI.Constant(j.f!),
            DI.Constant(j.y), DI.Constant(p), DI.Constant(t),
        )
    catch e
        _dual_failure(e) && throw(_dual_error(e, j.backend, j.advice))
        rethrow()
    end
    return nothing
end

function (j::ADComplexJacobian)(J, u, p, t)
    _real_jacobian!(j, u, p, t)
    _assemble_complex!(J, j.Jr)
    _check_finite(J, t, j.advice) do
        j.f!(j.du, u, p, t)
        return j.du
    end
    return nothing
end

function _ad_jacobian(
        backend, f!, jac_prototype, u0::AbstractVector{<:Complex}, p, t, calls, advice,
    )
    R, n = real(eltype(u0)), length(u0)
    b = _complex_backend(R, backend, jac_prototype)
    g! = Counted(f!, calls)
    x, y = real.(u0), imag.(u0)
    Jr = jac_prototype isa SparseMatrixCSC ? _stacked_pattern(R, jac_prototype) :
        zeros(R, 2n, n)
    out = zeros(R, 2n)
    prep = DI.prepare_jacobian(
        _complex_rhs!, out, b, copy(x), DI.Constant(g!), DI.Constant(y), DI.Constant(p),
        DI.Constant(t),
    )
    _check_holomorphic(z -> (dz = similar(z); g!(dz, z, p, t); dz), _off(u0), b, advice)
    return ADComplexJacobian(g!, b, prep, x, y, out, Jr, similar(u0), advice)
end

struct ADComplexDAEJacobian{G, B, P, R, JR, S}
    g!::G
    backend::B
    prep::P
    x::Vector{R}
    y::Vector{R}
    out::Vector{R}
    Jr::JR
    r::Vector{S}
    advice::String
end

function _real_jacobian!(j::ADComplexDAEJacobian, du, u, p, gamma, t)
    j.x .= real.(u)
    j.y .= imag.(u)
    try
        DI.jacobian!(
            _complex_residual!, j.out, j.Jr, j.prep, j.backend, j.x, DI.Constant(j.g!),
            DI.Constant(j.y), DI.Constant(du), DI.Constant(u), DI.Constant(gamma),
            DI.Constant(p), DI.Constant(t),
        )
    catch e
        _dual_failure(e) && throw(_dual_error(e, j.backend, j.advice))
        rethrow()
    end
    return nothing
end

function (j::ADComplexDAEJacobian)(J, du, u, p, gamma, t)
    _real_jacobian!(j, du, u, p, gamma, t)
    _assemble_complex!(J, j.Jr)
    _check_finite(J, t, j.advice) do
        j.g!(j.r, du, u, p, t)
        return j.r
    end
    return nothing
end

function _ad_dae_jacobian(
        backend, g!, jac_prototype, u0::AbstractVector{<:Complex}, p, t, calls, advice,
    )
    R, n = real(eltype(u0)), length(u0)
    b = _complex_backend(R, backend, jac_prototype)
    h! = Counted(g!, calls)
    x, y = real.(u0), imag.(u0)
    Jr = jac_prototype isa SparseMatrixCSC ? _stacked_pattern(R, jac_prototype) :
        zeros(R, 2n, n)
    out = zeros(R, 2n)
    prep = DI.prepare_jacobian(
        _complex_residual!, out, b, copy(x), DI.Constant(h!), DI.Constant(y),
        DI.Constant(zero(u0)), DI.Constant(copy(u0)), DI.Constant(one(t)), DI.Constant(p),
        DI.Constant(t),
    )
    v = _off(u0)
    _check_holomorphic(z -> (r = similar(z); h!(r, z .- v, z, p, t); r), v, b, advice)
    return ADComplexDAEJacobian(h!, b, prep, x, y, out, Jr, similar(u0), advice)
end

_direction(R, n) = [one(R) + R(k) / n for k in 1:n]
_off(u0) = (R = real(eltype(u0)); u0 .+ complex(R(0.01), R(0.02)) .* (1 .+ abs.(u0)) .* _direction(R, length(u0)))

_norms(::Nothing, xs...) = map(LinearAlgebra.norm, xs)
_norms(comm::MPI.Comm, xs...) = Tuple(sqrt.(MPI.Allreduce([sum(abs2, x) for x in xs], +, comm)))

function _check_holomorphic(h, v, backend, advice, comm = nothing)
    n = length(v)
    comm === nothing && n == 0 && return nothing
    R = real(eltype(v))
    w = _direction(R, n)
    function along(d)
        g = DI.derivative(ADTypes.dense_ad(backend), zero(R)) do e
            z = h(v .+ e .* d)
            return vcat(real.(z), imag.(z))
        end
        return complex.(g[1:n], g[(n + 1):(2n)])
    end
    along_real, along_imag = along(w), along(im .* w)
    gap, re, ims = _norms(comm, along_imag .- im .* along_real, along_real, along_imag)
    scale = re + ims
    (isfinite(gap) && isfinite(scale)) || return nothing
    gap <= sqrt(eps(R)) * scale && return nothing
    throw(
        ArgumentError(
            "the problem's function is not holomorphic in the complex state: its " *
                "derivative along the imaginary part is not `im` times its derivative along " *
                "the real part, as happens with `conj`, `abs`, `real` or `imag`, so it has " *
                "no complex Jacobian for PETSc's Newton iteration. Write the state as a real " *
                "vector of its real and imaginary parts, or use an explicit method",
        ),
    )
end

struct Guarded{F}
    f::F
    err::Base.RefValue{Any}
end

# A rank whose `f` throws goes on calling it, on NaN, so the other ranks' halo exchanges match.
function (g::Guarded)(out, args...)
    try
        g.f(out, args...)
    catch e
        g.err[] === nothing && (g.err[] = e)
        fill!(out, NaN)
    end
    return nothing
end

_guarded(f, err) = Guarded(f, err)
# Both parts run even when the first throws, since either may communicate.
_guarded(d::Partitioned, err) = Partitioned(Guarded(d.f1, err), Guarded(d.f2, err), d.nv)

function _global_colours(backend, proto::SparseMatrixCSC, comm, colmap)
    n, N = size(proto)
    rstart = MPI.Scan(n, +, comm) - n
    pairs = Vector{Int}(undef, 2 * SparseArrays.nnz(proto))
    for j in 1:N, q in nzrange(proto, j)
        pairs[2q - 1], pairs[2q] = rstart + rowvals(proto)[q], j
    end
    counts = MPI.Allgather(length(pairs), comm)
    root = MPI.Comm_rank(comm) == 0
    whole = root ? Vector{Int}(undef, sum(counts)) : nothing
    MPI.Gatherv!(pairs, root ? MPI.VBuffer(whole, counts) : nothing, comm)
    colours = _checked_everywhere(comm) do
        root || return zeros(Int, N)
        P = sparse(whole[1:2:end], whole[2:2:end], trues(length(whole) ÷ 2), N, N)
        problem = SparseMatrixColorings.ColoringProblem(;
            structure = :nonsymmetric, partition = :column,
        )
        result = SparseMatrixColorings.coloring(P, problem, _coloring_algorithm(backend))
        return Vector{Int}(SparseMatrixColorings.column_colors(result))
    end
    MPI.Bcast!(colours, comm)
    entry = [colours[j] for j in 1:N for _ in nzrange(proto, j)]
    block = rstart .+ (1:n)
    # A partitioned problem's columns are not in the state's order, so find this rank's.
    own = colmap === nothing ? block : invperm(colmap)[block]
    return colours[own], entry, maximum(colours; init = 0)
end

_batch(::AutoForwardDiff{C}, ncolours) where {C} =
    C === nothing ? ForwardDiff.pickchunksize(ncolours) : min(C, ncolours)

# Each rank seeds its own block; `f`'s halo exchange brings the other ranks' partials.
struct CommJacobian{F, G, B, P, R, T, W}
    fun::F
    g!::G
    backend::B
    prep::P
    x::Vector{R}
    y::Vector{R}
    out::Vector{R}
    tx::T
    ty::T
    w::W
    own::Vector{Int}
    entry::Vector{Int}
    ncolours::Int
    err::Base.RefValue{Any}
    advice::String
end

_entry(::Type{<:Real}, d, r, n) = d[r]
_entry(::Type{<:Complex}, d, r, n) = complex(d[r], d[r + n])

function _coloured!(J, j::CommJacobian, t, x, contexts...)
    j.err[] = nothing
    B, n = length(j.tx), size(J, 1)
    rows = rowvals(J)
    for b in 1:(B == 0 ? 0 : cld(j.ncolours, B))
        lo = (b - 1) * B
        for k in 1:B
            j.tx[k] .= j.own .== lo + k
        end
        DI.value_and_pushforward!(j.fun, j.out, j.ty, j.prep, j.backend, x, j.tx, contexts...)
        for q in eachindex(j.entry)
            k = j.entry[q] - lo
            1 <= k <= B && (J.nzval[q] = _entry(eltype(J), j.ty[k], rows[q], n))
        end
    end
    e = j.err[]
    e === nothing || throw(_dual_failure(e) ? _dual_error(e, j.backend, j.advice) : e)
    _check_finite(() -> j.out, J, t, j.advice)
    return nothing
end

(j::CommJacobian)(J, u, p, t) = _coloured!(J, j, t, u, DI.Constant(p), DI.Constant(t))

function (j::CommJacobian)(J, u::AbstractVector{<:Complex}, p, t)
    j.x .= real.(u)
    j.y .= imag.(u)
    return _coloured!(
        J, j, t, j.x, DI.Constant(j.g!), DI.Constant(j.y), DI.Constant(p), DI.Constant(t),
    )
end

(j::CommJacobian)(J, du, u, p, gamma, t) = _coloured!(
    J, j, t, u, DI.Constant(j.g!), DI.Constant(du), DI.Constant(u), DI.Constant(gamma),
    DI.Constant(p), DI.Constant(t), DI.Cache(j.w),
)

function (j::CommJacobian)(J, du, u::AbstractVector{<:Complex}, p, gamma, t)
    j.x .= real.(u)
    j.y .= imag.(u)
    return _coloured!(
        J, j, t, j.x, DI.Constant(j.g!), DI.Constant(j.y), DI.Constant(du), DI.Constant(u),
        DI.Constant(gamma), DI.Constant(p), DI.Constant(t),
    )
end

function _ad_comm_jacobian(
        backend, f!, proto, u0, p, t, calls, advice, comm, dae, colmap = nothing,
    )
    own, entry, ncolours = _global_colours(backend, proto, comm, colmap)
    dense = ADTypes.dense_ad(backend)
    err = Ref{Any}(nothing)
    g! = Counted(_guarded(f!, err), calls)
    R, n = real(eltype(u0)), length(u0)
    split = eltype(u0) <: Complex
    x, y = split ? (real.(u0), imag.(u0)) : (copy(u0), R[])
    out = zeros(R, split ? 2n : n)
    w = similar(u0)
    B = ncolours == 0 ? 0 : _batch(dense, ncolours)
    tx, ty = ntuple(_ -> zeros(R, n), B), ntuple(_ -> zeros(R, length(out)), B)
    du, u, gamma = DI.Constant(zero(u0)), DI.Constant(copy(u0)), DI.Constant(one(t))
    pt = (DI.Constant(p), DI.Constant(t))
    fun, contexts = if split && dae
        _complex_residual!, (DI.Constant(g!), DI.Constant(y), du, u, gamma, pt...)
    elseif split
        _complex_rhs!, (DI.Constant(g!), DI.Constant(y), pt...)
    elseif dae
        _shifted_residual!, (DI.Constant(g!), du, u, gamma, pt..., DI.Cache(w))
    else
        g!, pt
    end
    prepare() = DI.prepare_pushforward(fun, out, dense, copy(x), tx, contexts...)
    prep = B == 0 ? nothing : _checked_everywhere(prepare, comm)
    if split
        v = _off(u0)
        h = dae ? (z -> (r = similar(z); g!(r, z .- v, z, p, t); r)) :
            (z -> (dz = similar(z); g!(dz, z, p, t); dz))
        _checked_everywhere(() -> _check_holomorphic(h, v, dense, advice, comm), comm)
        _checked_everywhere(comm) do
            e, err[] = err[], nothing
            e === nothing || throw(_dual_failure(e) ? _dual_error(e, dense, advice) : e)
        end
    end
    return CommJacobian(
        fun, g!, dense, prep, x, y, out, tx, ty, w, own, entry, ncolours, err, advice,
    )
end

# PETSC_USE_POINTER: the colouring keeps the index sets and frees them with itself.
const _USE_POINTER = Cint(2)

# The colour, counted from 1, of each entry this rank owns, and the number of colours.
function _coloring_colours(pl, coloring, n, rstart)
    nc, sets = Ref{_PetscInt}(0), Ref{Ptr{Ptr{Cvoid}}}(C_NULL)
    _check_code(
        ccall(
            _symbol(pl, :ISColoringGetIS), LibPETSc.PetscErrorCode,
            (Ptr{Cvoid}, Cint, _P, Ptr{Ptr{Ptr{Cvoid}}}), coloring, _USE_POINTER, nc, sets,
        ),
    )
    colours = zeros(Int, n)
    len, idx = Ref{_PetscInt}(0), Ref{_P}(C_NULL)
    for c in 1:nc[]
        is = unsafe_load(sets[], c)
        _check_code(
            ccall(
                _symbol(pl, :ISGetLocalSize), LibPETSc.PetscErrorCode, (Ptr{Cvoid}, _P), is, len,
            ),
        )
        _check_code(
            ccall(
                _symbol(pl, :ISGetIndices), LibPETSc.PetscErrorCode, (Ptr{Cvoid}, Ptr{_P}), is,
                idx,
            ),
        )
        for q in 1:len[]
            colours[unsafe_load(idx[], q) - rstart + 1] = c
        end
        _check_code(
            ccall(
                _symbol(pl, :ISRestoreIndices), LibPETSc.PetscErrorCode, (Ptr{Cvoid}, Ptr{_P}),
                is, idx,
            ),
        )
    end
    _check_code(
        ccall(
            _symbol(pl, :ISColoringRestoreIS), LibPETSc.PetscErrorCode,
            (Ptr{Cvoid}, Cint, Ptr{Ptr{Ptr{Cvoid}}}), coloring, _USE_POINTER, sets,
        ),
    )
    return colours, Int(nc[])
end

# This rank's rows of the DM's matrix, as global columns counted from 0.
function _matrix_rows(pl, A, n, rstart)
    len, ptr = Ref{_PetscInt}(0), Ref{_P}(C_NULL)
    rows = Vector{Vector{_PetscInt}}(undef, n)
    for r in 1:n
        row = rstart + r - 1
        _check_code(
            ccall(
                _symbol(pl, :MatGetRow), LibPETSc.PetscErrorCode,
                (Ptr{Cvoid}, _PetscInt, _P, Ptr{_P}, Ptr{Cvoid}), A.ptr, row, len, ptr, C_NULL,
            ),
        )
        rows[r] = copy(unsafe_wrap(Array, ptr[], len[]))
        _check_code(
            ccall(
                _symbol(pl, :MatRestoreRow), LibPETSc.PetscErrorCode,
                (Ptr{Cvoid}, _PetscInt, _P, Ptr{_P}, Ptr{Cvoid}), A.ptr, row, len, ptr, C_NULL,
            ),
        )
    end
    return rows
end

# This rank's rows of the DM's matrix and the DM's colouring of its entries, if it gives one.
function _dm_pattern(pl, dm, n)
    clone = _clone_dm(pl, dm)
    A = nothing
    try
        A = LibPETSc.DMCreateMatrix(pl, clone)
        rstart = Int(first(LibPETSc.MatGetOwnershipRange(pl, A)))
        cols = _matrix_rows(pl, A, n, rstart)
        own, ncolours = zeros(Int, n), 0
        _dm_colours(pl, clone.ptr) do coloring
            own, ncolours = _coloring_colours(pl, coloring, n, rstart)
        end
        return rstart, cols, own, ncolours
    finally
        A === nothing || PETScCompat.destroy!(A)
        _check_code(
            ccall(
                _symbol(pl, :DMDestroy), LibPETSc.PetscErrorCode, (Ptr{Ptr{Cvoid}},),
                Ref(clone.ptr),
            ),
        )
    end
end

# The global index of each entry of the DM's ghosted array, negative where it has none.
function _dm_globals(pl, dm)
    ltog = _dm_vec!(pl, :DMGetLocalToGlobalMapping, dm)[]
    n, idx = Ref{_PetscInt}(0), Ref{_P}(C_NULL)
    _check_code(
        ccall(
            _symbol(pl, :ISLocalToGlobalMappingGetSize), LibPETSc.PetscErrorCode,
            (Ptr{Cvoid}, _P), ltog, n,
        ),
    )
    _check_code(
        ccall(
            _symbol(pl, :ISLocalToGlobalMappingGetIndices), LibPETSc.PetscErrorCode,
            (Ptr{Cvoid}, Ptr{_P}), ltog, idx,
        ),
    )
    globals = copy(unsafe_wrap(Array, idx[], n[]))
    _check_code(
        ccall(
            _symbol(pl, :ISLocalToGlobalMappingRestoreIndices), LibPETSc.PetscErrorCode,
            (Ptr{Cvoid}, Ptr{_P}), ltog, idx,
        ),
    )
    return globals
end

# `f` runs on the ghosted array with each entry seeded by its owner's colour, so the
# partials of its output are this rank's rows, one column of each row per colour.
struct DMJacobian{F, G, B, P, R, T}
    fun::F
    ghosts::G
    backend::B
    prep::P
    x::Vector{R}
    out::Vector{R}
    tx::T
    ty::T
    seed::Vector{Int}
    rstart::Int
    row::Vector{_PetscInt}
    cols::Vector{Vector{_PetscInt}}
    colour::Vector{Vector{Int}}
    vals::Vector{Vector{R}}
    ncolours::Int
    advice::String
end

function (j::DMJacobian)(J, u, p, t)
    _ghosted(a -> copyto!(j.x, a), j.ghosts, u)
    B = length(j.tx)
    try
        for b in 1:(B == 0 ? 0 : cld(j.ncolours, B))
            lo = (b - 1) * B
            for k in 1:B
                j.tx[k] .= j.seed .== lo + k
            end
            DI.value_and_pushforward!(
                j.fun, j.out, j.ty, j.prep, j.backend, j.x, j.tx, DI.Constant(p), DI.Constant(t),
            )
            for r in eachindex(j.vals), q in eachindex(j.vals[r])
                k = j.colour[r][q] - lo
                1 <= k <= B && (j.vals[r][q] = j.ty[k][r])
            end
        end
    catch e
        _dual_failure(e) && throw(_dual_error(e, j.backend, j.advice))
        rethrow()
    end
    pl = j.ghosts.petsclib
    for r in eachindex(j.vals)
        _check_finite(() -> j.out, j.vals[r], t, j.advice)
        j.row[1] = j.rstart + r - 1
        _mat_set_values!(pl, J, j.row, j.cols[r], j.vals[r], LibPETSc.INSERT_VALUES)
    end
    return nothing
end

# Each ghosted entry's colour, as the DM's scatter brings it, and each stored column's.
function _spread_colours(ghosts, globals, cols, own, R)
    seed = Int[]
    _ghosted(a -> append!(seed, round.(Int, a)), ghosts, R.(own))
    of = Dict{_PetscInt, Int}()
    for (g, c) in zip(globals, seed)
        g >= 0 && c > 0 && (of[g] = c)
    end
    return seed, Vector{Int}[[get(of, c, 0) for c in row] for row in cols]
end

_proper(colour) = all(row -> allunique(c for c in row if c > 0), colour)

function _ad_dm_jacobian(backend, f!, pl, dm, u0, p, t, calls, advice, comm, N)
    dense = ADTypes.dense_ad(backend)
    R, n = eltype(u0), length(u0)
    ghosts = Ghosted(nothing, pl, dm)
    rstart, cols, own, ncolours = _checked_everywhere(() -> _dm_pattern(pl, dm, n), comm)
    globals = _dm_globals(pl, dm)
    seed, colour = _spread_colours(ghosts, globals, cols, own, R)
    # PETSc's DMDA colouring fails or is wrong on some periodic grids; colour the pattern then.
    if !_everywhere(comm, ncolours > 0 && _proper(colour))
        proto = sparse(
            [r for r in 1:n for _ in cols[r]], [Int(c) + 1 for row in cols for c in row],
            true, n, N,
        )
        own, _, ncolours = _global_colours(backend, proto, something(comm, MPI.COMM_SELF))
        seed, colour = _spread_colours(ghosts, globals, cols, own, R)
    end
    g! = Counted(f!, calls)
    x, out = zeros(R, length(seed)), zeros(R, n)
    _ghosted(a -> copyto!(x, a), ghosts, u0)
    B = ncolours == 0 ? 0 : _batch(dense, ncolours)
    tx, ty = ntuple(_ -> zeros(R, length(x)), B), ntuple(_ -> zeros(R, n), B)
    prepare() =
        DI.prepare_pushforward(g!, out, dense, copy(x), tx, DI.Constant(p), DI.Constant(t))
    prep = B == 0 ? nothing : _checked_everywhere(prepare, comm)
    vals = Vector{R}[zeros(R, length(row)) for row in cols]
    return DMJacobian(
        g!, ghosts, dense, prep, x, out, tx, ty, seed, rstart, _PetscInt[0], cols, colour,
        vals, ncolours, advice,
    )
end

struct ADParamJacobian{F, B, P}
    f!::F
    backend::B
    prep::P
    du::Vector{Float64}
    advice::String
end

_with_p!(du, p, f!, u, t) = f!(du, u, p, t)

function (j::ADParamJacobian)(pJ, u, p, t)
    try
        DI.jacobian!(
            _with_p!, j.du, pJ, j.prep, j.backend, p, DI.Constant(j.f!), DI.Constant(u),
            DI.Constant(t),
        )
    catch e
        _dual_failure(e) && throw(_dual_error(e, j.backend, j.advice))
        rethrow()
    end
    _check_finite(pJ, t, j.advice) do
        j.f!(j.du, u, p, t)
        return j.du
    end
    return nothing
end

_param_backend(b, np) = b
_param_backend(b::AutoForwardDiff{C}, np) where {C} =
    C === nothing || C <= np ? b : AutoForwardDiff(; tag = b.tag)

function _ad_paramjacobian(backend, f!, u0, p, t, advice)
    b = _param_backend(ADTypes.dense_ad(backend), length(p))
    du = similar(u0)
    prep = DI.prepare_jacobian(
        _with_p!, du, b, p, DI.Constant(f!), DI.Constant(copy(u0)), DI.Constant(t),
    )
    return ADParamJacobian(f!, b, prep, du, advice)
end

# Every rank seeds the same parameters, so each calls `f` once per chunk, on its own rows.
struct CommParamJacobian{G, B, P, X, Y}
    g!::G
    backend::B
    prep::P
    out::Vector{Float64}
    tx::X
    ty::Y
    err::Base.RefValue{Any}
    advice::String
end

function (j::CommParamJacobian)(pJ, u, p, t)
    j.err[] = nothing
    B, np = length(j.tx), size(pJ, 2)
    for lo in 0:B:(np - 1)
        for k in 1:B
            j.tx[k] .= (1:np) .== lo + k
        end
        DI.value_and_pushforward!(
            _with_p!, j.out, j.ty, j.prep, j.backend, p, j.tx, DI.Constant(j.g!),
            DI.Constant(u), DI.Constant(t),
        )
        for k in 1:min(B, np - lo)
            pJ[:, lo + k] .= j.ty[k]
        end
    end
    e = j.err[]
    e === nothing || throw(_dual_failure(e) ? _dual_error(e, j.backend, j.advice) : e)
    _check_finite(() -> j.out, pJ, t, j.advice)
    return nothing
end

function _ad_comm_paramjacobian(backend, f!, u0, p, t, advice, comm)
    dense = ADTypes.dense_ad(backend)
    err = Ref{Any}(nothing)
    g! = _guarded(f!, err)
    out = zeros(length(u0))
    B = _batch(dense, length(p))
    tx = ntuple(_ -> zeros(eltype(p), length(p)), B)
    ty = ntuple(_ -> zeros(length(u0)), B)
    contexts = (DI.Constant(g!), DI.Constant(copy(u0)), DI.Constant(t))
    prep = _checked_everywhere(comm) do
        DI.prepare_pushforward(_with_p!, out, dense, p, tx, contexts...)
    end
    return CommParamJacobian(g!, dense, prep, out, tx, ty, err, advice)
end

struct ADCostGradient{G, B, P, W}
    g::G
    backend::B
    prep::P
    wrt::W
end

_g_of_u(u, g, p, t) = g(u, p, t)
_g_of_p(p, g, u, t) = g(u, p, t)

function (c::ADCostGradient)(out, u, p, t)
    x, other = c.wrt === _g_of_u ? (u, p) : (p, u)
    try
        DI.gradient!(
            c.wrt, out, c.prep, c.backend, x, DI.Constant(c.g), DI.Constant(other),
            DI.Constant(t),
        )
    catch e
        _dual_failure(e) && throw(_dual_error(e, c.backend, _ADJOINT_COST_ADVICE))
        rethrow()
    end
    all(isfinite, out) || !isfinite(c.g(u, p, t)) || throw(
        ArgumentError(
            "the derivative of the integral cost `g` from automatic differentiation has a " *
                "non-finite entry at t = $t where `g` itself is finite, as the derivative " *
                "of `sqrt` or `norm` at zero is. " * _ADJOINT_COST_ADVICE,
        ),
    )
    return nothing
end

function _ad_cost_gradient(backend, g, u0, p, t, wrt)
    x, other = wrt === _g_of_u ? (copy(u0), p) : (p, copy(u0))
    b = _param_backend(ADTypes.dense_ad(backend), length(x))
    prep = DI.prepare_gradient(wrt, b, x, DI.Constant(g), DI.Constant(other), DI.Constant(t))
    return ADCostGradient(g, b, prep, wrt)
end

_has_dual(x) = _is_dual(x) || (x isa Type && _is_dual_type(x)) ||
    (x isa AbstractArray && _is_dual_type(eltype(x)))
_is_dual(x) = x isa Union{ForwardDiff.Dual, Complex{<:ForwardDiff.Dual}}
_is_dual_type(T) = T <: Union{ForwardDiff.Dual, Complex{<:ForwardDiff.Dual}}

function _dual_failure(e)
    e isa MethodError && any(_has_dual, e.args) && return true
    e isa TypeError && _has_dual(e.got) && return true
    return occursin("ForwardDiff.Dual", sprint(showerror, e))
end

_dual_error(e, backend, advice) = ArgumentError(
    "`$(ADTypes.dense_ad(backend))` could not differentiate the problem's function: " *
        first(split(sprint(showerror, e), '\n')) * ". " * advice,
)

_stored_values(J::SparseArrays.AbstractSparseMatrix) = SparseArrays.nonzeros(J)
_stored_values(J) = J

function _check_finite(value, J, t, advice)
    all(isfinite, _stored_values(J)) && return nothing
    all(isfinite, value()) || return nothing
    throw(
        ArgumentError(
            "the Jacobian from automatic differentiation has a non-finite entry at " *
                "t = $t where the function itself is finite, as the derivative of `sqrt` " *
                "or `norm` at zero is. " * advice,
        ),
    )
end

const _ODE_ADVICE = "Give the ODEFunction a `jac`, or pass " *
    "`autodiff = PETScDiffEq.AutoFiniteDiff()` to the algorithm to have PETSc difference it"
const _DAE_ADVICE = "Give the DAEFunction a `jac`, or pass " *
    "`autodiff = PETScDiffEq.AutoFiniteDiff()` to the algorithm to have PETSc difference it"
const _ADJOINT_JAC_ADVICE = "PETScAdjoint has no finite-difference fallback, so give the " *
    "ODEFunction a `jac`"
const _ADJOINT_PARAMJAC_ADVICE = "PETScAdjoint has no finite-difference fallback, so give " *
    "the ODEFunction a `paramjac`"
const _ADJOINT_COST_ADVICE = "PETScAdjoint has no finite-difference fallback, so give " *
    "`dgdu_continuous` and `dgdp_continuous`"
