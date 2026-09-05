module PETScDiffEq

using DiffEqBase: DiffEqBase
using LinearAlgebra: LinearAlgebra, mul!
using MPI: MPI
using PETSc: PETSc
using PETSc.LibPETSc: LibPETSc
using SciMLBase: SciMLBase
using SparseArrays: SparseArrays, SparseMatrixCSC, findnz, sparse

export TSRK, TSRosW, TSImplicit, TSARKIMEX, TSGeneric, PETScIntegrator

abstract type PETScTSAlgorithm <: SciMLBase.AbstractODEAlgorithm end

struct TSRK <: PETScTSAlgorithm
    subtype::String
    petsc_options::Vector{String}
end

TSRK(subtype::AbstractString = "5dp", petsc_options::AbstractVector{<:AbstractString} = String[]) =
    TSRK(String(subtype), String[String(o) for o in petsc_options])

struct TSRosW <: PETScTSAlgorithm
    subtype::String
    petsc_options::Vector{String}
end

TSRosW(subtype::AbstractString = "ra34pw2", petsc_options::AbstractVector{<:AbstractString} = String[]) =
    TSRosW(String(subtype), String[String(o) for o in petsc_options])

struct TSImplicit <: PETScTSAlgorithm
    subtype::String
    theta::Union{Nothing, Float64}
    petsc_options::Vector{String}
end

TSImplicit(subtype::AbstractString = "beuler") =
    TSImplicit(String(subtype), nothing, String[])
TSImplicit(subtype::AbstractString, theta::Real) =
    TSImplicit(String(subtype), Float64(theta), String[])
TSImplicit(subtype::AbstractString, petsc_options::AbstractVector{<:AbstractString}) =
    TSImplicit(String(subtype), nothing, String[String(o) for o in petsc_options])
TSImplicit(subtype::AbstractString, theta::Real, petsc_options::AbstractVector{<:AbstractString}) =
    TSImplicit(String(subtype), Float64(theta), String[String(o) for o in petsc_options])

struct TSARKIMEX <: PETScTSAlgorithm
    subtype::String
    petsc_options::Vector{String}
end

TSARKIMEX(subtype::AbstractString = "3", petsc_options::AbstractVector{<:AbstractString} = String[]) =
    TSARKIMEX(String(subtype), String[String(o) for o in petsc_options])

struct TSGeneric <: PETScTSAlgorithm
    ts_type::String
    explicit::Bool
    petsc_options::Vector{String}
end

TSGeneric(
    ts_type::AbstractString,
    petsc_options::AbstractVector{<:AbstractString} = String[];
    explicit::Bool = false,
) = TSGeneric(String(ts_type), explicit, String[String(o) for o in petsc_options])

_uses_ifunction(::TSRK) = false
_uses_ifunction(::TSRosW) = true
_uses_ifunction(::TSImplicit) = true
_uses_ifunction(::TSARKIMEX) = true
_uses_ifunction(alg::TSGeneric) = !alg.explicit

# Only these PETSc TS families carry an embedded error estimate. The rest step
# at the requested dt and ignore any tolerance. `nothing` means "not known",
# which is the honest answer for an arbitrary TSGeneric type.
_adapts(::TSRK) = true
_adapts(::TSRosW) = true
_adapts(::TSARKIMEX) = true
_adapts(alg::TSImplicit) = alg.subtype == "bdf"
_adapts(::TSGeneric) = nothing

mutable struct TSContext{F, F2, JAC, JBUF, P, T, V}
    petsclib::T
    f!::F
    f2!::F2
    jac!::JAC
    p::P
    du::Vector{Float64}
    u::Vector{Float64}
    mudot::Vector{Float64}
    M::Union{Nothing, Matrix{Float64}}
    missing_diag::Vector{Int}
    W::Matrix{Float64}
    idx0::Vector{LibPETSc.PetscInt}
    row_cols0::Vector{Vector{LibPETSc.PetscInt}}
    row_src::Vector{Vector{Int}}
    row_buf::Vector{Vector{Float64}}
    J::JBUF
    ts::Vector{Float64}
    us::Vector{Vector{Float64}}
    dus::Vector{Vector{Float64}}
    saveat::Vector{Float64}
    saveat_idx::Int
    save_everystep::Bool
    dense::Bool
    work::V
    nf::Int
    njacs::Int
    err::Union{Nothing, Any}
end

function _record!(ctx, t, x)
    u = Vector{Float64}(x)
    push!(ctx.ts, Float64(t))
    push!(ctx.us, u)
    ctx.dense && push!(ctx.dus, _derivative(ctx, Float64(t), u))
    return nothing
end

function _derivative(ctx, t, u)
    du = similar(u)
    ctx.f!(du, u, ctx.p, t)
    if ctx.f2! !== nothing
        ctx.f2!(ctx.du, u, ctx.p, t)
        du .+= ctx.du
    end
    ctx.nf += ctx.f2! === nothing ? 1 : 2
    return du
end

_interp(ctx) = ctx.dense ? SciMLBase.HermiteInterpolation(ctx.ts, ctx.us, ctx.dus) :
    SciMLBase.LinearInterpolation(ctx.ts, ctx.us)

_mass(ctx, i, j) = ctx.M === nothing ? (i == j ? 1.0 : 0.0) : ctx.M[i, j]

function _fill_rows!(ctx, shift, n)
    @inbounds for i in 1:n
        cols = ctx.row_cols0[i]
        src = ctx.row_src[i]
        buf = ctx.row_buf[i]
        for k in eachindex(cols)
            j = Int(cols[k]) + 1
            jv = src[k] == 0 ? 0.0 : ctx.J.nzval[src[k]]
            buf[k] = shift * _mass(ctx, i, j) - jv
        end
    end
    return nothing
end

function _setrows!(ctx, A, n)
    one_ = LibPETSc.PetscInt(1)
    @inbounds for i in 1:n
        cols = ctx.row_cols0[i]
        isempty(cols) && continue
        LibPETSc.MatSetValues(
            ctx.petsclib, A, one_, LibPETSc.PetscInt[i - 1],
            LibPETSc.PetscInt(length(cols)), cols, ctx.row_buf[i],
            LibPETSc.INSERT_VALUES,
        )
    end
    return nothing
end

function _row_structure(J::SparseMatrixCSC, n)
    cols = [Int[] for _ in 1:n]
    src = [Int[] for _ in 1:n]
    for j in 1:n, k in J.colptr[j]:(J.colptr[j + 1] - 1)
        i = J.rowval[k]
        push!(cols[i], j)
        push!(src[i], k)
    end
    for i in 1:n
        if !(i in cols[i])
            push!(cols[i], i)
            push!(src[i], 0)
        end
        perm = sortperm(cols[i])
        cols[i] = cols[i][perm]
        src[i] = src[i][perm]
    end
    cols0 = [LibPETSc.PetscInt[c - 1 for c in cols[i]] for i in 1:n]
    buf = [zeros(length(cols[i])) for i in 1:n]
    return cols0, src, buf
end

function _setblock!(ctx, A, n)
    return LibPETSc.MatSetValues(
        ctx.petsclib, A, LibPETSc.PetscInt(n), ctx.idx0,
        LibPETSc.PetscInt(n), ctx.idx0, vec(ctx.W), LibPETSc.INSERT_VALUES,
    )
end

_stored(A::SparseMatrixCSC, i, j) = any(==(i), @view A.rowval[A.colptr[j]:(A.colptr[j + 1] - 1)])

function _rhs!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    f = PETSc.VecPtr(ctx.petsclib, f_ptr, false)
    try
        PETSc.withlocalarray!((x, f); read = (true, false), write = (false, true)) do xa, fa
            copyto!(ctx.u, xa)
            ctx.f!(ctx.du, ctx.u, ctx.p, t)
            copyto!(fa, ctx.du)
        end
        ctx.nf += 1
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(1)
    end
    return LibPETSc.PetscErrorCode(0)
end

const RHS_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _split_rhs!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    f = PETSc.VecPtr(ctx.petsclib, f_ptr, false)
    try
        PETSc.withlocalarray!((x, f); read = (true, false), write = (false, true)) do xa, fa
            copyto!(ctx.u, xa)
            ctx.f2!(ctx.du, ctx.u, ctx.p, t)
            copyto!(fa, ctx.du)
        end
        ctx.nf += 1
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(1)
    end
    return LibPETSc.PetscErrorCode(0)
end

const SPLIT_RHS_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _ifunction!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        xdot_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    xdot = PETSc.VecPtr(ctx.petsclib, xdot_ptr, false)
    f = PETSc.VecPtr(ctx.petsclib, f_ptr, false)
    try
        PETSc.withlocalarray!(
            (x, xdot, f); read = (true, true, false), write = (false, false, true),
        ) do xa, xda, fa
            copyto!(ctx.u, xa)
            ctx.f!(ctx.du, ctx.u, ctx.p, t)
            if ctx.M === nothing
                @. fa = xda - ctx.du
            else
                mul!(ctx.mudot, ctx.M, xda)
                @. fa = ctx.mudot - ctx.du
            end
        end
        ctx.nf += 1
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(1)
    end
    return LibPETSc.PetscErrorCode(0)
end

const IFUNCTION_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _ijacobian!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        ::LibPETSc.CVec,
        shift::LibPETSc.PetscReal,
        A_ptr::LibPETSc.CMat,
        B_ptr::LibPETSc.CMat,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    A = LibPETSc.PetscMat(A_ptr, ctx.petsclib)
    B = LibPETSc.PetscMat(B_ptr, ctx.petsclib)
    try
        PETSc.withlocalarray!(x; read = true, write = false) do xa
            copyto!(ctx.u, xa)
        end
        ctx.jac!(ctx.J, ctx.u, ctx.p, t)
        ctx.njacs += 1
        n = length(ctx.u)
        # One batched MatSetValues beats n^2 single-entry ccalls by orders of
        # magnitude. PETSc reads the block row-major, so entry (i,j) is stored
        # at W[j,i] and `vec` then yields the order PETSc wants.
        @inbounds for j in 1:n, i in 1:n
            ctx.W[j, i] = shift * _mass(ctx, i, j) - ctx.J[i, j]
        end
        _setblock!(ctx, A, n)
        PETSc.assemble!(A)
        if B.ptr != A.ptr
            _setblock!(ctx, B, n)
            PETSc.assemble!(B)
        end
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(1)
    end
    return LibPETSc.PetscErrorCode(0)
end

const IJACOBIAN_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _sparse_ijacobian!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        ::LibPETSc.CVec,
        shift::LibPETSc.PetscReal,
        A_ptr::LibPETSc.CMat,
        B_ptr::LibPETSc.CMat,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    A = LibPETSc.PetscMat(A_ptr, ctx.petsclib)
    B = LibPETSc.PetscMat(B_ptr, ctx.petsclib)
    try
        PETSc.withlocalarray!(x; read = true, write = false) do xa
            copyto!(ctx.u, xa)
        end
        ctx.jac!(ctx.J, ctx.u, ctx.p, t)
        ctx.njacs += 1
        n = length(ctx.u)
        # One MatSetValues per row rather than one per stored entry. The row
        # structure is the prototype's pattern unioned with the diagonal, which
        # is what was preallocated, and it never changes.
        _fill_rows!(ctx, shift, n)
        _setrows!(ctx, A, n)
        PETSc.assemble!(A)
        if B.ptr != A.ptr
            _setrows!(ctx, B, n)
            PETSc.assemble!(B)
        end
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(1)
    end
    return LibPETSc.PetscErrorCode(0)
end

const SPARSE_IJACOBIAN_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _monitor!(
        ts_ptr::LibPETSc.CTS,
        step::LibPETSc.PetscInt,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    try
        if isempty(ctx.saveat)
            if ctx.save_everystep || step == 0
                PETSc.withlocalarray!(xa -> _record!(ctx, t, xa), x; read = true, write = false)
            end
        else
            ts = LibPETSc.TS(ts_ptr, ctx.petsclib)
            tol = 100 * eps(max(one(Float64), abs(Float64(t))))
            while ctx.saveat_idx <= length(ctx.saveat) &&
                    ctx.saveat[ctx.saveat_idx] <= Float64(t) + tol
                want = ctx.saveat[ctx.saveat_idx]
                # Before any step has been taken there is nothing to interpolate
                # from, and the incoming vector is already the initial state.
                if step == 0 || abs(want - Float64(t)) <= tol
                    PETSc.withlocalarray!(
                        xa -> _record!(ctx, want, xa), x; read = true, write = false,
                    )
                else
                    LibPETSc.TSInterpolate(ctx.petsclib, ts, want, ctx.work)
                    PETSc.withlocalarray!(
                        wa -> _record!(ctx, want, wa), ctx.work; read = true, write = false,
                    )
                end
                ctx.saveat_idx += 1
            end
        end
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(1)
    end
    return LibPETSc.PetscErrorCode(0)
end

const MONITOR_PTR = Ref{Ptr{Cvoid}}(C_NULL)

# `@cfunction` pointers do not survive precompilation, so they are built at load time.
function __init__()
    RHS_PTR[] = @cfunction(
        _rhs!,
        LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec, Ptr{Cvoid})
    )
    SPLIT_RHS_PTR[] = @cfunction(
        _split_rhs!,
        LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec, Ptr{Cvoid})
    )
    MONITOR_PTR[] = @cfunction(
        _monitor!,
        LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, LibPETSc.PetscInt, LibPETSc.PetscReal, LibPETSc.CVec, Ptr{Cvoid})
    )
    IFUNCTION_PTR[] = @cfunction(
        _ifunction!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec,
            LibPETSc.CVec, Ptr{Cvoid},
        )
    )
    IJACOBIAN_PTR[] = @cfunction(
        _ijacobian!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec,
            LibPETSc.PetscReal, LibPETSc.CMat, LibPETSc.CMat, Ptr{Cvoid},
        )
    )
    SPARSE_IJACOBIAN_PTR[] = @cfunction(
        _sparse_ijacobian!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec,
            LibPETSc.PetscReal, LibPETSc.CMat, LibPETSc.CMat, Ptr{Cvoid},
        )
    )
    return nothing
end

function _jacobian_pattern(jac_prototype::SparseMatrixCSC, n::Integer)
    rows, cols, _ = findnz(jac_prototype)
    all_rows = vcat(rows, 1:n)
    all_cols = vcat(cols, 1:n)
    return sparse(all_rows, all_cols, ones(length(all_rows)), n, n)
end

function _cstr(f::F, s::AbstractString) where {F}
    str = String(s)
    return GC.@preserve str f(Base.unsafe_convert(Ptr{Cchar}, str))
end

_ts_type(::TSRK) = "rk"
_ts_type(::TSRosW) = "rosw"
_ts_type(alg::TSImplicit) = alg.subtype
_ts_type(::TSARKIMEX) = "arkimex"
_ts_type(alg::TSGeneric) = alg.ts_type

_set_subtype!(petsclib, ts, alg::TSRK) =
    _cstr(p -> LibPETSc.TSRKSetType(petsclib, ts, p), alg.subtype)
_set_subtype!(petsclib, ts, alg::TSRosW) =
    _cstr(p -> LibPETSc.TSRosWSetType(petsclib, ts, p), alg.subtype)
function _set_subtype!(petsclib, ts, alg::TSImplicit)
    if alg.subtype == "theta" && alg.theta !== nothing
        LibPETSc.TSThetaSetTheta(petsclib, ts, alg.theta)
    end
    return nothing
end
_set_subtype!(petsclib, ts, alg::TSARKIMEX) =
    _cstr(p -> LibPETSc.TSARKIMEXSetType(petsclib, ts, p), alg.subtype)
_set_subtype!(petsclib, ts, ::TSGeneric) = nothing

const UNSUPPORTED_KWARGS = (
    :tstops, :save_idxs, :d_discontinuities, :isoutofdomain,
    :unstable_check, :internalnorm, :calck, :force_dtmin, :alias_u0, :sensealg,
    :controller, :qmax, :qmin, :gamma, :beta1, :beta2,
)

mutable struct TSHandles{CTX, T}
    ctx::CTX
    petsclib::T
    ts::Any
    u::Any
    jac_mat::Any
    opts::Any
    t0::Float64
    tf::Float64
    u0::Vector{Float64}
    maxiters::Int
    save_start::Bool
    save_end::Bool
    destroyed::Bool
end

function _destroy!(h::TSHandles)
    h.destroyed && return nothing
    h.destroyed = true
    # Once PETSc has finalized (at process exit) its objects are already gone
    # and calling into it reaches MPI after MPI has shut down, so an integrator
    # collected late must not try to free anything.
    (PETSc.finalized(h.petsclib) || MPI.Finalized()) && return nothing
    h.opts === nothing || PETSc.destroy(h.opts)
    h.jac_mat === nothing || PETSc.destroy(h.jac_mat)
    h.ctx.work.ptr == C_NULL || PETSc.destroy(h.ctx.work)
    h.u === nothing || PETSc.destroy(h.u)
    h.ts === nothing || LibPETSc.TSDestroy(h.petsclib, h.ts)
    return nothing
end

function _setup(
        prob::SciMLBase.AbstractODEProblem,
        alg::PETScTSAlgorithm;
        dt = nothing,
        reltol = nothing,
        abstol = nothing,
        maxiters = 1000000,
        adaptive = true,
        dtmin = nothing,
        dtmax = nothing,
        saveat = Float64[],
        save_everystep = true,
        save_start = true,
        save_end = true,
        dense = nothing,
        kwargs...,
    )
    for key in UNSUPPORTED_KWARGS
        if haskey(kwargs, key)
            @warn "PETScDiffEq does not support `$key` and is ignoring it"
        end
    end
    SciMLBase.isinplace(prob) ||
        throw(ArgumentError("PETScDiffEq requires an in-place ODEProblem, f!(du, u, p, t)"))
    prob.u0 isa AbstractVector{<:Real} ||
        throw(ArgumentError("PETScDiffEq requires a real AbstractVector u0"))
    dt === nothing && throw(ArgumentError("PETScDiffEq requires an initial dt"))
    is_split = prob.f isa SciMLBase.SplitFunction
    is_split && !(alg isa TSARKIMEX) &&
        throw(ArgumentError("PETScDiffEq only supports SplitODEProblem with TSARKIMEX"))
    mass_matrix = prob.f.mass_matrix
    has_mass = !(mass_matrix === nothing || mass_matrix == LinearAlgebra.I)
    if has_mass && !_uses_ifunction(alg)
        throw(
            ArgumentError(
                "PETScDiffEq cannot apply a mass matrix with an explicit algorithm; " *
                    "use an implicit one such as TSImplicit or TSRosW",
            ),
        )
    end
    if has_mass && is_split
        throw(ArgumentError("PETScDiffEq does not support a mass matrix on a SplitODEProblem"))
    end

    t0, tf = Float64(prob.tspan[1]), Float64(prob.tspan[2])
    t0 < tf || throw(ArgumentError("PETScDiffEq requires tspan[1] < tspan[2]"))

    u0 = Vector{Float64}(vec(prob.u0))
    n = length(u0)

    petsclib = PETSc.getlib(PetscScalar = Float64)
    PETSc.initialized(petsclib) || PETSc.initialize(petsclib)

    f1 = is_split ? prob.f.f1.f : prob.f.f
    f2 = is_split ? prob.f.f2.f : nothing
    for g in (f1, f2)
        # An AbstractSciMLOperator ignores the f(du,u,p,t) call this package
        # makes, leaving the derivative buffer untouched rather than erroring.
        g isa SciMLBase.AbstractSciMLOperator && throw(
            ArgumentError(
                "PETScDiffEq does not support an operator-valued right-hand side; " *
                    "supply a function f!(du, u, p, t)",
            ),
        )
    end
    has_jac = _uses_ifunction(alg) && prob.f.jac !== nothing
    jac_fn = has_jac ? prob.f.jac : nothing
    jac_prototype = has_jac ? prob.f.jac_prototype : nothing
    uses_sparse_jac = jac_prototype isa SparseMatrixCSC
    J0 = if !has_jac
        zeros(0, 0)
    elseif uses_sparse_jac
        SparseMatrixCSC{Float64, Int}(jac_prototype)
    else
        zeros(n, n)
    end
    saveat_times = saveat isa Number ?
        collect(Float64, t0:Float64(saveat):tf) : sort!(Vector{Float64}(collect(saveat)))
    filter!(t -> t0 - eps(tf) <= t <= tf + eps(tf), saveat_times)
    if !isempty(saveat_times) && save_start &&
            saveat_times[1] > t0 + 100 * eps(max(one(Float64), abs(tf)))
        pushfirst!(saveat_times, t0)
    end
    M = has_mass ? Matrix{Float64}(mass_matrix) : nothing
    if M !== nothing && uses_sparse_jac &&
            any(M[i, j] != 0 for i in 1:n, j in 1:n if i != j)
        throw(
            ArgumentError(
                "PETScDiffEq supports a jac_prototype only with a diagonal mass " *
                    "matrix; the off-diagonal entries have no preallocated slot",
            ),
        )
    end
    missing_diag = uses_sparse_jac ?
        [i for i in 1:n if !_stored(J0, i, i)] : Int[]
    W0 = has_jac && !uses_sparse_jac ? zeros(n, n) : zeros(0, 0)
    idx0 = has_jac && !uses_sparse_jac ?
        LibPETSc.PetscInt[i - 1 for i in 1:n] : LibPETSc.PetscInt[]
    row_cols0, row_src, row_buf = uses_sparse_jac ? _row_structure(J0, n) :
        (Vector{LibPETSc.PetscInt}[], Vector{Int}[], Vector{Float64}[])
    dense_out = dense === nothing ? (save_everystep && isempty(saveat_times) && !has_mass) :
        Bool(dense)
    if dense_out && has_mass
        throw(
            ArgumentError(
                "PETScDiffEq cannot produce dense output with a mass matrix; " *
                    "the saved derivative would have to be M \\ f(u)",
            ),
        )
    end
    ctx = TSContext(
        petsclib, f1, f2, jac_fn, prob.p,
        similar(u0), similar(u0), similar(u0), M, missing_diag, W0, idx0,
        row_cols0, row_src, row_buf, J0,
        Float64[], Vector{Float64}[], Vector{Float64}[],
        saveat_times, 1, save_everystep, dense_out,
        PETSc.VecSeq(petsclib, n), 0, 0, nothing,
    )
    h = TSHandles(
        ctx, petsclib, nothing, nothing, nothing, nothing,
        t0, tf, u0, Int(maxiters), save_start, save_end, false,
    )
    finalizer(_destroy!, h)

    try
        h.ts = LibPETSc.TSCreate(petsclib, MPI.COMM_SELF)
        ts = h.ts
        LibPETSc.TSSetProblemType(petsclib, ts, LibPETSc.TS_NONLINEAR)
        LibPETSc.TSSetType(petsclib, ts, _ts_type(alg))
        _set_subtype!(petsclib, ts, alg)

        h.u = PETSc.VecSeq(petsclib, n)
        u = h.u
        PETSc.withlocalarray!(u; read = false, write = true) do ua
            copyto!(ua, u0)
        end
        LibPETSc.TSSetSolution(petsclib, ts, u)

        ctxptr = pointer_from_objref(ctx)
        GC.@preserve ctx begin
            if _uses_ifunction(alg)
                LibPETSc.TSSetIFunction(petsclib, ts, nothing, IFUNCTION_PTR[], ctxptr)
            else
                LibPETSc.TSSetRHSFunction(petsclib, ts, nothing, RHS_PTR[], ctxptr)
            end
            if is_split
                LibPETSc.TSSetRHSFunction(petsclib, ts, nothing, SPLIT_RHS_PTR[], ctxptr)
            end
            if has_jac && uses_sparse_jac
                pattern = _jacobian_pattern(J0, n)
                h.jac_mat = PETSc.MatSeqAIJWithArrays(petsclib, MPI.COMM_SELF, pattern)
                LibPETSc.TSSetIJacobian(
                    petsclib, ts, h.jac_mat, h.jac_mat, SPARSE_IJACOBIAN_PTR[], ctxptr,
                )
            elseif has_jac
                h.jac_mat = PETSc.MatSeqAIJ(petsclib, n, n, n)
                LibPETSc.TSSetIJacobian(
                    petsclib, ts, h.jac_mat, h.jac_mat, IJACOBIAN_PTR[], ctxptr,
                )
            end
            LibPETSc.TSMonitorSet(petsclib, ts, MONITOR_PTR[], ctxptr)
            LibPETSc.TSSetTime(petsclib, ts, t0)
            LibPETSc.TSSetTimeStep(petsclib, ts, Float64(dt))
            LibPETSc.TSSetMaxTime(petsclib, ts, tf)
            LibPETSc.TSSetMaxSteps(petsclib, ts, LibPETSc.PetscInt(maxiters))
            LibPETSc.TSSetExactFinalTime(
                petsclib, ts, LibPETSc.TS_EXACTFINALTIME_MATCHSTEP,
            )
            if (reltol !== nothing || abstol !== nothing) && _adapts(alg) === false
                @warn "`$(_ts_type(alg))` has no embedded error estimate in PETSc, so " *
                    "it steps at the requested dt and ignores reltol/abstol"
            end
            if reltol !== nothing || abstol !== nothing
                novec = LibPETSc.PetscVec{typeof(petsclib)}()
                LibPETSc.TSSetTolerances(
                    petsclib, ts,
                    abstol === nothing ? 1.0e-6 : Float64(abstol), novec,
                    reltol === nothing ? 1.0e-3 : Float64(reltol), novec,
                )
            end
            # A user's own PETSc options are parsed last, so they win.
            effective_options = String[]
            adaptive || append!(effective_options, ["-ts_adapt_type", "none"])
            dtmin === nothing ||
                append!(effective_options, ["-ts_adapt_dt_min", string(Float64(dtmin))])
            dtmax === nothing ||
                append!(effective_options, ["-ts_adapt_dt_max", string(Float64(dtmax))])
            append!(effective_options, alg.petsc_options)
            if !isempty(effective_options)
                parsed = PETSc.parse_options(effective_options)
                h.opts = PETSc.Options(petsclib; parsed...)
                push!(h.opts)
                try
                    LibPETSc.TSSetFromOptions(petsclib, ts)
                finally
                    pop!(h.opts)
                end
            else
                LibPETSc.TSSetFromOptions(petsclib, ts)
            end
        end
    catch
        _destroy!(h)
        rethrow()
    end
    return h
end

function _read_stats(h::TSHandles)
    pl, ts = h.petsclib, h.ts
    return (
        nsteps = Int(LibPETSc.TSGetStepNumber(pl, ts)),
        nreject = Int(LibPETSc.TSGetStepRejections(pl, ts)),
        nnonliniter = Int(LibPETSc.TSGetSNESIterations(pl, ts)),
        nnonlinfail = Int(LibPETSc.TSGetSNESFailures(pl, ts)),
    )
end

function _assemble(prob, alg, h::TSHandles, tend, uend, st)
    ctx = h.ctx
    tf, t0, tol = h.tf, h.t0, 100 * eps(max(one(Float64), abs(h.tf)))
    # `-ts_exact_final_time interpolate` steps past tf and then interpolates
    # back, so the monitor reports an overshoot point mid-sequence, not last.
    keep = findall(t -> t <= tf + tol, ctx.ts)
    if length(keep) != length(ctx.ts)
        ctx.ts = ctx.ts[keep]
        ctx.us = ctx.us[keep]
        ctx.dense && (ctx.dus = ctx.dus[keep])
    end
    if h.save_end && (isempty(ctx.ts) || ctx.ts[end] < tend - tol)
        _record!(ctx, tend, uend)
    end
    if !h.save_start && length(ctx.ts) > 1 && abs(ctx.ts[1] - t0) <= tol
        popfirst!(ctx.ts)
        popfirst!(ctx.us)
        ctx.dense && popfirst!(ctx.dus)
    end

    finite = isempty(ctx.us) || all(isfinite, ctx.us[end])
    retcode = if !finite
        SciMLBase.ReturnCode.Unstable
    elseif tend >= tf - tol
        SciMLBase.ReturnCode.Success
    elseif st.nsteps >= h.maxiters
        SciMLBase.ReturnCode.MaxIters
    else
        SciMLBase.ReturnCode.Failure
    end
    stats = SciMLBase.DEStats(
        ctx.nf, -1, -1, -1, ctx.njacs, st.nnonliniter, st.nnonlinfail, -1, -1, -1,
        st.nsteps, st.nreject, 0.0,
    )
    return SciMLBase.build_solution(
        prob, alg, ctx.ts, ctx.us; retcode = retcode, stats = stats,
        dense = ctx.dense, interp = _interp(ctx),
    )
end

function SciMLBase.__solve(
        prob::SciMLBase.AbstractODEProblem, alg::PETScTSAlgorithm;
        callback = nothing, kwargs...,
    )
    if callback !== nothing
        return SciMLBase.solve!(SciMLBase.__init(prob, alg; callback = callback, kwargs...))
    end
    h = _setup(prob, alg; kwargs...)
    ctx, pl = h.ctx, h.petsclib
    tend, uend, st = h.t0, copy(h.u0), nothing
    try
        GC.@preserve ctx begin
            try
                LibPETSc.TSSolve(pl, h.ts, h.u)
            catch
                # A callback that threw reports failure to PETSc, which raises a
                # PetscError here. The user's own exception is the useful one.
                ctx.err === nothing && rethrow()
            end
        end
        ctx.err === nothing || throw(ctx.err)
        tend = Float64(LibPETSc.TSGetSolveTime(pl, h.ts))
        st = _read_stats(h)
        uend = PETSc.withlocalarray!(
            ua -> Vector{Float64}(ua), h.u; read = true, write = false,
        )
    finally
        _destroy!(h)
    end
    return _assemble(prob, alg, h, tend, uend, st)
end

mutable struct PETScIntegrator{Alg, P, H, Pr, CB} <:
    SciMLBase.AbstractODEIntegrator{Alg, true, Vector{Float64}, Float64}
    alg::Alg
    u::Vector{Float64}
    uprev::Vector{Float64}
    t::Float64
    tprev::Float64
    dt::Float64
    tdir::Float64
    p::P
    h::H
    prob::Pr
    callbacks::CB
    kwargs::Any
    sol::Any
    finished::Bool
    derivative_discontinuity::Bool
end

_discrete_callbacks(::Nothing) = ()
_discrete_callbacks(cb::SciMLBase.DiscreteCallback) = (cb,)
function _discrete_callbacks(cb::SciMLBase.CallbackSet)
    isempty(cb.continuous_callbacks) || throw(
        ArgumentError(
            "PETScDiffEq supports DiscreteCallback only; a ContinuousCallback needs " *
                "root finding on the interpolant, which is not implemented",
        ),
    )
    return Tuple(cb.discrete_callbacks)
end
_discrete_callbacks(::SciMLBase.AbstractContinuousCallback) = throw(
    ArgumentError(
        "PETScDiffEq supports DiscreteCallback only; a ContinuousCallback needs " *
            "root finding on the interpolant, which is not implemented",
    ),
)

function SciMLBase.derivative_discontinuity!(integ::PETScIntegrator, bool::Bool)
    integ.derivative_discontinuity = bool
    return nothing
end

SciMLBase.get_dt(integ::PETScIntegrator) = integ.dt
function SciMLBase.get_proposed_dt(integ::PETScIntegrator)
    integ.finished && return integ.dt
    return Float64(LibPETSc.TSGetTimeStep(integ.h.petsclib, integ.h.ts))
end
function SciMLBase.set_proposed_dt!(integ::PETScIntegrator, dt)
    integ.finished || LibPETSc.TSSetTimeStep(integ.h.petsclib, integ.h.ts, Float64(dt))
    return nothing
end
function SciMLBase.savevalues!(integ::PETScIntegrator)
    push!(integ.h.ctx.ts, integ.t)
    push!(integ.h.ctx.us, copy(integ.u))
    return nothing
end

function _apply_callbacks!(integ::PETScIntegrator)
    h = integ.h
    ctx = h.ctx
    for cb in integ.callbacks
        integ.finished && return nothing
        cb.condition(integ.u, integ.t, integ) || continue
        # An affect! is assumed to change the state unless it says otherwise
        # through derivative_discontinuity!(integ, false).
        integ.derivative_discontinuity = true
        cb.affect!(integ)
        integ.finished && return nothing
        if integ.derivative_discontinuity
            # The changed state has to reach PETSc's own solution vector, and a
            # multistep method must drop history taken before the jump.
            PETSc.withlocalarray!(ua -> copyto!(ua, integ.u), h.u; read = false, write = true)
            LibPETSc.TSRestartStep(h.petsclib, h.ts)
        end
        if cb.save_positions[2] && ctx.save_everystep && isempty(ctx.saveat)
            _record!(ctx, integ.t, integ.u)
        end
    end
    return nothing
end

function SciMLBase.__init(
        prob::SciMLBase.AbstractODEProblem, alg::PETScTSAlgorithm;
        callback = nothing, kwargs...,
    )
    callbacks = _discrete_callbacks(callback)
    h = _setup(prob, alg; kwargs...)
    LibPETSc.TSSetUp(h.petsclib, h.ts)
    _initial_save!(h)
    integ = PETScIntegrator(
        alg, copy(h.u0), copy(h.u0), h.t0, h.t0,
        Float64(LibPETSc.TSGetTimeStep(h.petsclib, h.ts)), 1.0,
        prob.p, h, prob, callbacks, NamedTuple(kwargs), _initial_solution(prob, alg, h),
        false, false,
    )
    for cb in callbacks
        cb.initialize(cb, integ.u, integ.t, integ)
    end
    return integ
end

function _initial_save!(h::TSHandles)
    ctx = h.ctx
    tol = 100 * eps(max(one(Float64), abs(h.tf)))
    if isempty(ctx.saveat)
        if h.save_start
            _record!(ctx, h.t0, h.u0)
        end
    else
        while ctx.saveat_idx <= length(ctx.saveat) && ctx.saveat[ctx.saveat_idx] <= h.t0 + tol
            _record!(ctx, ctx.saveat[ctx.saveat_idx], h.u0)
            ctx.saveat_idx += 1
        end
    end
    return nothing
end

_initial_solution(prob, alg, h::TSHandles) = SciMLBase.build_solution(
    prob, alg, h.ctx.ts, h.ctx.us; retcode = SciMLBase.ReturnCode.Default,
    dense = h.ctx.dense, interp = _interp(h.ctx),
)

# A reinitialised integrator gets fresh PETSc objects built from the keywords
# given to `init`, so its solve is identical to a fresh one and it works even
# after the previous solve released them.
function SciMLBase.reinit!(
        integ::PETScIntegrator, u0 = integ.prob.u0;
        t0 = integ.prob.tspan[1], tf = integ.prob.tspan[2],
        erase_sol = true, saveat = nothing, reinit_callbacks = true,
        initialize_save = true,
    )
    old = integ.h
    _destroy!(old)
    prob = SciMLBase.remake(integ.prob; u0 = u0, tspan = (t0, tf))
    setup_kwargs = saveat === nothing ? integ.kwargs : merge(integ.kwargs, (saveat = saveat,))
    h = _setup(prob, integ.alg; setup_kwargs...)
    LibPETSc.TSSetUp(h.petsclib, h.ts)
    if !erase_sol
        append!(h.ctx.ts, old.ctx.ts)
        append!(h.ctx.us, old.ctx.us)
        h.ctx.dense && append!(h.ctx.dus, old.ctx.dus)
    end
    initialize_save && _initial_save!(h)
    integ.h = h
    integ.u = copy(h.u0)
    integ.uprev = copy(h.u0)
    integ.t = h.t0
    integ.tprev = h.t0
    integ.dt = Float64(LibPETSc.TSGetTimeStep(h.petsclib, h.ts))
    integ.finished = false
    integ.derivative_discontinuity = false
    integ.sol = _initial_solution(integ.prob, integ.alg, h)
    if reinit_callbacks
        for cb in integ.callbacks
            cb.initialize(cb, integ.u, integ.t, integ)
        end
    end
    return nothing
end

@static if isdefined(SciMLBase, :has_reinit)
    SciMLBase.has_reinit(::PETScIntegrator) = true
end

function _finish!(integ::PETScIntegrator, retcode = nothing)
    integ.finished && return nothing
    h = integ.h
    for cb in integ.callbacks
        cb.finalize(cb, integ.u, integ.t, integ)
    end
    st = _read_stats(h)
    sol = _assemble(integ.prob, integ.alg, h, integ.t, copy(integ.u), st)
    integ.sol = retcode === nothing ? sol : SciMLBase.solution_new_retcode(sol, retcode)
    integ.finished = true
    _destroy!(h)
    return nothing
end

function SciMLBase.terminate!(
        integ::PETScIntegrator, retcode = SciMLBase.ReturnCode.Terminated,
    )
    _finish!(integ, retcode)
    return nothing
end

function SciMLBase.step!(integ::PETScIntegrator)
    integ.finished && return nothing
    h = integ.h
    ctx, pl = h.ctx, h.petsclib
    copyto!(integ.uprev, integ.u)
    integ.tprev = integ.t
    GC.@preserve ctx begin
        try
            LibPETSc.TSStep(pl, h.ts)
        catch
            ctx.err === nothing && rethrow()
        end
    end
    if ctx.err !== nothing
        err = ctx.err
        _finish!(integ)
        throw(err)
    end
    integ.t = Float64(LibPETSc.TSGetTime(pl, h.ts))
    integ.dt = Float64(LibPETSc.TSGetTimeStep(pl, h.ts))
    PETSc.withlocalarray!(ua -> copyto!(integ.u, ua), h.u; read = true, write = false)
    tol = 100 * eps(max(one(Float64), abs(h.tf)))
    if isempty(ctx.saveat)
        if ctx.save_everystep
            _record!(ctx, integ.t, integ.u)
        end
    else
        while ctx.saveat_idx <= length(ctx.saveat) &&
                ctx.saveat[ctx.saveat_idx] <= integ.t + tol
            want = ctx.saveat[ctx.saveat_idx]
            if abs(want - integ.t) <= tol
                _record!(ctx, want, integ.u)
            else
                LibPETSc.TSInterpolate(pl, h.ts, want, ctx.work)
                PETSc.withlocalarray!(
                    wa -> _record!(ctx, want, wa),
                    ctx.work; read = true, write = false,
                )
            end
            ctx.saveat_idx += 1
        end
    end
    _apply_callbacks!(integ)
    integ.finished && return nothing
    if !all(isfinite, integ.u) || integ.t >= h.tf - tol ||
            Int(LibPETSc.TSGetStepNumber(pl, h.ts)) >= h.maxiters
        _finish!(integ)
    end
    return nothing
end

function SciMLBase.solve!(integ::PETScIntegrator)
    while !integ.finished
        SciMLBase.step!(integ)
    end
    return integ.sol
end

SciMLBase.done(integ::PETScIntegrator) = integ.finished

end
