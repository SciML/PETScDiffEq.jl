module PETScDiffEq

using DiffEqBase: DiffEqBase
using LinearAlgebra: LinearAlgebra, mul!
using MPI: MPI
using PETSc: PETSc
using PETSc.LibPETSc: LibPETSc
using SciMLBase: SciMLBase
using SparseArrays: SparseArrays, SparseMatrixCSC, findnz, sparse

export TSRK, TSRosW, TSImplicit, TSARKIMEX, TSGeneric

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
    J::JBUF
    ts::Vector{Float64}
    us::Vector{Vector{Float64}}
    saveat::Vector{Float64}
    saveat_idx::Int
    save_everystep::Bool
    work::V
    nf::Int
    err::Union{Nothing, Any}
end

_record!(ctx, t, ua) = (push!(ctx.ts, Float64(t)); push!(ctx.us, Vector{Float64}(ua)))

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
        n = length(ctx.u)
        for j in 1:n, i in 1:n
            m = ctx.M === nothing ? (i == j ? 1.0 : 0.0) : ctx.M[i, j]
            A[i, j] = shift * m - ctx.J[i, j]
        end
        PETSc.assemble!(A)
        if B.ptr != A.ptr
            for j in 1:n, i in 1:n
                m = ctx.M === nothing ? (i == j ? 1.0 : 0.0) : ctx.M[i, j]
                B[i, j] = shift * m - ctx.J[i, j]
            end
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
        n = length(ctx.u)
        # The diagonal always gets a `shift*I` contribution, whether or not
        # the ODE Jacobian has a stored entry there, so it is set first and
        # then overwritten below wherever the ODE Jacobian also contributes.
        for i in 1:n
            A[i, i] = shift
        end
        for j in 1:n, k in ctx.J.colptr[j]:(ctx.J.colptr[j + 1] - 1)
            i = ctx.J.rowval[k]
            A[i, j] = (i == j ? shift : 0.0) - ctx.J.nzval[k]
        end
        PETSc.assemble!(A)
        if B.ptr != A.ptr
            for i in 1:n
                B[i, i] = shift
            end
            for j in 1:n, k in ctx.J.colptr[j]:(ctx.J.colptr[j + 1] - 1)
                i = ctx.J.rowval[k]
                B[i, j] = (i == j ? shift : 0.0) - ctx.J.nzval[k]
            end
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

const UNSUPPORTED_KWARGS = (:tstops, :save_idxs, :d_discontinuities, :callback, :dense)

function SciMLBase.__solve(
        prob::SciMLBase.AbstractODEProblem,
        alg::PETScTSAlgorithm;
        dt = nothing,
        reltol = nothing,
        abstol = nothing,
        maxiters = 1000000,
        adaptive = true,
        saveat = Float64[],
        save_everystep = true,
        save_start = true,
        save_end = true,
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
    has_jac = _uses_ifunction(alg) && !is_split && prob.f.jac !== nothing
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
    ctx = TSContext(
        petsclib, f1, f2, jac_fn, prob.p,
        similar(u0), similar(u0), similar(u0), M, J0,
        Float64[], Vector{Float64}[],
        saveat_times, 1, save_everystep,
        PETSc.VecSeq(petsclib, n), 0, nothing,
    )

    ts = LibPETSc.TS(petsclib)
    u = LibPETSc.PetscVec(petsclib)
    jac_mat = LibPETSc.PetscMat(petsclib)
    opts = PETSc.Options(petsclib)
    pushed = false
    nsteps = 0
    nreject = 0
    nnonliniter = 0
    nnonlinfail = 0
    uend = copy(u0)
    tend = t0

    try
        ts = LibPETSc.TSCreate(petsclib, MPI.COMM_SELF)
        LibPETSc.TSSetProblemType(petsclib, ts, LibPETSc.TS_NONLINEAR)
        LibPETSc.TSSetType(petsclib, ts, _ts_type(alg))
        _set_subtype!(petsclib, ts, alg)

        u = PETSc.VecSeq(petsclib, n)
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
                jac_mat = PETSc.MatSeqAIJWithArrays(petsclib, MPI.COMM_SELF, pattern)
                LibPETSc.TSSetIJacobian(
                    petsclib, ts, jac_mat, jac_mat, SPARSE_IJACOBIAN_PTR[], ctxptr,
                )
            elseif has_jac
                jac_mat = PETSc.MatSeqAIJ(petsclib, n, n, n)
                LibPETSc.TSSetIJacobian(
                    petsclib, ts, jac_mat, jac_mat, IJACOBIAN_PTR[], ctxptr,
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
            if reltol !== nothing || abstol !== nothing
                novec = LibPETSc.PetscVec{typeof(petsclib)}()
                LibPETSc.TSSetTolerances(
                    petsclib, ts,
                    abstol === nothing ? 1.0e-6 : Float64(abstol), novec,
                    reltol === nothing ? 1.0e-3 : Float64(reltol), novec,
                )
            end
            # A user's own -ts_adapt_type wins: their options are parsed last.
            effective_options = adaptive ? alg.petsc_options :
                vcat(["-ts_adapt_type", "none"], alg.petsc_options)
            if !isempty(effective_options)
                parsed = PETSc.parse_options(effective_options)
                opts = PETSc.Options(petsclib; parsed...)
                push!(opts)
                pushed = true
            end
            LibPETSc.TSSetFromOptions(petsclib, ts)
            if pushed
                pop!(opts)
                pushed = false
            end
            try
                LibPETSc.TSSolve(petsclib, ts, u)
            catch
                # A callback that threw reports failure to PETSc, which raises a
                # PetscError here. The user's own exception is the useful one.
                ctx.err === nothing && rethrow()
            end
        end

        ctx.err === nothing || throw(ctx.err)
        tend = Float64(LibPETSc.TSGetSolveTime(petsclib, ts))
        nsteps = Int(LibPETSc.TSGetStepNumber(petsclib, ts))
        nreject = Int(LibPETSc.TSGetStepRejections(petsclib, ts))
        nnonliniter = Int(LibPETSc.TSGetSNESIterations(petsclib, ts))
        nnonlinfail = Int(LibPETSc.TSGetSNESFailures(petsclib, ts))
        uend = PETSc.withlocalarray!(
            ua -> Vector{Float64}(ua), u; read = true, write = false,
        )
    finally
        pushed && pop!(opts)
        opts.ptr == C_NULL || PETSc.destroy(opts)
        jac_mat.ptr == C_NULL || PETSc.destroy(jac_mat)
        ctx.work.ptr == C_NULL || PETSc.destroy(ctx.work)
        u.ptr == C_NULL || PETSc.destroy(u)
        ts.ptr == C_NULL || LibPETSc.TSDestroy(petsclib, ts)
    end

    tol = 100 * eps(max(one(Float64), abs(tf)))
    if save_end && (isempty(ctx.ts) || ctx.ts[end] < tend - tol)
        push!(ctx.ts, tend)
        push!(ctx.us, uend)
    end
    if !save_start && length(ctx.ts) > 1 && abs(ctx.ts[1] - t0) <= tol
        popfirst!(ctx.ts)
        popfirst!(ctx.us)
    end

    finite = isempty(ctx.us) || all(isfinite, ctx.us[end])
    retcode = if !finite
        SciMLBase.ReturnCode.Unstable
    elseif tend >= tf - tol
        SciMLBase.ReturnCode.Success
    elseif nsteps >= maxiters
        SciMLBase.ReturnCode.MaxIters
    else
        SciMLBase.ReturnCode.Failure
    end
    stats = SciMLBase.DEStats(
        ctx.nf, -1, -1, -1, -1, nnonliniter, nnonlinfail, -1, -1, -1,
        nsteps, nreject, 0.0,
    )
    return SciMLBase.build_solution(
        prob, alg, ctx.ts, ctx.us; retcode = retcode, stats = stats,
    )
end

end
