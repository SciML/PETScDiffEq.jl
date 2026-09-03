module PETScDiffEq

using DiffEqBase: DiffEqBase
using MPI: MPI
using PETSc: PETSc
using PETSc.LibPETSc: LibPETSc
using SciMLBase: SciMLBase

export TSRK, TSRosW, TSImplicit, TSARKIMEX

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

_uses_ifunction(::TSRK) = false
_uses_ifunction(::TSRosW) = true
_uses_ifunction(::TSImplicit) = true
_uses_ifunction(::TSARKIMEX) = true

mutable struct TSContext{F, F2, P, T}
    petsclib::T
    f!::F
    f2!::F2
    p::P
    du::Vector{Float64}
    u::Vector{Float64}
    ts::Vector{Float64}
    us::Vector{Vector{Float64}}
    err::Union{Nothing, Any}
end

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
            @. fa = xda - ctx.du
        end
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(1)
    end
    return LibPETSc.PetscErrorCode(0)
end

const IFUNCTION_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _monitor!(
        ::LibPETSc.CTS,
        ::LibPETSc.PetscInt,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    try
        PETSc.withlocalarray!(x; read = true, write = false) do xa
            push!(ctx.ts, Float64(t))
            push!(ctx.us, Vector{Float64}(xa))
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
    return nothing
end

function _cstr(f::F, s::AbstractString) where {F}
    str = String(s)
    return GC.@preserve str f(Base.unsafe_convert(Ptr{Cchar}, str))
end

_ts_type(::TSRK) = "rk"
_ts_type(::TSRosW) = "rosw"
_ts_type(alg::TSImplicit) = alg.subtype
_ts_type(::TSARKIMEX) = "arkimex"

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

function SciMLBase.__solve(
        prob::SciMLBase.AbstractODEProblem,
        alg::PETScTSAlgorithm;
        dt = nothing,
        reltol = nothing,
        abstol = nothing,
        maxiters = 1000000,
        kwargs...,
    )
    SciMLBase.isinplace(prob) ||
        throw(ArgumentError("PETScDiffEq requires an in-place ODEProblem, f!(du, u, p, t)"))
    prob.u0 isa AbstractVector{<:Real} ||
        throw(ArgumentError("PETScDiffEq requires a real AbstractVector u0"))
    dt === nothing && throw(ArgumentError("PETScDiffEq requires an initial dt"))
    is_split = prob.f isa SciMLBase.SplitFunction
    is_split && !(alg isa TSARKIMEX) &&
        throw(ArgumentError("PETScDiffEq only supports SplitODEProblem with TSARKIMEX"))

    t0, tf = Float64(prob.tspan[1]), Float64(prob.tspan[2])
    t0 < tf || throw(ArgumentError("PETScDiffEq requires tspan[1] < tspan[2]"))

    u0 = Vector{Float64}(vec(prob.u0))
    n = length(u0)

    petsclib = PETSc.getlib(PetscScalar = Float64)
    PETSc.initialized(petsclib) || PETSc.initialize(petsclib)

    f1 = is_split ? prob.f.f1.f : prob.f.f
    f2 = is_split ? prob.f.f2.f : nothing
    ctx = TSContext(
        petsclib, f1, f2, prob.p,
        similar(u0), similar(u0),
        Float64[], Vector{Float64}[], nothing,
    )

    ts = LibPETSc.TS(petsclib)
    u = LibPETSc.PetscVec(petsclib)
    opts = PETSc.Options(petsclib)
    pushed = false
    nsteps = 0
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
            if !isempty(alg.petsc_options)
                parsed = PETSc.parse_options(alg.petsc_options)
                opts = PETSc.Options(petsclib; parsed...)
                push!(opts)
                pushed = true
            end
            LibPETSc.TSSetFromOptions(petsclib, ts)
            if pushed
                pop!(opts)
                pushed = false
            end
            LibPETSc.TSSolve(petsclib, ts, u)
        end

        ctx.err === nothing || throw(ctx.err)
        tend = Float64(LibPETSc.TSGetSolveTime(petsclib, ts))
        nsteps = Int(LibPETSc.TSGetStepNumber(petsclib, ts))
    finally
        pushed && pop!(opts)
        opts.ptr == C_NULL || PETSc.destroy(opts)
        u.ptr == C_NULL || PETSc.destroy(u)
        ts.ptr == C_NULL || LibPETSc.TSDestroy(petsclib, ts)
    end

    if isempty(ctx.ts) || ctx.ts[end] < tend
        push!(ctx.ts, tend)
        push!(ctx.us, copy(ctx.us[end]))
    end

    retcode = if tend >= tf - 100 * eps(tf)
        SciMLBase.ReturnCode.Success
    elseif nsteps >= maxiters
        SciMLBase.ReturnCode.MaxIters
    else
        SciMLBase.ReturnCode.Failure
    end
    return SciMLBase.build_solution(prob, alg, ctx.ts, ctx.us; retcode = retcode)
end

end
