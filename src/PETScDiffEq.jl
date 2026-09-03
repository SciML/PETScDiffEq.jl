module PETScDiffEq

using DiffEqBase: DiffEqBase
using MPI: MPI
using PETSc: PETSc
using PETSc.LibPETSc: LibPETSc
using SciMLBase: SciMLBase

export TSRK

abstract type PETScTSAlgorithm <: SciMLBase.AbstractODEAlgorithm end

struct TSRK <: PETScTSAlgorithm
    subtype::String
    petsc_options::Vector{String}
end

TSRK(subtype::AbstractString = "5dp", petsc_options::AbstractVector{<:AbstractString} = String[]) =
    TSRK(String(subtype), String[String(o) for o in petsc_options])

mutable struct TSContext{F, P, T}
    petsclib::T
    f!::F
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
    MONITOR_PTR[] = @cfunction(
        _monitor!,
        LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, LibPETSc.PetscInt, LibPETSc.PetscReal, LibPETSc.CVec, Ptr{Cvoid})
    )
    return nothing
end

function _cstr(f::F, s::AbstractString) where {F}
    str = String(s)
    return GC.@preserve str f(Base.unsafe_convert(Ptr{Cchar}, str))
end

_ts_type(::TSRK) = "rk"

_set_subtype!(petsclib, ts, alg::TSRK) =
    _cstr(p -> LibPETSc.TSRKSetType(petsclib, ts, p), alg.subtype)

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

    t0, tf = Float64(prob.tspan[1]), Float64(prob.tspan[2])
    t0 < tf || throw(ArgumentError("PETScDiffEq requires tspan[1] < tspan[2]"))

    u0 = Vector{Float64}(vec(prob.u0))
    n = length(u0)

    petsclib = PETSc.getlib(PetscScalar = Float64)
    PETSc.initialized(petsclib) || PETSc.initialize(petsclib)

    ctx = TSContext(
        petsclib, prob.f.f, prob.p,
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
            LibPETSc.TSSetRHSFunction(petsclib, ts, nothing, RHS_PTR[], ctxptr)
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
