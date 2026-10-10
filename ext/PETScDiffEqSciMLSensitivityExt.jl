module PETScDiffEqSciMLSensitivityExt

using PETScDiffEq: PETScDiffEq, PETScAdjoint
using SciMLBase: SciMLBase
using SciMLSensitivity: SciMLSensitivity
using ChainRulesCore: ChainRulesCore

function SciMLSensitivity._adjoint_sensitivities(
        sol, sensealg::PETScAdjoint, alg::PETScDiffEq.AnyPETScTS;
        t = nothing, dgdu_discrete = nothing, dgdp_discrete = nothing,
        dgdu_continuous = nothing, dgdp_continuous = nothing, g = nothing,
        no_start = false, callback = nothing, kwargs...,
    )
    return PETScDiffEq._discrete_adjoint(
        sol.prob, alg, sensealg;
        t, dgdu_discrete, dgdp_discrete, dgdu_continuous, dgdp_continuous, g, no_start,
        callback, kwargs...,
    )
end

function SciMLSensitivity._adjoint_sensitivities(sol, ::PETScAdjoint, alg; kwargs...)
    throw(
        ArgumentError(
            "PETScAdjoint runs PETSc's own adjoint, so it needs one of PETScDiffEq's " *
                "algorithms such as TSRK, but got $(nameof(typeof(alg))); for other " *
                "solvers use an adjoint from SciMLSensitivity such as GaussAdjoint",
        ),
    )
end

# Building an algorithm copies its option strings in a loop, which Zygote cannot trace.
function ChainRulesCore.rrule(
        T::Type{<:Union{PETScAdjoint, PETScDiffEq.AnyPETScTS}}, args...; kwargs...,
    )
    constant(_) = (ChainRulesCore.NoTangent(), map(_ -> ChainRulesCore.NoTangent(), args)...)
    return T(args...; kwargs...), constant
end

const _USE_ADJOINT_SENSITIVITIES =
    "call `adjoint_sensitivities(sol, alg; sensealg = PETScAdjoint(), t, dgdu_discrete, ...)` instead"

function _check_differentiated_solve(prob, alg, originator, args)
    originator isa SciMLBase.ChainRulesOriginator || throw(
        ArgumentError(
            "PETScAdjoint differentiates `solve` for Zygote and other ChainRules-based " *
                "AD only, not for $(nameof(typeof(originator))), which it has not been " *
                "verified with; $_USE_ADJOINT_SENSITIVITIES",
        ),
    )
    isempty(args) || throw(
        ArgumentError(
            "PETScAdjoint differentiates `solve(prob, alg; kwargs...)` only, without " *
                "further positional arguments",
        ),
    )
    prob isa SciMLBase.AbstractODEProblem ||
        throw(ArgumentError("PETScAdjoint supports an ODEProblem, not a DAEProblem"))
    prob.f isa SciMLBase.DynamicalODEFunction && throw(
        ArgumentError(
            "PETScAdjoint does not differentiate `solve` of a DynamicalODEProblem or " *
                "SecondOrderODEProblem, which it has not been verified with; " *
                _USE_ADJOINT_SENSITIVITIES,
        ),
    )
    (PETScDiffEq._distributed(alg) || PETScDiffEq._alg_dm(alg) !== nothing) && throw(
        ArgumentError(
            "PETScAdjoint differentiates `solve` on MPI.COMM_SELF without a `dm` only, " *
                "since the pullback is handed whole states; $_USE_ADJOINT_SENSITIVITIES",
        ),
    )
    (eltype(prob.u0) === Float64 && eltype(prob.tspan) === Float64) || throw(
        ArgumentError(
            "PETScAdjoint differentiates `solve` of a Float64 problem only, but `u0` " *
                "holds $(eltype(prob.u0)) and `tspan` $(eltype(prob.tspan)); " *
                _USE_ADJOINT_SENSITIVITIES,
        ),
    )
    return nothing
end

_absent(x) = x === nothing || x isa ChainRulesCore.AbstractZero
_absent(x::ChainRulesCore.AbstractThunk) = _absent(ChainRulesCore.unthunk(x))
_absent(x::Union{Tuple, NamedTuple}) = all(_absent, x)
_absent(x::ChainRulesCore.Tangent) = _absent(ChainRulesCore.backing(x))

# One cotangent per saved time, from whichever form the AD package gives the solution's.
function _saved_cotangents(Δ, nt)
    Δ = ChainRulesCore.unthunk(Δ)
    _absent(Δ) && return nothing
    if Δ isa Union{NamedTuple, ChainRulesCore.Tangent}
        for name in propertynames(Δ)
            name in (:u, :t) || _absent(getproperty(Δ, name)) || throw(
                ArgumentError(
                    "the loss depends on the solution's `$name`, but PETScAdjoint carries " *
                        "back only the states `solve` saved: PETSc's adjoint has no " *
                        "derivative of interpolation, so save at the times the loss needs " *
                        "with `saveat` instead of calling `sol(t)`, and read `u0` and `p` " *
                        "from the loss's own arguments instead of `sol.prob`",
                ),
            )
        end
        Δ = hasproperty(Δ, :u) ? ChainRulesCore.unthunk(Δ.u) : nothing
    end
    # A solution or a VectorOfArray holds its columns in `u`.
    hasproperty(Δ, :u) && (Δ = Δ.u)
    _absent(Δ) && return nothing
    if Δ isa AbstractArray{<:Number} && nt > 0 && length(Δ) % nt == 0
        return eachcol(reshape(Δ, :, nt))
    end
    Δ isa Union{AbstractVector, Tuple} && length(Δ) == nt && return Δ
    throw(
        ArgumentError(
            "PETScAdjoint cannot match a cotangent of type $(typeof(Δ)) to the $nt " *
                "states `solve` saved; take the loss from `Array(sol)`, `sol[end]`, " *
                "`sol.u[i]` or `sol[i, j]`",
        ),
    )
end

# The cotangent of the whole state, or `nothing` where it is zero.
function _state_cotangent(x, n, kept)
    x = ChainRulesCore.unthunk(x)
    _absent(x) && return nothing
    idxs = kept === nothing ? (1:n) : kept
    x isa Union{Real, AbstractArray{<:Real}} && length(x) == length(idxs) || throw(
        ArgumentError(
            "PETScAdjoint cannot match a cotangent of type $(typeof(x)) to a saved " *
                "state of $(length(idxs)) real entries",
        ),
    )
    iszero(x) && return nothing
    g = zeros(n)
    for (i, v) in zip(idxs, x)
        g[i] += v
    end
    return g
end

function PETScDiffEq._solve_and_pullback(
        prob, alg, sensealg::PETScAdjoint, u0, p, originator, args...;
        save_idxs = nothing, kwargs...,
    )
    _check_differentiated_solve(prob, alg, originator, args)
    given = values(prob.kwargs)
    whole = haskey(given, :save_idxs) ?
        SciMLBase.remake(prob; kwargs = Base.structdiff(given, NamedTuple{(:save_idxs,)})) :
        prob
    # Refuses what the adjoint cannot take, such as a callback, before the solve runs.
    PETScDiffEq._adjoint_solve_kwargs(whole, kwargs)
    sol = SciMLBase.solve(prob, alg; save_idxs, kwargs...)
    ts = copy(sol.t)
    n = length(prob.u0)
    kept = save_idxs === nothing ? nothing :
        save_idxs isa Integer ? [Int(save_idxs)] : Vector{Int}(collect(save_idxs))
    has_p = !(p === nothing || p isa SciMLBase.NullParameters)

    function petsc_adjoint_pullback(Δ)
        saved = _saved_cotangents(Δ, length(ts))
        at, costs = Int[], Vector{Float64}[]
        # A time whose cotangent is zero adds nothing, so the adjoint's run need not stop there.
        for (i, x) in enumerate(something(saved, ()))
            g = _state_cotangent(x, n, kept)
            g === nothing && continue
            push!(at, i)
            push!(costs, g)
        end
        none = ChainRulesCore.NoTangent()
        isempty(at) && return (none, none, none, zero(u0), has_p ? zero(p) : none, none)
        dgdu(out, u, p, t, i) = (copyto!(out, costs[i]); nothing)
        du0, dp = PETScDiffEq._discrete_adjoint(
            whole, alg, sensealg; kwargs..., t = ts[at], dgdu_discrete = dgdu,
        )
        return (none, none, none, du0, has_p ? dp' : none, none)
    end
    return sol, petsc_adjoint_pullback
end

end
