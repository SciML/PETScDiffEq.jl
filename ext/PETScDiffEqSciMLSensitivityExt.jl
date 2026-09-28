module PETScDiffEqSciMLSensitivityExt

using PETScDiffEq: PETScDiffEq, PETScAdjoint
using SciMLSensitivity: SciMLSensitivity

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

end
