# A problem without a `jac` gets one from `autodiff`, in the form a user's `jac` takes, so
# it runs through the same path. `AutoFiniteDiff()` leaves the Jacobian to PETSc instead.

_check_autodiff(ad::ADTypes.AbstractADType) = ad
_check_autodiff(ad) = throw(
    ArgumentError(
        "`autodiff` takes an ADTypes backend, such as `AutoForwardDiff()` or " *
            "`AutoFiniteDiff()`; got $(repr(ad))",
    ),
)

_autodiff(alg::Union{TSRosW, TSImplicit, TSIRK, TSDAE, TSARKIMEX, TSGeneric}) = alg.autodiff
_autodiff(::Union{TSRK, TSMPRK}) = nothing

# Whether PETSc differences the step's equations itself rather than being handed a Jacobian.
function _petsc_differences(alg)
    ad = _autodiff(alg)
    return ad !== nothing && ADTypes.dense_ad(ad) isa AutoFiniteDiff
end

# The Jacobian is written into a matrix with the prototype's pattern, so a sparse
# prototype is the pattern to colour, as OrdinaryDiffEq uses it, whatever detector the
# backend brings; only its colouring is kept. Without one the matrix is dense, and a
# sparse backend that has no way to find a pattern is used dense.
function _with_pattern(backend, proto)
    if proto isa SparseArrays.AbstractSparseMatrix
        coloring = backend isa ADTypes.AutoSparse ? ADTypes.coloring_algorithm(backend) :
            ADTypes.NoColoringAlgorithm()
        coloring isa ADTypes.NoColoringAlgorithm &&
            (coloring = SparseMatrixColorings.GreedyColoringAlgorithm())
        return ADTypes.AutoSparse(
            ADTypes.dense_ad(backend);
            sparsity_detector = ADTypes.KnownJacobianSparsityDetector(proto),
            coloring_algorithm = coloring,
        )
    end
    backend isa ADTypes.AutoSparse &&
        ADTypes.sparsity_detector(backend) isa ADTypes.NoSparsityDetector &&
        return ADTypes.dense_ad(backend)
    return backend
end

# Counts the evaluations a Jacobian makes, which the statistics include, as
# OrdinaryDiffEq's do.
struct Counted{F}
    f::F
    n::Base.RefValue{Int}
end

(c::Counted)(args...) = (c.n[] += 1; c.f(args...))

struct ADJacobian{F, B, P}
    f!::F
    backend::B
    prep::P
    du::Vector{Float64}
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

# `gamma * dG/du' + dG/du` is the derivative of `v -> G(du + gamma * (v - u), v)` at
# `v = u`, so one Jacobian of that is the whole matrix PETSc wants.
function _shifted_residual!(r, v, g!, du, u, gamma, p, t, w)
    @. w = du + gamma * (v - u)
    g!(r, w, v, p, t)
    return nothing
end

struct ADDAEJacobian{G, B, P}
    g!::G
    backend::B
    prep::P
    r::Vector{Float64}
    w::Vector{Float64}
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
        DI.Constant(copy(u0)), DI.Constant(1.0), DI.Constant(p), DI.Constant(t),
        DI.Cache(w),
    )
    return ADDAEJacobian(h!, b, prep, r, w, advice)
end

# The adjoint's parameter Jacobian, `df/dp`, in the form a user's `paramjac` takes.
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

# A chunk size picked for the state can exceed the number of parameters.
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

# A function written for Float64 alone fails on the first dual number it is handed, as an
# argument it has no method for, a type assertion, or a conversion.
_has_dual(x) = x isa ForwardDiff.Dual || (x isa Type && x <: ForwardDiff.Dual) ||
    (x isa AbstractArray && eltype(x) <: ForwardDiff.Dual)

function _dual_failure(e)
    e isa MethodError && any(_has_dual, e.args) && return true
    e isa TypeError && _has_dual(e.got) && return true
    return occursin("ForwardDiff.Dual", sprint(showerror, e))
end

_dual_error(e, backend, advice) = ArgumentError(
    "`$(ADTypes.dense_ad(backend))` could not differentiate the problem's function: " *
        first(split(sprint(showerror, e), '\n')) * ". " * advice,
)

# A derivative that is infinite where the function is not, such as that of `sqrt` or
# `norm` at zero, would reach PETSc's Newton matrix as NaN. Where the function is not
# finite either, the step fails as it would with any Jacobian. The dual pass's own value
# can be NaN where the function is not, so the function is evaluated again to tell.
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

# What each Jacobian's caller can do instead when differentiating fails.
const _ODE_ADVICE = "Give the ODEFunction a `jac`, or pass " *
    "`autodiff = PETScDiffEq.AutoFiniteDiff()` to the algorithm to have PETSc difference it"
const _DAE_ADVICE = "Give the DAEFunction a `jac`, or pass " *
    "`autodiff = PETScDiffEq.AutoFiniteDiff()` to the algorithm to have PETSc difference it"
const _ADJOINT_JAC_ADVICE = "PETScAdjoint has no finite-difference fallback, so give the " *
    "ODEFunction a `jac`"
const _ADJOINT_PARAMJAC_ADVICE = "PETScAdjoint has no finite-difference fallback, so give " *
    "the ODEFunction a `paramjac`"
