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

# A sparse prototype is the pattern to colour, as OrdinaryDiffEq uses it, unless the
# backend already says how to find one.
function _with_pattern(backend, proto)
    (backend isa ADTypes.AutoSparse || !(proto isa SparseArrays.AbstractSparseMatrix)) &&
        return backend
    return ADTypes.AutoSparse(
        backend;
        sparsity_detector = ADTypes.KnownJacobianSparsityDetector(proto),
        coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm(),
    )
end

struct ADJacobian{F, B, P}
    f!::F
    backend::B
    prep::P
    du::Vector{Float64}
end

function (j::ADJacobian)(J, u, p, t)
    try
        DI.jacobian!(j.f!, j.du, J, j.prep, j.backend, u, DI.Constant(p), DI.Constant(t))
    catch e
        _dual_failure(e) && throw(_dual_error(e, j.backend, "right-hand side", "ODEFunction"))
        rethrow()
    end
    return nothing
end

function _ad_jacobian(backend, f!, jac_prototype, u0, p, t)
    b = _with_pattern(backend, jac_prototype)
    du = similar(u0)
    prep = DI.prepare_jacobian(f!, du, b, copy(u0), DI.Constant(p), DI.Constant(t))
    return ADJacobian(f!, b, prep, du)
end

# `gamma * dG/du' + dG/du` is the derivative of `v -> G(du + gamma * (v - u), v)` at
# `v = u`, so one Jacobian of that is the whole matrix PETSc wants.
function _shifted_residual!(r, v, g!, du, u, gamma, p, t)
    g!(r, du .+ gamma .* (v .- u), v, p, t)
    return nothing
end

struct ADDAEJacobian{G, B, P}
    g!::G
    backend::B
    prep::P
    r::Vector{Float64}
end

function (j::ADDAEJacobian)(J, du, u, p, gamma, t)
    try
        DI.jacobian!(
            _shifted_residual!, j.r, J, j.prep, j.backend, u, DI.Constant(j.g!),
            DI.Constant(du), DI.Constant(u), DI.Constant(gamma), DI.Constant(p),
            DI.Constant(t),
        )
    catch e
        _dual_failure(e) && throw(_dual_error(e, j.backend, "residual", "DAEFunction"))
        rethrow()
    end
    return nothing
end

function _ad_dae_jacobian(backend, g!, jac_prototype, u0, p, t)
    b = _with_pattern(backend, jac_prototype)
    r = similar(u0)
    prep = DI.prepare_jacobian(
        _shifted_residual!, r, b, copy(u0), DI.Constant(g!), DI.Constant(zero(u0)),
        DI.Constant(copy(u0)), DI.Constant(1.0), DI.Constant(p), DI.Constant(t),
    )
    return ADDAEJacobian(g!, b, prep, r)
end

# A function written for Float64 alone fails on the first dual number it is handed.
_dual_failure(e) = e isa MethodError && any(a -> a isa ForwardDiff.Dual, e.args)

_dual_error(e, backend, what, fname) = ArgumentError(
    "the $what could not be differentiated with `$(ADTypes.dense_ad(backend))`, which " *
        "builds the Jacobian when the $fname has no `jac`: " *
        first(split(sprint(showerror, e), '\n')) * ". Give the $fname a `jac`, or pass " *
        "`autodiff = AutoFiniteDiff()` to the algorithm to have PETSc difference it",
)

# The adjoint's parameter Jacobian, `df/dp`, in the form a user's `paramjac` takes.
struct ADParamJacobian{F, B, P}
    f!::F
    backend::B
    prep::P
    du::Vector{Float64}
end

_with_p!(du, p, f!, u, t) = f!(du, u, p, t)

function (j::ADParamJacobian)(pJ, u, p, t)
    try
        DI.jacobian!(
            _with_p!, j.du, pJ, j.prep, j.backend, p, DI.Constant(j.f!), DI.Constant(u),
            DI.Constant(t),
        )
    catch e
        _dual_failure(e) && throw(_dual_error(e, j.backend, "right-hand side", "ODEFunction"))
        rethrow()
    end
    return nothing
end

function _ad_paramjacobian(backend, f!, u0, p, t)
    b = ADTypes.dense_ad(backend)
    du = similar(u0)
    prep = DI.prepare_jacobian(
        _with_p!, du, b, copy(p), DI.Constant(f!), DI.Constant(copy(u0)), DI.Constant(t),
    )
    return ADParamJacobian(f!, b, prep, du)
end
