_check_autodiff(ad::ADTypes.AbstractADType) = ad
_check_autodiff(ad) = throw(
    ArgumentError(
        "`autodiff` takes an ADTypes backend, such as `AutoForwardDiff()` or " *
            "`AutoFiniteDiff()`; got $(repr(ad))",
    ),
)

_autodiff(alg::Union{TSRosW, TSImplicit, TSIRK, TSDAE, TSARKIMEX, TSGeneric}) = alg.autodiff
_autodiff(::Union{TSRK, TSMPRK}) = nothing

function _petsc_differences(alg)
    ad = _autodiff(alg)
    return ad !== nothing && ADTypes.dense_ad(ad) isa AutoFiniteDiff
end

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

function _check_holomorphic(h, v, backend, advice)
    n = length(v)
    n == 0 && return nothing
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
    gap = LinearAlgebra.norm(along_imag .- im .* along_real)
    scale = LinearAlgebra.norm(along_real) + LinearAlgebra.norm(along_imag)
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
