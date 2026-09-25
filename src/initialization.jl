_rms(r, ::Nothing) = DiffEqBase.ODE_DEFAULT_NORM(r, 0)
_rms(r, comm::MPI.Comm) = _global_norm(comm, r, 0)

_within(r, tol::Number, comm) = _rms(r, comm) <= tol
_within(r, tol, comm) = _rms(r ./ tol, comm) <= 1

_zero_rows(M::LinearAlgebra.Diagonal) = iszero.(M.diag)
_zero_rows(M) = [all(iszero, r) for r in eachrow(M)]
_zero_cols(M::LinearAlgebra.Diagonal) = iszero.(M.diag)
_zero_cols(M) = [all(iszero, c) for c in eachcol(M)]

const _INIT_ALGS = Union{
    SciMLBase.CheckInit, DiffEqBase.BrownFullBasicInit, DiffEqBase.ShampineCollocationInit,
}

# `f` and `jac` are the user's in-place functions, in user time.
function _initialize!(
        u0::Vector{S}, prob, init, f, jac, pl, comm, t0, tf, abstol, dt, dtmax,
    ) where {S}
    init isa SciMLBase.NoInit && return true
    if init isa SciMLBase.OverrideInit
        SciMLBase.has_initialization_data(prob.f) && throw(
            ArgumentError(
                "PETScDiffEq does not solve a problem's initialization system, so it " *
                    "cannot run `OverrideInit`",
            ),
        )
        return true
    end
    init isa DiffEqBase.DefaultInit && (init = SciMLBase.CheckInit())
    is_dae = prob isa SciMLBase.AbstractDAEProblem
    mass = is_dae ? nothing : prob.f.mass_matrix
    eqs = vars = nothing
    if !is_dae
        plain = mass === nothing || mass == LinearAlgebra.I
        comm === nothing && plain && return true
        # A rank whose own block is the identity still joins the collectives below.
        eqs = plain ? falses(length(u0)) : _zero_rows(mass)
        vars = plain ? falses(length(u0)) : _zero_cols(mass)
        _anywhere(comm, any(eqs)) && _anywhere(comm, any(vars)) || return true
    end
    init isa _INIT_ALGS || throw(
        ArgumentError("PETScDiffEq does not support `initializealg = $(repr(init))`"),
    )
    name = nameof(typeof(init))
    init isa SciMLBase.CheckInit || init.nlsolve === nothing || throw(
        ArgumentError("PETScDiffEq solves `$name` with PETSc's SNES and takes no `nlsolve`"),
    )
    du0 = is_dae ? Vector{S}(vec(prob.du0)) : nothing
    r = similar(u0)
    _checked_everywhere(comm) do
        is_dae ? f(r, du0, u0, prob.p, t0) : f(r, u0, prob.p, t0)
        is_dae || (r[.!eqs] .= 0)
        return nothing
    end
    tol = init isa DiffEqBase.BrownFullBasicInit ? init.abstol : abstol
    _within(r, tol, comm) && return true
    init isa SciMLBase.CheckInit &&
        throw(SciMLBase.CheckInitFailureError(_rms(r, comm), abstol, !is_dae))
    comm === nothing || throw(
        ArgumentError(
            "PETScDiffEq cannot run `$name` $_NOT_SELF; start from consistent values, " *
                "which `CheckInit()` checks",
        ),
    )
    proto = prob.f.jac_prototype
    sp = jac === nothing ? proto isa SparseArrays.AbstractSparseMatrix :
        proto isa SparseMatrixCSC
    pattern = sp ? SparseMatrixCSC(proto) : nothing
    atol = tol isa Number ? tol : minimum(tol)
    if init isa DiffEqBase.BrownFullBasicInit
        is_dae && return _brown_dae!(u0, du0, prob, f, jac, pl, t0, atol, pattern)
        return _brown_mass!(u0, prob, f, jac, pl, t0, atol, pattern, eqs, vars)
    end
    span = abs(tf - t0)
    top = dtmax === nothing || isinf(dtmax) ? span : abs(dtmax)
    step = if init.initdt !== nothing
        init.initdt
    elseif is_dae
        sign(tf - t0) * (iszero(t0) ? top / 10 : min(abs(t0) / 1000, top / 10))
    elseif dt !== nothing
        sign(tf - t0) * min(abs(dt) / 5, top)
    else
        (tf - t0) / 1000
    end
    return _shampine!(u0, prob, f, jac, pl, t0, atol, pattern, oftype(t0, step))
end

_jac_buffer(S, pattern, n) = pattern === nothing ? zeros(S, n, n) : _structure(S, pattern)
_or_nothing(jac, jacobian) = jac === nothing ? nothing : jacobian

function _brown_mass!(u0::Vector{S}, prob, f, jac, pl, t0, atol, pattern, eqs, vars) where {S}
    rows, cols = findall(eqs), findall(vars)
    length(rows) == length(cols) || throw(
        ArgumentError(
            "BrownFullBasicInit needs as many algebraic equations as algebraic variables, " *
                "but the mass matrix has $(length(rows)) zero rows and $(length(cols)) zero " *
                "columns",
        ),
    )
    p, u, du = prob.p, copy(u0), similar(u0)
    function residual!(out, x)
        u[cols] .= x
        f(du, u, p, t0)
        out .= @view du[rows]
        return nothing
    end
    J = _jac_buffer(S, pattern, length(u0))
    function jacobian(x)
        u[cols] .= x
        jac(J, u, p, t0)
        return J[rows, cols]
    end
    sub = pattern === nothing ? nothing : _jacobian_pattern(pattern[rows, cols], length(rows))
    x = u0[cols]
    _snes_solve!(x, residual!, _or_nothing(jac, jacobian), sub, pl, atol) || return false
    u0[cols] .= x
    return true
end

function _brown_dae!(u0::Vector{S}, du0, prob, f, jac, pl, t0, atol, pattern) where {S}
    dv = prob.differential_vars
    dv === nothing && throw(
        ArgumentError(
            "BrownFullBasicInit needs the DAEProblem's `differential_vars` to know which " *
                "variables are algebraic; set them, start from consistent values, or use " *
                "another `initializealg`",
        ),
    )
    p, n = prob.p, length(u0)
    du, u = similar(u0), similar(u0)
    function split!(x)
        @. du = ifelse(dv, x, du0)
        @. u = ifelse(dv, u0, x)
        return nothing
    end
    residual!(out, x) = (split!(x); f(out, du, u, p, t0); nothing)
    J0, J1 = _jac_buffer(S, pattern, n), _jac_buffer(S, pattern, n)
    diff = LinearAlgebra.Diagonal(S.(dv))
    function jacobian(x)
        split!(x)
        jac(J0, du, u, p, zero(t0), t0)
        jac(J1, du, u, p, one(t0), t0)
        return (J1 - J0) * diff + J0 * (LinearAlgebra.I - diff)
    end
    x = ifelse.(dv, du0, u0)
    full = pattern === nothing ? nothing : _jacobian_pattern(pattern, n)
    _snes_solve!(x, residual!, _or_nothing(jac, jacobian), full, pl, atol) || return false
    @. u0 = ifelse(dv, u0, x)
    return true
end

function _shampine!(u0::Vector{S}, prob, f, jac, pl, t0, atol, pattern, h) where {S}
    p, n = prob.p, length(u0)
    is_dae = prob isa SciMLBase.AbstractDAEProblem
    M = is_dae ? nothing : Matrix{S}(prob.f.mass_matrix)
    start, slope, fx = copy(u0), similar(u0), similar(u0)
    function residual!(out, x)
        @. slope = (x - start) / h
        if is_dae
            f(out, slope, x, p, t0)
        else
            f(fx, x, p, t0)
            mul!(out, M, slope)
            out .-= fx
        end
        return nothing
    end
    J = _jac_buffer(S, pattern, n)
    shifted = is_dae ? nothing : (pattern === nothing ? M : SparseMatrixCSC(M)) ./ h
    function jacobian(x)
        @. slope = (x - start) / h
        is_dae && (jac(J, slope, x, p, inv(h), t0); return J)
        jac(J, x, p, t0)
        return shifted - J
    end
    full = pattern === nothing ? nothing : _jacobian_pattern(pattern, n, M)
    x = copy(u0)
    _snes_solve!(x, residual!, _or_nothing(jac, jacobian), full, pl, atol) || return false
    copyto!(u0, x)
    return true
end

mutable struct InitNewton{L, S}
    petsclib::L
    residual!::Any
    jacobian::Any
    x::Vector{S}
    r::Vector{S}
    idx::Vector{LibPETSc.PetscInt}
    err::Any
end

function _init_residual!(
        ::LibPETSc.CSNES, x_ptr::LibPETSc.CVec, f_ptr::LibPETSc.CVec, ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    s = unsafe_pointer_to_objref(ctx_ptr)::InitNewton
    pl = s.petsclib
    try
        _readvec!(s.x, pl, PETSc.VecPtr(pl, x_ptr, false))
        s.residual!(s.r, s.x)
        _writevec!(pl, PETSc.VecPtr(pl, f_ptr, false), s.r)
    catch e
        s.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

function _init_jacobian!(
        ::LibPETSc.CSNES, x_ptr::LibPETSc.CVec, A_ptr::LibPETSc.CMat, B_ptr::LibPETSc.CMat,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    s = unsafe_pointer_to_objref(ctx_ptr)::InitNewton
    pl = s.petsclib
    A = LibPETSc.PetscMat(A_ptr, pl)
    B = LibPETSc.PetscMat(B_ptr, pl)
    try
        _readvec!(s.x, pl, PETSc.VecPtr(pl, x_ptr, false))
        _set_matrix!(pl, B, s.jacobian(s.x), s.idx)
        PETSc.assemble!(B)
        B.ptr == A.ptr || PETSc.assemble!(A)
    catch e
        s.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const INIT_RESIDUAL_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const INIT_JACOBIAN_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _init_initialization_pointers!()
    INIT_RESIDUAL_PTR[] = @cfunction(
        _init_residual!, LibPETSc.PetscErrorCode,
        (LibPETSc.CSNES, LibPETSc.CVec, LibPETSc.CVec, Ptr{Cvoid})
    )
    INIT_JACOBIAN_PTR[] = @cfunction(
        _init_jacobian!, LibPETSc.PetscErrorCode,
        (LibPETSc.CSNES, LibPETSc.CVec, LibPETSc.CMat, LibPETSc.CMat, Ptr{Cvoid})
    )
    return nothing
end

# MatSetValues reads the block row-major.
function _set_matrix!(pl, B, J::AbstractMatrix, idx)
    m = LibPETSc.PetscInt(length(idx))
    return LibPETSc.MatSetValues(
        pl, B, m, idx, m, idx, vec(permutedims(Matrix(J))), LibPETSc.INSERT_VALUES,
    )
end

function _set_matrix!(pl, B, J::SparseMatrixCSC, idx)
    LibPETSc.MatZeroEntries(pl, B)
    rows, vals = rowvals(J), nonzeros(J)
    for j in axes(J, 2)
        r = nzrange(J, j)
        isempty(r) && continue
        LibPETSc.MatSetValues(
            pl, B, LibPETSc.PetscInt(length(r)), LibPETSc.PetscInt[rows[k] - 1 for k in r],
            LibPETSc.PetscInt(1), idx[j:j], vals[r], LibPETSc.INSERT_VALUES,
        )
    end
    return nothing
end

function _direct_solver!(pl, snes, sparse)
    ksp, pc = Ref{LibPETSc.CKSP}(C_NULL), Ref{Ptr{Cvoid}}(C_NULL)
    _check_code(
        ccall(
            _symbol(pl, :SNESGetKSP), LibPETSc.PetscErrorCode,
            (LibPETSc.CSNES, Ptr{LibPETSc.CKSP}), snes.ptr, ksp,
        ), "SNESGetKSP",
    )
    _check_code(
        ccall(
            _symbol(pl, :KSPSetType), LibPETSc.PetscErrorCode, (LibPETSc.CKSP, Cstring),
            ksp[], "preonly",
        ), "KSPSetType",
    )
    _check_code(
        ccall(
            _symbol(pl, :KSPGetPC), LibPETSc.PetscErrorCode,
            (LibPETSc.CKSP, Ptr{Ptr{Cvoid}}), ksp[], pc,
        ), "KSPGetPC",
    )
    _check_code(
        ccall(
            _symbol(pl, :PCSetType), LibPETSc.PetscErrorCode, (Ptr{Cvoid}, Cstring), pc[],
            "lu",
        ), "PCSetType",
    )
    # PETSc's sparse LU does not pivot, and an algebraic row can have a zero diagonal.
    sparse && _reorder_for_diagonal(pl, pc[], pl.PetscReal)
    return nothing
end

for R in (Float32, Float64)
    @eval _reorder_for_diagonal(pl, pc, ::Type{$R}) = _check_code(
        ccall(
            _symbol(pl, :PCFactorReorderForNonzeroDiagonal), LibPETSc.PetscErrorCode,
            (Ptr{Cvoid}, $R), pc, $R(1.0e-10),
        ), "PCFactorReorderForNonzeroDiagonal",
    )
end

function _snes_run!(pl, snes, xv, s)
    try
        _quiet_errors(() -> LibPETSc.SNESSolve(pl, snes, C_NULL, xv), pl, false)
    catch e
        s.err === nothing || throw(s.err)
        e isa LibPETSc.PetscError && e.code in (PETSC_ERR_MAT_LU_ZRPVT, PETSC_ERR_FP) ||
            rethrow()
        return false
    end
    s.err === nothing || throw(s.err)
    return true
end

_snes_tolerances!(pl, snes, ::Type{R}, atol, stol, maxit) where {R} =
    LibPETSc.SNESSetTolerances(
    pl, snes, R(atol), zero(R), R(stol), LibPETSc.PetscInt(maxit),
    LibPETSc.PetscInt(10000),
)

function _snes_solve!(x::Vector{S}, residual!, jacobian, pattern, pl, atol) where {S}
    m, R = length(x), real(S)
    s = InitNewton(
        pl, residual!, jacobian, similar(x), similar(x),
        LibPETSc.PetscInt[i - 1 for i in 1:m], nothing,
    )
    xv, rv = PETScCompat.PetscVec(pl, m), PETScCompat.PetscVec(pl, m)
    A = pattern === nothing ? PETScCompat.PetscMat(pl, zeros(S, m, m)) :
        PETScCompat.PetscMat(pl, MPI.COMM_SELF, pattern; with_arrays = true)
    snes = LibPETSc.SNESCreate(pl, MPI.COMM_SELF)
    try
        GC.@preserve s begin
            ctxptr = pointer_from_objref(s)
            LibPETSc.SNESSetFunction(pl, snes, rv, INIT_RESIDUAL_PTR[], ctxptr)
            if jacobian === nothing
                fd = pattern === nothing ? :SNESComputeJacobianDefault :
                    :SNESComputeJacobianDefaultColor
                LibPETSc.SNESSetJacobian(pl, snes, A, A, _symbol(pl, fd), C_NULL)
            else
                LibPETSc.SNESSetJacobian(pl, snes, A, A, INIT_JACOBIAN_PTR[], ctxptr)
            end
            _snes_tolerances!(pl, snes, R, atol, 1.0e-8, 50)
            _direct_solver!(pl, snes, pattern !== nothing)
            _writevec!(pl, xv, x)
            _snes_run!(pl, snes, xv, s) || return false
            Int(LibPETSc.SNESGetConvergedReason(pl, snes)) > 0 || return false
            _readvec!(x, pl, xv)
            # PETSc's steppers take a residual left at the start as error no step size removes.
            fnorm = LibPETSc.SNESGetFunctionNorm(pl, snes)
            if fnorm > 0
                _snes_tolerances!(pl, snes, R, 0, 0, 1)
                _snes_run!(pl, snes, xv, s) &&
                    LibPETSc.SNESGetFunctionNorm(pl, snes) < fnorm && _readvec!(x, pl, xv)
            end
        end
    finally
        LibPETSc.SNESDestroy(pl, snes)
        PETScCompat.destroy!(A)
        PETScCompat.destroy!(xv)
        PETScCompat.destroy!(rv)
    end
    return true
end
