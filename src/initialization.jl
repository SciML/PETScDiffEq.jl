_rms(r, ::Nothing) = DiffEqBase.ODE_DEFAULT_NORM(r, 0)
_rms(r, comm::MPI.Comm) = _global_norm(comm, r, 0)

_within(r, tol::Number, comm) = _rms(r, comm) <= tol
# A zero `abstol` entry meets a zero residual, where `r / tol` would be NaN.
_within(r, tol, comm) = _rms(map((x, t) -> iszero(x) ? zero(x / t) : x / t, r, tol), comm) <= 1

_zero_rows(M::LinearAlgebra.Diagonal) = iszero.(M.diag)
_zero_rows(M) = [all(iszero, r) for r in eachrow(M)]
_zero_cols(M::LinearAlgebra.Diagonal) = iszero.(M.diag)
_zero_cols(M) = [all(iszero, c) for c in eachcol(M)]
function _zero_lines(M, dim)
    z = trues(size(M, dim))
    for (i, j, v) in zip(findnz(M)...)
        iszero(v) || (z[dim == 1 ? i : j] = false)
    end
    return z
end
_zero_rows(M::SparseArrays.AbstractSparseMatrix) = _zero_lines(M, 1)
_zero_cols(M::SparseArrays.AbstractSparseMatrix) = _zero_lines(M, 2)

_own_zero_cols(vars, mass, n, ::Nothing) = vars
# On a communicator a column is zero only if it is in every rank's rows.
function _own_zero_cols(vars, mass, n, comm::MPI.Comm)
    _anywhere(comm, mass isa SparseArrays.AbstractSparseMatrix) || return vars
    N, rstart = MPI.Allreduce(n, +, comm), MPI.Scan(n, +, comm) - n
    used = zeros(Bool, N)
    if mass isa SparseArrays.AbstractSparseMatrix
        used .= .!vars
    else
        used[rstart .+ (1:n)] .= .!vars
    end
    return .!MPI.Allreduce(used, |, comm)[rstart .+ (1:n)]
end

const _INIT_ALGS = Union{
    SciMLBase.CheckInit, DiffEqBase.BrownFullBasicInit, DiffEqBase.ShampineCollocationInit,
}

_resolve_init(init, f) = init isa DiffEqBase.DefaultInit ?
    (SciMLBase.has_initialization_data(f) ? SciMLBase.OverrideInit() : SciMLBase.CheckInit()) :
    init

_overrides(init, f) = SciMLBase.has_initialization_data(f) &&
    _resolve_init(init, f) isa SciMLBase.OverrideInit

# `valp` is what the problem's initialization hooks read: the problem, or the integrator.
function _initialize!(
        u0, prob, valp, init, f, jac, pl, comm, t0, tf, abstol, reltol, dt, dtmax,
    )
    init = _resolve_init(init, prob.f)
    init isa SciMLBase.NoInit && return prob.p, true
    init isa SciMLBase.OverrideInit &&
        return _override!(u0, prob, valp, init, pl, comm, abstol, reltol)
    return prob.p, _consistent!(u0, prob, init, f, jac, pl, comm, t0, tf, abstol, dt, dtmax)
end

struct PETScSNES{L} <: SciMLBase.AbstractNonlinearAlgorithm
    petsclib::L
end

const _SNES_PROBLEMS = Union{
    SciMLBase.NonlinearProblem, SciMLBase.ImmutableNonlinearProblem,
    SciMLBase.NonlinearLeastSquaresProblem,
}
const _SCC_SOLVER =
    Base.PkgId(Base.UUID("9dfe8606-65a1-4bb3-9748-cb89d1561431"), "SCCNonlinearSolve")
const _PASS_NLSOLVE = "pass a solver as `initializealg = OverrideInit(; nlsolve = ...)`"

function _check_snes_problem(iprob)
    if iprob isa SciMLBase.SCCNonlinearProblem
        haskey(Base.loaded_modules, _SCC_SOLVER) || throw(
            ArgumentError(
                "the problem's initialization is an SCCNonlinearProblem, which PETSc's SNES " *
                    "solves block by block through SCCNonlinearSolve; load that package, or " *
                    _PASS_NLSOLVE,
            ),
        )
        return nothing
    end
    iprob isa _SNES_PROBLEMS &&
        (iprob.u0 === nothing || iprob.u0 isa AbstractVector{<:Number}) || throw(
        ArgumentError(
            "PETSc's SNES cannot solve the problem's initialization, a " *
                "`$(nameof(typeof(iprob)))` with a `$(typeof(iprob.u0))` state; " * _PASS_NLSOLVE,
        ),
    )
    iprob.u0 === nothing && return nothing
    n, proto = length(iprob.u0), iprob.f.resid_prototype
    m = proto === nothing ? n : length(proto)
    m == n || throw(
        ArgumentError(
            "the problem's initialization has $m equations for $n unknowns, and PETSc's " *
                "SNES solves a square system only; $_PASS_NLSOLVE, with a least-squares " *
                "solver from NonlinearSolve",
        ),
    )
    return nothing
end

function _override!(u0, prob, valp, init, pl, comm, abstol, reltol)
    _anywhere(comm, SciMLBase.has_initialization_data(prob.f)) || return prob.p, true
    comm === nothing || throw(
        ArgumentError(
            "PETScDiffEq cannot run `OverrideInit` $_NOT_SELF; start from consistent " *
                "values and pass `initializealg = CheckInit()`",
        ),
    )
    data = prob.f.initialization_data
    nlsolve = init.nlsolve
    if nlsolve === nothing && !SciMLBase.is_trivial_initialization(data)
        _check_snes_problem(data.initializeprob)
        nlsolve = PETScSNES(pl)
    end
    scalar(tol) = tol isa Number ? tol : minimum(tol)
    u, p, ok = SciMLBase.get_initial_values(
        prob, valp, prob.f, init, Val(SciMLBase.isinplace(prob));
        nlsolve_alg = nlsolve, abstol = scalar(abstol), reltol = scalar(reltol),
    )
    copyto!(u0, u)
    return p, ok
end

function SciMLBase.solve(prob::_SNES_PROBLEMS, alg::PETScSNES; abstol, kwargs...)
    f, u0, p = prob.f, prob.u0, prob.p
    function resid(u)
        SciMLBase.isinplace(prob) || return f(u, p)
        proto = f.resid_prototype
        r = proto === nothing ? (u === nothing ? Float64[] : similar(u)) :
            similar(proto, u === nothing ? eltype(proto) : promote_type(eltype(proto), eltype(u)))
        f(r, u, p)
        return r
    end
    if u0 === nothing
        r = resid(nothing)
        code = LinearAlgebra.norm(r) <= abstol ? SciMLBase.ReturnCode.Success :
            SciMLBase.ReturnCode.Failure
        return SciMLBase.build_solution(prob, alg, Float64[], r; retcode = code)
    end
    pl = alg.petsclib
    like(x) = ismutable(u0) ? copyto!(similar(u0), x) : typeof(u0)(x)
    # PETSc's difference step is below a narrower eltype's spacing, so f runs at PETSc's.
    wide(x) = ismutable(u0) ?
        copyto!(similar(u0, promote_type(eltype(u0), pl.PetscScalar)), x) : typeof(u0)(x)
    residual!(out, x) = (copyto!(out, resid(wide(x))); nothing)
    x = Vector{pl.PetscScalar}(u0)
    code = _snes_solve!(x, residual!, nothing, nothing, pl, abstol) ?
        SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.ConvergenceFailure
    u = like(x)
    return SciMLBase.build_solution(prob, alg, u, resid(u); retcode = code)
end

# `f` and `jac` are the user's in-place functions, in user time.
function _consistent!(
        u0::Vector{S}, prob, init, f, jac, pl, comm, t0, tf, abstol, dt, dtmax,
    ) where {S}
    is_dae = prob isa SciMLBase.AbstractDAEProblem
    mass = is_dae ? nothing : prob.f.mass_matrix
    eqs = vars = nothing
    if !is_dae
        plain = mass === nothing || mass == LinearAlgebra.I
        comm === nothing && plain && return true
        # A rank whose own block is the identity still joins the collectives below.
        n = length(u0)
        eqs = plain ? falses(n) : _zero_rows(mass)
        _anywhere(comm, any(eqs)) || return true
        vars = _own_zero_cols(plain ? falses(n) : _zero_cols(mass), mass, n, comm)
        _anywhere(comm, any(vars)) || return true
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
    brown = init isa DiffEqBase.BrownFullBasicInit
    if comm !== nothing
        atol = _smallest(comm, Float64(tol isa Number ? tol : minimum(tol; init = Inf)))
        h = brown ? nothing : _collocation_step(init, is_dae, t0, tf, dt, dtmax)
        return _consistent_on!(comm, u0, du0, prob, f, jac, pl, t0, atol, h, eqs, vars)
    end
    proto = prob.f.jac_prototype
    sp = jac === nothing ? proto isa SparseArrays.AbstractSparseMatrix :
        proto isa SparseMatrixCSC
    pattern = sp ? SparseMatrixCSC(proto) : nothing
    atol = tol isa Number ? tol : minimum(tol)
    if brown
        is_dae && return _brown_dae!(u0, du0, prob, f, jac, pl, t0, atol, pattern)
        return _brown_mass!(u0, prob, f, jac, pl, t0, atol, pattern, eqs, vars)
    end
    h = _collocation_step(init, is_dae, t0, tf, dt, dtmax)
    return _shampine!(u0, prob, f, jac, pl, t0, atol, pattern, h)
end

function _collocation_step(init, is_dae, t0, tf, dt, dtmax)
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
    return oftype(t0, step)
end

const _NO_DIFFERENTIAL_VARS = "BrownFullBasicInit needs the DAEProblem's `differential_vars` " *
    "to know which variables are algebraic; set them, start from consistent values, or use " *
    "another `initializealg`"

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
    dv === nothing && throw(ArgumentError(_NO_DIFFERENTIAL_VARS))
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

_init_dm(f::Ghosted) = f.dm
_init_dm(f) = nothing

function _global_flags(comm, flags, rstart, N)
    whole = zeros(Bool, N)
    whole[rstart .+ (1:length(flags))] .= flags
    return MPI.Allreduce(whole, |, comm)
end

# A row Brown's algorithm does not solve pins its variable, keeping the state's layout.
function _consistent_on!(
        comm, u0::Vector{S}, du0, prob, f, jac, pl, t0, atol, h, eqs, vars,
    ) where {S}
    is_dae, brown = du0 !== nothing, h === nothing
    p, n, dm = prob.p, length(u0), _init_dm(f)
    dv = nothing
    if brown && is_dae
        dv = _checked_everywhere(comm) do
            prob.differential_vars === nothing && throw(ArgumentError(_NO_DIFFERENTIAL_VARS))
            Vector{Bool}(vec(prob.differential_vars))
        end
    elseif brown
        _everywhere(comm, eqs == vars) || throw(
            ArgumentError(
                "BrownFullBasicInit $_NOT_SELF solves each zero row of the mass matrix for " *
                    "the variable of the same index, so its zero columns have to be its zero " *
                    "rows; start from consistent values, which `CheckInit()` checks, or use " *
                    "`ShampineCollocationInit()`",
            ),
        )
    end
    N, rstart = MPI.Allreduce(n, +, comm), MPI.Scan(n, +, comm) - n
    collocation = !(is_dae || brown)
    mass = collocation ? prob.f.mass_matrix : nothing
    own = !(mass === nothing || mass == LinearAlgebra.I)
    sparse_mass = collocation && dm === nothing &&
        _anywhere(comm, own && mass isa SparseArrays.AbstractSparseMatrix)
    M = if !collocation
        nothing
    elseif sparse_mass
        _distributed_mass(S, own ? mass : nothing, n, N, comm)
    else
        LinearAlgebra.Diagonal(own ? Vector{S}(mass.diag) : ones(S, n))
    end
    use_jac = dm === nothing && _everywhere(comm, jac !== nothing)
    dvg = use_jac && is_dae && brown ? _global_flags(comm, dv, rstart, N) : nothing
    x = brown && is_dae ? ifelse.(dv, du0, u0) : copy(u0)
    a, b, start = similar(u0), similar(u0), copy(u0)
    R = real(S)
    clone = xv = rv = A = mm = mv = snes = opts = nothing
    try
        if dm === nothing
            xv = _state_vec(pl, comm, n)
            J = _structure(S, SparseMatrixCSC(prob.f.jac_prototype))
            rows, cols, coo = _coo_structure(J, rstart, M)
            A = LibPETSc.MatCreate(pl, comm)
            # PETSc's MPIAIJ preallocation rewrites the index arrays it is given.
            _coo_matrix!(A, pl, n, N, copy(rows), copy(cols))
            if !use_jac
                LibPETSc.MatSetValuesCOO(pl, A, coo.vals, LibPETSc.INSERT_VALUES)
                PETSc.assemble!(A)
            end
        else
            clone = LibPETSc.PetscDM(_dm_vec!(pl, :DMClone, dm)[], pl)
            xv = _plain_mdot(() -> LibPETSc.DMCreateGlobalVector(pl, clone), pl)
            A = LibPETSc.DMCreateMatrix(pl, clone)
        end
        rv = LibPETSc.VecDuplicate(pl, xv)
        if sparse_mass
            mm = LibPETSc.MatCreate(pl, comm)
            _fill_mass!(mm, pl, M, rstart, N)
            mv = (LibPETSc.VecDuplicate(pl, xv), LibPETSc.VecDuplicate(pl, xv))
        end
        residual! = if brown && is_dae
            function (out, x)
                @. a = ifelse(dv, x, du0)
                @. b = ifelse(dv, u0, x)
                f(out, a, b, p, t0)
                return nothing
            end
        elseif brown
            (out, x) -> (f(a, x, p, t0); @. out = ifelse(eqs, a, x - u0); nothing)
        elseif is_dae
            (out, x) -> (@. a = (x - start) / h; f(out, a, x, p, t0); nothing)
        else
            function (out, x)
                @. a = (x - start) / h
                if sparse_mass
                    # MatMult is collective, so it comes before `f`, which may throw.
                    _writevec!(pl, mv[1], a)
                    _mat_mult!(pl, mm, mv[1].ptr, mv[2].ptr)
                    _readvec!(out, pl, mv[2])
                else
                    @. out = M.diag * a
                end
                f(b, x, p, t0)
                out .-= b
                return nothing
            end
        end
        values! = nothing
        if use_jac
            src = coo.src
            entry = (J, k) -> src[k] == 0 ? zero(S) : J.nzval[src[k]]
            values! = if brown && is_dae
                J1, deriv = copy(J), [dvg[c + 1] for c in cols]
                function (vals, x)
                    @. a = ifelse(dv, x, du0)
                    @. b = ifelse(dv, u0, x)
                    jac(J, a, b, p, zero(t0), t0)
                    jac(J1, a, b, p, one(t0), t0)
                    for k in eachindex(vals)
                        vals[k] = deriv[k] ? entry(J1, k) - entry(J, k) : entry(J, k)
                    end
                    return nothing
                end
            elseif brown
                keep = [eqs[r - rstart + 1] for r in rows]
                function (vals, x)
                    jac(J, x, p, t0)
                    for k in eachindex(vals)
                        vals[k] = keep[k] ? entry(J, k) : S(rows[k] == cols[k])
                    end
                    return nothing
                end
            elseif is_dae
                function (vals, x)
                    @. a = (x - start) / h
                    jac(J, a, x, p, inv(h), t0)
                    for k in eachindex(vals)
                        vals[k] = entry(J, k)
                    end
                    return nothing
                end
            else
                function (vals, x)
                    jac(J, x, p, t0)
                    for k in eachindex(vals)
                        vals[k] = coo.mass[k] / h - entry(J, k)
                    end
                    return nothing
                end
            end
        end
        s = InitNewton(
            pl, residual!, values!, similar(x), similar(x), LibPETSc.PetscInt[], nothing, comm,
            use_jac ? coo.vals : S[],
        )
        snes = LibPETSc.SNESCreate(pl, comm)
        options = [
            "-pc_factor_nonzeros_along_diagonal", "-sub_pc_factor_nonzeros_along_diagonal",
            "-ksp_rtol", _option(sqrt(eps(R))),
        ]
        dm === nothing || _everywhere(comm, _dm_colours(pl, clone.ptr)) ||
            push!(options, "-snes_fd_color_use_mat")
        opts = PETScCompat.PetscOptions(pl; PETSc.parse_options(options)...)
        GC.@preserve s begin
            ctxptr = pointer_from_objref(s)
            dm === nothing || _check_code(
                ccall(
                    _symbol(pl, :SNESSetDM), LibPETSc.PetscErrorCode,
                    (LibPETSc.CSNES, Ptr{Cvoid}), snes.ptr, clone.ptr,
                ),
            )
            LibPETSc.SNESSetFunction(pl, snes, rv, INIT_RESIDUAL_PTR[], ctxptr)
            if use_jac
                LibPETSc.SNESSetJacobian(pl, snes, A, A, INIT_JACOBIAN_PTR[], ctxptr)
            else
                LibPETSc.SNESSetJacobian(
                    pl, snes, A, A, _symbol(pl, :SNESComputeJacobianDefaultColor), C_NULL,
                )
            end
            push!(opts)
            try
                _check_code(
                    ccall(
                        _symbol(pl, :SNESSetFromOptions), LibPETSc.PetscErrorCode,
                        (LibPETSc.CSNES,), snes.ptr,
                    ),
                )
                _snes_tolerances!(pl, snes, R, atol, 1.0e-8, 50)
                _newton!(x, pl, snes, xv, s) || return false
            finally
                pop!(opts)
            end
        end
    finally
        snes === nothing || LibPETSc.SNESDestroy(pl, snes)
        opts === nothing || PETScCompat.destroy!(opts)
        mv === nothing || foreach(PETScCompat.destroy!, mv)
        for obj in (mm, rv, A, xv)
            obj === nothing || PETScCompat.destroy!(obj)
        end
        clone === nothing || _check_code(
            ccall(
                _symbol(pl, :DMDestroy), LibPETSc.PetscErrorCode, (Ptr{Ptr{Cvoid}},),
                Ref(clone.ptr),
            ),
        )
    end
    if brown && is_dae
        @. u0 = ifelse(dv, u0, x)
    elseif brown
        @. u0 = ifelse(eqs, x, u0)
    else
        copyto!(u0, x)
    end
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
    comm::Union{Nothing, MPI.Comm}
    vals::Vector{S}
end

# A rank whose function throws keeps making the collective calls, on NaN until the ranks agree.
function _init_call!(f, s::InitNewton, out)
    s.comm === nothing && return f()
    try
        f()
    catch e
        s.err === nothing && (s.err = e)
        fill!(out, NaN)
    end
    return nothing
end

function _init_residual!(
        ::LibPETSc.CSNES, x_ptr::LibPETSc.CVec, f_ptr::LibPETSc.CVec, ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    s = unsafe_pointer_to_objref(ctx_ptr)::InitNewton
    pl = s.petsclib
    try
        _readvec!(s.x, pl, PETSc.VecPtr(pl, x_ptr, false))
        _init_call!(() -> s.residual!(s.r, s.x), s, s.r)
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
        if s.comm === nothing
            _set_matrix!(pl, B, s.jacobian(s.x), s.idx)
        else
            _init_call!(() -> s.jacobian(s.vals, s.x), s, s.vals)
            LibPETSc.MatSetValuesCOO(pl, B, s.vals, LibPETSc.INSERT_VALUES)
        end
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
    s.comm === nothing || return _snes_run_everywhere!(pl, snes, xv, s)
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

function _snes_run_everywhere!(pl, snes, xv, s)
    failed, err = false, nothing
    try
        _quiet_errors(() -> LibPETSc.SNESSolve(pl, snes, C_NULL, xv), pl, false)
    catch e
        failed = e isa LibPETSc.PetscError && e.code in (PETSC_ERR_MAT_LU_ZRPVT, PETSC_ERR_FP)
        failed || (err = e)
    end
    err = something(s.err, err, Some(nothing))
    threw, failed = MPI.Allreduce([err !== nothing, failed], |, s.comm)
    threw && throw(something(err, _remote_error()))
    return !failed
end

_snes_tolerances!(pl, snes, ::Type{R}, atol, stol, maxit) where {R} =
    LibPETSc.SNESSetTolerances(
    pl, snes, R(atol), zero(R), R(stol), LibPETSc.PetscInt(maxit),
    LibPETSc.PetscInt(10000),
)

function _newton!(x, pl, snes, xv, s)
    _writevec!(pl, xv, x)
    _snes_run!(pl, snes, xv, s) || return false
    Int(LibPETSc.SNESGetConvergedReason(pl, snes)) > 0 || return false
    _readvec!(x, pl, xv)
    # PETSc's steppers take a residual left at the start as error no step size removes.
    fnorm = LibPETSc.SNESGetFunctionNorm(pl, snes)
    if fnorm > 0
        _snes_tolerances!(pl, snes, real(eltype(x)), 0, 0, 1)
        _snes_run!(pl, snes, xv, s) &&
            LibPETSc.SNESGetFunctionNorm(pl, snes) < fnorm && _readvec!(x, pl, xv)
    end
    return true
end

function _snes_solve!(x::Vector{S}, residual!, jacobian, pattern, pl, atol) where {S}
    m, R = length(x), real(S)
    s = InitNewton(
        pl, residual!, jacobian, similar(x), similar(x),
        LibPETSc.PetscInt[i - 1 for i in 1:m], nothing, nothing, S[],
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
            _newton!(x, pl, snes, xv, s) || return false
        end
    finally
        LibPETSc.SNESDestroy(pl, snes)
        PETScCompat.destroy!(A)
        PETScCompat.destroy!(xv)
        PETScCompat.destroy!(rv)
    end
    return true
end
