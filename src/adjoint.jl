"""
    PETScAdjoint(; petsc_options = String[])

PETSc's own discrete adjoint, `TSAdjointSolve`, as a `sensealg` for SciMLSensitivity's
`adjoint_sensitivities`:

```julia
using PETScDiffEq, SciMLSensitivity
sol = solve(prob, TSRK("4"); dt = 0.01, adaptive = false)
du0, dp = adjoint_sensitivities(
    sol, TSRK("4"); sensealg = PETScAdjoint(),
    t = ts, dgdu_discrete = dg!, dt = 0.01, adaptive = false,
)
```

The result is the gradient of the solution PETSc computes at these steps, not of the exact
solution, so it agrees with finite differences of the same fixed-step `solve`. PETSc runs
the forward solve again, saving a trajectory, and then the adjoint. Only the problem is
taken from `sol`, so the keywords that set the steps (`dt`, `adaptive`, `abstol`,
`reltol`, `dtmin`, `dtmax`, `maxiters`) must be repeated exactly as `solve` was given
them; otherwise the gradient belongs to a different discretization and nothing says so.
Saving keywords are ignored, and `callback`, `tstops`, `d_discontinuities` and
`save_idxs` are refused, whether given here or to the problem.

Supported: an `ODEProblem` without a mass matrix, in place or out of place, solved with
`TSRK` of any subtype, `TSImplicit("beuler")` or `TSImplicit("cn")`, in either time
direction. The `ODEFunction`'s `jac` and `paramjac` are used when given, and otherwise
built as the forward solve builds a missing `jac`: with the algorithm's `autodiff`, and
with ForwardDiff for a `TSRK`. Under `AutoFiniteDiff()` both have to be given, since
PETSc's own differences never reach its adjoint. A sparse `jac_prototype` is used.
`paramjac` takes the form `jac` does, `paramjac(pJ, u, p, t)` in place or
`paramjac(u, p, t)` out of place, with a row per state and a column per entry of `p`,
which therefore has to be a vector of real numbers. A hand-written `jac` or `paramjac`
goes into the gradient unchecked, so a wrong one gives a wrong gradient without an error;
compare it against a gradient computed without it.

The adjoint runs in PETSc's double real build. A `Float32` problem is solved there in
`Float64`, so `jac`, `paramjac` and the cost functions are handed `Float64` states, and
`du0` and `dp` come back as `Float32` where `u0` and `p` are. Its cost times are matched to
the steps at single precision, so the times its own `solve` saved are accepted with `dt`
repeated as that `solve` was given it. A complex state is refused.

Costs are discrete: at each `t[i]`, `dgdu_discrete(out, u, p, t, i)` writes the cost's
derivative with respect to the state and `dgdp_discrete(out, u, p, t, i)`, if given, its
direct derivative with respect to `p`; `no_start = true` leaves out `t[1]`. PETSc's
adjoint has no derivative of interpolation, so with fixed steps every cost time must be a
time the solve steps to, such as `tspan[1]` plus a multiple of `dt`. An adaptive solve
lands only on the ends of `tspan`, so its costs are limited to those, and its gradient
treats the accepted step sizes as constants rather than differentiating the controller.

`petsc_options` apply to this adjoint's own run and are parsed after the algorithm's. The
trajectory is kept in memory, every stage of every step; `-ts_trajectory_solution_only 1`
keeps only the states and recomputes each step during the adjoint, and
`-ts_trajectory_type basic` writes one file per step to the working directory instead.
For an adaptive solve the memory trajectory reserves 8 bytes for each of `maxiters` steps
before starting, 8 MB at the default and 8 GB at `maxiters = 10^9`. `TSImplicit` solves
transposed linear systems with the same Krylov solver and tolerances as its Newton steps,
so with default options the gradient can be off by up to about their relative tolerance
of 1e-5 while the forward states are far closer; pass `-ksp_type preonly -pc_type lu`, or
a tighter `-ksp_rtol`, when that matters. Options PETSc reads only while the adjoint
runs, such as `-ts_trajectory_view` and `-ts_adjoint_view_solution`, have no effect in
`petsc_options`, though they do when set globally, for example through `PETSC_OPTIONS`.

With an algorithm whose `comm` is not `MPI.COMM_SELF`, the adjoint runs distributed as the
solve does, for the same three families. The `ODEFunction` then needs `jac`, filling this
rank's rows of a sparse `jac_prototype` with global columns, and `paramjac` when there are
parameters, filling this rank's rows; both are collective like `f`. `dgdu_discrete` gets this
rank's rows of the state and writes their derivative, and `dgdp_discrete` gives this rank's
share of the direct derivative, which the ranks add up. `du0` holds this rank's rows, and `dp`
is the whole gradient on every rank. The cost times, `no_start`, the length of `p` and whether
`dgdp_discrete` is given must agree across the ranks.

Returns `(du0, dp')`, where `dp` is `nothing` when `p` is `nothing` or
`SciMLBase.NullParameters()`. Integral costs, and differentiating `solve` itself with a
reverse-mode AD package, are not supported.
"""
struct PETScAdjoint <: SciMLBase.AbstractAdjointSensitivityAlgorithm{0, false, Val{:central}}
    petsc_options::Vector{String}
end

PETScAdjoint(; petsc_options::AbstractVector{<:AbstractString} = String[]) =
    PETScAdjoint(String[String(o) for o in petsc_options])

mutable struct AdjointContext{T, P, JAC, JBUF, PJAC, DG}
    petsclib::T
    tdir::Float64
    p::P
    jac!::JAC
    J::JBUF
    row_cols0::Vector{Vector{LibPETSc.PetscInt}}
    row_src::Vector{Vector{Int}}
    row_buf::Vector{Vector{Float64}}
    paramjac!::PJAC
    pJ::Matrix{Float64}
    pscale::Float64
    dgdu!::DG
    no_start::Bool
    cost_t::Vector{Float64}
    cost_s::Vector{Float64}
    order::Vector{Int}
    next::Int
    s_prev::Float64
    clock::DataType
    cost_at_step::Dict{Int, Vector{Int}}
    u_at_step::Dict{Int, Vector{Float64}}
    u::Vector{Float64}
    g::Vector{Float64}
    work::Vector{Float64}
    lam_buf::Vector{Float64}
    mu_buf::Vector{Float64}
    lamarr::Vector{LibPETSc.CVec}
    muarr::Vector{LibPETSc.CVec}
    jac_mat::Any
    pmat::Any
    lam::Any
    mu::Any
    err::Any
    coo::Union{Nothing, COOJacobian{Float64}}
    comm::Union{Nothing, MPI.Comm}
end

# A throwing rank fills NaN and keeps making PETSc's collective calls until the ranks agree.
function _adjoint_call!(f, adj, out)
    adj.comm === nothing && return f()
    try
        f()
    catch e
        adj.err === nothing && (adj.err = e)
        fill!(out, NaN)
    end
    return nothing
end

_load_jacobian!(::AdjointContext{<:Any, <:Any, <:Any, Matrix{Float64}}, A) = nothing

function _load_jacobian!(adj::AdjointContext{<:Any, <:Any, <:Any, <:SparseMatrixCSC}, A)
    if adj.coo !== nothing
        coo, J = adj.coo, adj.J.nzval
        @inbounds for k in eachindex(coo.vals)
            coo.vals[k] = coo.src[k] == 0 ? 0.0 : J[coo.src[k]]
        end
        LibPETSc.MatSetValuesCOO(adj.petsclib, A, coo.vals, LibPETSc.INSERT_VALUES)
        return nothing
    end
    n = length(adj.u)
    @inbounds for i in 1:n
        src = adj.row_src[i]
        buf = adj.row_buf[i]
        for k in eachindex(src)
            buf[k] = src[k] == 0 ? 0.0 : adj.J.nzval[src[k]]
        end
    end
    _setrows!(adj, A, n)
    return nothing
end

function _adjoint_rhsjacobian!(
        ::LibPETSc.CTS,
        s::Float64,
        x_ptr::LibPETSc.CVec,
        A_ptr::LibPETSc.CMat,
        ::LibPETSc.CMat,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    adj = unsafe_pointer_to_objref(ctx_ptr)::AdjointContext
    return _adjoint_rhsjacobian_body!(adj, s, x_ptr, A_ptr)
end

function _adjoint_rhsjacobian_body!(adj, s, x_ptr, A_ptr)
    pl = adj.petsclib
    try
        _readvec!(adj.u, pl, PETSc.VecPtr(pl, x_ptr, false))
        _adjoint_call!(() -> adj.jac!(adj.J, adj.u, adj.p, s), adj, _stored_values(adj.J))
        A = LibPETSc.PetscMat(A_ptr, pl)
        _load_jacobian!(adj, A)
        PETSc.assemble!(A)
    catch e
        adj.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const ADJ_RHSJACOBIAN_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _adjoint_rhsjacobianp!(
        ::LibPETSc.CTS,
        s::Float64,
        x_ptr::LibPETSc.CVec,
        A_ptr::LibPETSc.CMat,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    adj = unsafe_pointer_to_objref(ctx_ptr)::AdjointContext
    return _adjoint_paramjac_body!(adj, s, x_ptr, A_ptr)
end

function _adjoint_ijacobianp!(
        ::LibPETSc.CTS,
        s::Float64,
        x_ptr::LibPETSc.CVec,
        ::LibPETSc.CVec,
        ::Float64,
        A_ptr::LibPETSc.CMat,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    adj = unsafe_pointer_to_objref(ctx_ptr)::AdjointContext
    return _adjoint_paramjac_body!(adj, s, x_ptr, A_ptr)
end

# Explicit methods take tdir * f_p, implicit ones (dv/ds - tdir * f = 0) take -tdir * f_p.
function _adjoint_paramjac_body!(adj, s, x_ptr, A_ptr)
    pl = adj.petsclib
    try
        _readvec!(adj.u, pl, PETSc.VecPtr(pl, x_ptr, false))
        _adjoint_call!(adj, adj.pJ) do
            adj.paramjac!(adj.pJ, adj.u, adj.p, _user_t(adj.tdir, s))
        end
        LinearAlgebra.rmul!(adj.pJ, adj.pscale)
        PETSc.assemble!(LibPETSc.PetscMat(A_ptr, pl))
    catch e
        adj.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const ADJ_RHSJACOBIANP_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const ADJ_IJACOBIANP_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _adjoint_record!(
        ::LibPETSc.CTS,
        step::LibPETSc.PetscInt,
        s::Float64,
        x_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    adj = unsafe_pointer_to_objref(ctx_ptr)::AdjointContext
    return _adjoint_record_body!(adj, Int(step), Float64(s), x_ptr)
end

function _adjoint_record_body!(adj, step, s, x_ptr)
    pl = adj.petsclib
    try
        R = adj.clock
        tol = max(
            sqrt(Float64(eps(R))) * (s - adj.s_prev),
            (step + 100) * Float64(eps(R(max(1.0, abs(s))))),
        )
        adj.s_prev = s
        order, cost_s = adj.order, adj.cost_s
        while adj.next <= length(order) && cost_s[order[adj.next]] <= s + tol
            k = order[adj.next]
            if abs(cost_s[k] - s) <= tol
                push!(get!(Vector{Int}, adj.cost_at_step, step), k)
                haskey(adj.u_at_step, step) || (
                    adj.u_at_step[step] =
                        _readvec!(similar(adj.u), pl, PETSc.VecPtr(pl, x_ptr, false))
                )
            end
            adj.next += 1
        end
    catch e
        adj.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const ADJ_RECORD_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _adjoint_jump!(
        ::LibPETSc.CTS,
        step::LibPETSc.PetscInt,
        ::Float64,
        ::LibPETSc.CVec,
        ::LibPETSc.PetscInt,
        ::Ptr{LibPETSc.CVec},
        ::Ptr{LibPETSc.CVec},
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    adj = unsafe_pointer_to_objref(ctx_ptr)::AdjointContext
    return _adjoint_jump_body!(adj, Int(step))
end

# A memory trajectory gives this monitor a stale state (and time at step 0), so key by step.
function _adjoint_jump_body!(adj, step)
    ks = get(adj.cost_at_step, step, nothing)
    ks === nothing && return LibPETSc.PetscErrorCode(0)
    pl = adj.petsclib
    try
        _readvec!(adj.work, pl, adj.lam)
        for k in ks
            adj.no_start && k == 1 && continue
            copyto!(adj.u, adj.u_at_step[step])
            fill!(adj.g, 0.0)
            _adjoint_call!(() -> adj.dgdu!(adj.g, adj.u, adj.p, adj.cost_t[k], k), adj, adj.g)
            adj.work .+= adj.g
        end
        _writevec!(pl, adj.lam, adj.work)
    catch e
        adj.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const ADJ_JUMP_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _init_adjoint_pointers!()
    ADJ_RHSJACOBIAN_PTR[] = @cfunction(
        _adjoint_rhsjacobian!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, Float64, LibPETSc.CVec, LibPETSc.CMat,
            LibPETSc.CMat, Ptr{Cvoid},
        )
    )
    ADJ_RHSJACOBIANP_PTR[] = @cfunction(
        _adjoint_rhsjacobianp!,
        LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Float64, LibPETSc.CVec, LibPETSc.CMat, Ptr{Cvoid})
    )
    ADJ_IJACOBIANP_PTR[] = @cfunction(
        _adjoint_ijacobianp!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, Float64, LibPETSc.CVec, LibPETSc.CVec,
            Float64, LibPETSc.CMat, Ptr{Cvoid},
        )
    )
    ADJ_RECORD_PTR[] = @cfunction(
        _adjoint_record!,
        LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, LibPETSc.PetscInt, Float64, LibPETSc.CVec, Ptr{Cvoid})
    )
    ADJ_JUMP_PTR[] = @cfunction(
        _adjoint_jump!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscInt, Float64, LibPETSc.CVec,
            LibPETSc.PetscInt, Ptr{LibPETSc.CVec}, Ptr{LibPETSc.CVec}, Ptr{Cvoid},
        )
    )
    return nothing
end

# PETSc.jl's wrappers lose ctx, free arrays PETSc keeps, or drop results, so ccall these.
function _check_code(code, name)
    iszero(code) || throw(ErrorException("$name failed with $code"))
    return nothing
end

function _exact_final_time(petsclib, ts)
    opt = Ref{LibPETSc.TSExactFinalTimeOption}()
    code = ccall(
        _symbol(petsclib, :TSGetExactFinalTime), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Ptr{LibPETSc.TSExactFinalTimeOption}), ts, opt,
    )
    _check_code(code, "TSGetExactFinalTime")
    return opt[]
end

function _trajectory_type(petsclib, ts)
    tj = Ref{Ptr{Cvoid}}(C_NULL)
    code = ccall(
        _symbol(petsclib, :TSGetTrajectory), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Ptr{Ptr{Cvoid}}), ts, tj,
    )
    _check_code(code, "TSGetTrajectory")
    tj[] == C_NULL && return "none"
    name = Ref{Ptr{Cchar}}(C_NULL)
    code = ccall(
        _symbol(petsclib, :TSTrajectoryGetType), LibPETSc.PetscErrorCode,
        (Ptr{Cvoid}, LibPETSc.CTS, Ptr{Ptr{Cchar}}), tj[], ts, name,
    )
    _check_code(code, "TSTrajectoryGetType")
    return name[] == C_NULL ? "none" : unsafe_string(name[])
end

const _ADJOINT_TYPES = "TSRK, TSImplicit(\"beuler\") or TSImplicit(\"cn\")"

_adjoint_unsupported(::TSRK) = nothing
_adjoint_unsupported(::TSGeneric) = nothing
function _adjoint_unsupported(alg::TSImplicit)
    alg.subtype in ("beuler", "cn") && return nothing
    alg.subtype == "theta" && return "PETScAdjoint has not been verified on " *
        "TSImplicit(\"theta\"); use TSImplicit(\"cn\") for theta = 0.5 or " *
        "TSImplicit(\"beuler\") for theta = 1"
    return "PETSc has no adjoint for TSImplicit(\"$(alg.subtype)\"); use $_ADJOINT_TYPES"
end
_adjoint_unsupported(::TSARKIMEX) = "PETScAdjoint does not support TSARKIMEX: PETSc " *
    "checks its adjoint's identity mass matrix only in debug builds and leaves the " *
    "explicit part's Jacobian out of a split problem; use $_ADJOINT_TYPES"
_adjoint_unsupported(alg::AnyPETScTS) =
    "PETSc has no adjoint for $(nameof(typeof(alg))); use $_ADJOINT_TYPES"

const _ADJOINT_REFUSED_KWARGS = (
    callback = "PETScAdjoint does not support callbacks: this package applies them " *
        "between PETSc steps, where the saved trajectory does not see what they change; " *
        "remove the callback from the call and from the problem",
    tstops = "PETScAdjoint does not support `tstops`: a solve with stops is stepped " *
        "outside PETSc's own loop, which saves no trajectory; remove them and choose a " *
        "`dt` whose steps land on the cost times",
    d_discontinuities = "PETScAdjoint does not support `d_discontinuities`; remove them",
    isoutofdomain = "PETScAdjoint does not support `isoutofdomain`: its forward solve is " *
        "PETSc's own, which does not take a step again that leaves the domain; remove it",
    save_idxs = "PETScAdjoint does not support `save_idxs`: `dgdu_discrete` is given " *
        "the whole state; remove it",
)

const _ADJOINT_OWNED_KWARGS =
    (:saveat, :save_everystep, :save_start, :save_end, :dense, :extra_options, :sensealg)

_names_option(opt, name) =
    startswith(opt, "-") && lowercase(first(split(opt[2:end], "="))) == name

_unset(v) = _no_callback(v) || ((v isa Tuple || v isa AbstractArray) && isempty(v))

function _adjoint_solve_kwargs(prob, kwargs)
    given = hasproperty(prob, :kwargs) ? values(prob.kwargs) : NamedTuple()
    call = values(kwargs)
    merged = merge(given, call)
    for (key, why) in pairs(_ADJOINT_REFUSED_KWARGS)
        # `solve` merges the problem's callback with the call's. Other keywords override.
        vals = key === :callback ? (get(given, key, nothing), get(call, key, nothing)) :
            (get(merged, key, nothing),)
        all(_unset, vals) || throw(ArgumentError(why))
    end
    dropped = (keys(_ADJOINT_REFUSED_KWARGS)..., _ADJOINT_OWNED_KWARGS...)
    return Base.structdiff(merged, NamedTuple{dropped})
end

function _check_adjoint_problem(prob, alg, sensealg, t, dgdu_discrete, dgdp_discrete, comm)
    prob.f isa SciMLBase.DynamicalODEFunction && throw(
        ArgumentError(
            "PETScAdjoint does not support a DynamicalODEProblem or SecondOrderODEProblem",
        ),
    )
    (prob isa SciMLBase.AbstractODEProblem && !(prob.f isa SciMLBase.SplitFunction)) ||
        throw(
        ArgumentError(
            "PETScAdjoint supports an ODEProblem, not a DAEProblem or SplitODEProblem",
        ),
    )
    eltype(prob.u0) <: Real || throw(
        ArgumentError(
            "PETScAdjoint supports a real state only; it runs in PETSc's double real " *
                "build, which cannot hold a $(eltype(prob.u0)) one",
        ),
    )
    why = _adjoint_unsupported(alg)
    why === nothing || throw(ArgumentError(why))
    any(
        o -> _names_option(o, "ts_adjoint_solve"),
        vcat(alg.petsc_options, sensealg.petsc_options),
    ) && throw(
        ArgumentError(
            "`-ts_adjoint_solve` makes PETSc start the adjoint inside the forward solve, " *
                "before PETScAdjoint has set up the cost; remove it from petsc_options",
        ),
    )
    mm = prob.f.mass_matrix
    mm === nothing || mm == LinearAlgebra.I || throw(
        ArgumentError(
            "PETScAdjoint does not support a mass matrix; PETSc's Crank-Nicolson adjoint " *
                "assumes a constant one and none of these methods has been verified with one",
        ),
    )
    comm === nothing || prob.f.jac !== nothing || throw(
        ArgumentError(
            "PETScAdjoint needs the ODEFunction's `jac` $_NOT_SELF, since automatic " *
                "differentiation would call `f` a different number of times on each rank",
        ),
    )
    differences = _petsc_differences(alg)
    differences && prob.f.jac === nothing && throw(
        ArgumentError(
            "PETScAdjoint needs the ODEFunction's `jac` under `autodiff = AutoFiniteDiff()`: " *
                "PETSc's adjoint step multiplies by the Jacobian it is given and has no " *
                "other source for it",
        ),
    )
    p = prob.p
    has_p = !(p === nothing || p isa SciMLBase.NullParameters)
    has_p && !(p isa AbstractVector{<:Real}) && throw(
        ArgumentError(
            "PETScAdjoint needs `p` to be a vector of real numbers or `nothing`, since " *
                "`paramjac` fills a matrix with a column per entry of `p`; got $(typeof(p))",
        ),
    )
    has_p && !isempty(p) && comm !== nothing && prob.f.paramjac === nothing && throw(
        ArgumentError(
            "PETScAdjoint needs the ODEFunction's `paramjac` $_NOT_SELF when the problem has " *
                "parameters, since automatic differentiation would call `f` a different " *
                "number of times on each rank",
        ),
    )
    has_p && !isempty(p) && differences && prob.f.paramjac === nothing && throw(
        ArgumentError(
            "PETScAdjoint needs the ODEFunction's `paramjac` under " *
                "`autodiff = AutoFiniteDiff()` when the problem has parameters: PETSc " *
                "builds the parameter gradient from it",
        ),
    )
    !has_p && dgdp_discrete !== nothing && throw(
        ArgumentError(
            "`dgdp_discrete` was given, but the problem has no parameters to " *
                "differentiate with respect to; leave it out",
        ),
    )
    (_unset(t) || dgdu_discrete === nothing) && throw(
        ArgumentError(
            "PETScAdjoint needs cost times `t` and `dgdu_discrete(out, u, p, t, i)`; " *
                "integral costs are not supported",
        ),
    )
    t isa AbstractVector{<:Real} || throw(
        ArgumentError(
            "PETScAdjoint needs the cost times `t` as a vector of real numbers; got $(typeof(t))",
        ),
    )
    tdir = prob.tspan[1] <= prob.tspan[2] ? 1.0 : -1.0
    s0, sf = tdir * Float64(prob.tspan[1]), tdir * Float64(prob.tspan[2])
    tol = 100 * eps(max(1.0, abs(s0), abs(sf)))
    for ti in t
        s0 - tol <= tdir * ti <= sf + tol ||
            throw(ArgumentError("cost time $ti lies outside tspan = $(prob.tspan)"))
    end
    return has_p
end

function _check_adjoint_ts(h::TSHandles, alg, cost_s)
    pl, ts = h.petsclib, h.ts
    implicit = _uses_ifunction(alg)
    ts_type = LibPETSc.TSGetType(pl, ts)
    ts_type in (implicit ? ("beuler", "cn") : ("rk",)) || throw(
        ArgumentError(
            "PETScAdjoint supports PETSc's rk, beuler and cn as this package drives them, " *
                "but this solve runs `$ts_type`; use $_ADJOINT_TYPES and leave " *
                "`-ts_type` out of petsc_options",
        ),
    )
    if ts_type == "rk" && LibPETSc.TSRKGetMultirate(pl, ts) == LibPETSc.PETSC_TRUE
        throw(
            ArgumentError(
                "PETScAdjoint does not support multirate TSRK: PETSc replaces its " *
                    "forward step but not its adjoint step; remove `-ts_rk_multirate`",
            ),
        )
    end
    _exact_final_time(pl, ts) == LibPETSc.TS_EXACTFINALTIME_MATCHSTEP || throw(
        ArgumentError(
            "PETScAdjoint needs the solve to end on a step at tspan's end, since the " *
                "adjoint starts from the last step saved; remove `-ts_exact_final_time`",
        ),
    )
    traj = _trajectory_type(pl, ts)
    traj in ("memory", "basic") || throw(
        ArgumentError(
            "PETScAdjoint keeps its trajectory in memory, or on disk with " *
                "`-ts_trajectory_type basic`, but this solve's is `$traj`; use one of those",
        ),
    )
    if LibPETSc.TSAdaptGetType(pl, LibPETSc.TSGetAdapt(pl, ts)) != "none"
        tol = 100 * eps(max(1.0, abs(h.t0), abs(h.tf)))
        all(s -> abs(s - h.t0) <= tol || abs(s - h.tf) <= tol, cost_s) || throw(
            ArgumentError(
                "an adaptive solve steps onto no time but tspan's ends, so its cost " *
                    "times can only be those; pass `adaptive = false` and a `dt` whose " *
                    "steps land on every cost time, to `solve` as well as here",
            ),
        )
    end
    return implicit
end

_throw_callback_error(ctx, adj, comm = nothing) =
    _throw_anywhere(comm, ctx.err === nothing ? adj.err : ctx.err)

function _check_agreement(comm, args)
    _everywhere(comm, MPI.bcast(args, 0, comm) == args) && return nothing
    throw(
        ArgumentError(
            "PETScAdjoint $_NOT_SELF needs the same cost times `t`, `no_start`, number of " *
                "parameters and choice of `dgdp_discrete` on every rank",
        ),
    )
end

function _gathered(pl, v, len, comm)
    lo, hi = LibPETSc.VecGetOwnershipRange(pl, v)
    full = zeros(len)
    _readvec!(view(full, (lo + 1):hi), pl, v)
    return MPI.Allreduce(full, +, comm)
end

function _destroy_adjoint!(adj::AdjointContext)
    (PETScCompat.isfinalized(adj.petsclib) || MPI.Finalized()) && return nothing
    for obj in (adj.jac_mat, adj.pmat, adj.lam, adj.mu)
        obj === nothing || PETScCompat.destroy!(obj)
    end
    return nothing
end

const _ADJOINT_TRAJECTORY = ["-ts_save_trajectory", "1", "-ts_trajectory_type", "memory"]

function _discrete_adjoint_unlocked(
        prob, alg::AnyPETScTS, sensealg::PETScAdjoint;
        t = nothing, dgdu_discrete = nothing, dgdp_discrete = nothing, no_start = false,
        kwargs...,
    )
    comm = _distributed(alg) ? alg.comm : nothing
    solve_kwargs, has_p = _checked_everywhere(comm) do
        given = _adjoint_solve_kwargs(prob, kwargs)
        given, _check_adjoint_problem(prob, alg, sensealg, t, dgdu_discrete, dgdp_discrete, comm)
    end
    p = prob.p
    np = has_p ? length(p) : 0
    cost_t = collect(Float64, t)
    comm === nothing ||
        _check_agreement(comm, (cost_t, Bool(no_start), np, dgdp_discrete === nothing))

    h = _setup(
        prob, alg; solve_kwargs...,
        saveat = Float64[], save_everystep = false, save_start = true, save_end = true,
        dense = false, extra_options = vcat(_ADJOINT_TRAJECTORY, sensealg.petsc_options),
        jac_advice = _ADJOINT_JAC_ADVICE, eltypes = (Float64, Float64, Float64),
    )
    pl, ts, ctx = h.petsclib, h.ts, h.ctx
    n = length(h.u0)
    N = comm === nothing ? n : MPI.Allreduce(n, +, comm)
    adj = nothing
    local du0, dp
    try
        cost_s = h.tdir .* cost_t
        implicit = _check_adjoint_ts(h, alg, cost_s)
        iip = SciMLBase.isinplace(prob)
        backend = something(_autodiff(alg), AutoForwardDiff())
        f_ad = _as_inplace(SciMLBase.unwrapped_f(prob.f.f), iip)
        user_t0 = Float64(prob.tspan[1])
        jac, J = nothing, zeros(0, 0)
        if !implicit
            jac = prob.f.jac === nothing ?
                _ad_jacobian(
                    backend, f_ad, prob.f.jac_prototype, h.u0, p, user_t0, Ref(0),
                    _ADJOINT_JAC_ADVICE,
                ) :
                _as_inplace_jac(prob.f.jac, iip)
            h.tdir < 0 && (jac = _reverse_jac(jac))
            proto = prob.f.jac_prototype
            J = proto isa SparseMatrixCSC ? SparseMatrixCSC{Float64, Int}(proto) : zeros(n, n)
        end
        rows = J isa SparseMatrixCSC && comm === nothing ? _row_structure(J, n) :
            (Vector{LibPETSc.PetscInt}[], Vector{Int}[], Vector{Float64}[])
        coo_rows, coo_cols, coo = if comm === nothing || implicit
            nothing, nothing, nothing
        else
            _coo_structure(J, first(LibPETSc.VecGetOwnershipRange(pl, h.u)), nothing)
        end
        adj = AdjointContext(
            pl, h.tdir, p, jac, J, rows...,
            np == 0 ? nothing : prob.f.paramjac === nothing ?
                _ad_paramjacobian(backend, f_ad, h.u0, p, user_t0, _ADJOINT_PARAMJAC_ADVICE) :
                _as_inplace_jac(prob.f.paramjac, iip),
            zeros(n, np),
            implicit ? -h.tdir : h.tdir, dgdu_discrete, Bool(no_start),
            cost_t, cost_s, sortperm(cost_s), 1, h.t0, first(_eltypes(prob)),
            Dict{Int, Vector{Int}}(), Dict{Int, Vector{Float64}}(),
            zeros(n), zeros(n), zeros(n), zeros(n), zeros(np),
            LibPETSc.CVec[], LibPETSc.CVec[], nothing, nothing, nothing, nothing, nothing,
            coo, comm,
        )
        adjptr = pointer_from_objref(adj)
        GC.@preserve ctx adj begin
            if !implicit && comm !== nothing
                adj.jac_mat = LibPETSc.MatCreate(pl, comm)
                _coo_matrix!(adj.jac_mat, pl, n, N, coo_rows, coo_cols)
            elseif !implicit
                adj.jac_mat = J isa SparseMatrixCSC ?
                    PETScCompat.PetscMat(
                        pl, MPI.COMM_SELF, _jacobian_pattern(J, n); with_arrays = true,
                    ) :
                    PETScCompat.PetscMat(pl, J)
            end
            if !implicit
                code = ccall(
                    _symbol(pl, :TSSetRHSJacobian), LibPETSc.PetscErrorCode,
                    (LibPETSc.CTS, LibPETSc.CMat, LibPETSc.CMat, Ptr{Cvoid}, Ptr{Cvoid}),
                    ts, adj.jac_mat.ptr, adj.jac_mat.ptr, ADJ_RHSJACOBIAN_PTR[], adjptr,
                )
                _check_code(code, "TSSetRHSJacobian")
            end
            if np > 0
                # MPIDENSE stores this rank's rows of every column, column-major, which is adj.pJ.
                adj.pmat = comm === nothing ? PETScCompat.PetscMat(pl, adj.pJ) :
                    LibPETSc.MatCreateDense(
                        pl, comm, LibPETSc.PetscInt(n), LibPETSc.PetscInt(LibPETSc.PETSC_DECIDE),
                        LibPETSc.PetscInt(N), LibPETSc.PetscInt(np), pointer(adj.pJ),
                    )
                name = implicit ? :TSSetIJacobianP : :TSSetRHSJacobianP
                code = ccall(
                    _symbol(pl, name), LibPETSc.PetscErrorCode,
                    (LibPETSc.CTS, LibPETSc.CMat, Ptr{Cvoid}, Ptr{Cvoid}),
                    ts, adj.pmat.ptr,
                    implicit ? ADJ_IJACOBIANP_PTR[] : ADJ_RHSJACOBIANP_PTR[], adjptr,
                )
                _check_code(code, String(name))
                if comm === nothing
                    adj.mu = PETScCompat.PetscVec(pl, adj.mu_buf)
                else
                    adj.mu, left = LibPETSc.MatCreateVecs(pl, adj.pmat)
                    PETScCompat.destroy!(left)
                end
                push!(adj.muarr, adj.mu.ptr)
            end
            adj.lam = comm === nothing ? PETScCompat.PetscVec(pl, adj.lam_buf) :
                LibPETSc.VecCreateMPIWithArray(
                    pl, comm, LibPETSc.PetscInt(1), LibPETSc.PetscInt(n),
                    LibPETSc.PetscInt(LibPETSc.PETSC_DECIDE), adj.lam_buf,
                )
            push!(adj.lamarr, adj.lam.ptr)
            LibPETSc.TSMonitorSet(pl, ts, ADJ_RECORD_PTR[], adjptr)
            code = ccall(
                _symbol(pl, :TSAdjointMonitorSet), LibPETSc.PetscErrorCode,
                (LibPETSc.CTS, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                ts, ADJ_JUMP_PTR[], adjptr, C_NULL,
            )
            _check_code(code, "TSAdjointMonitorSet")

            try
                _quiet_errors(h) do
                    _with_options(() -> LibPETSc.TSSolve(pl, ts, h.u), h)
                end
            catch
                ctx.err === nothing && adj.err === nothing && rethrow()
            end
            # The forward solve's post-step reduction has already put an error on every rank.
            _throw_callback_error(ctx, adj)
            stopped_at = _user_t(h.tdir, LibPETSc.TSGetTime(pl, ts))
            ctx.unstable_hit && throw(
                ArgumentError(
                    "the forward solve's `unstable_check` fired at t = $stopped_at, which " *
                        "`solve` reports as ReturnCode.Unstable, so there is no gradient to compute",
                ),
            )
            ctx.dt_too_small && throw(
                ArgumentError(
                    "the forward solve's step fell below `dtmin` at t = $stopped_at, which " *
                        "`solve` reports as ReturnCode.DtLessThanMin, so there is no gradient to " *
                        "compute; lower `dtmin` or tighten the tolerances",
                ),
            )
            reason = LibPETSc.TSGetConvergedReason(pl, ts)
            reason == LibPETSc.TS_CONVERGED_TIME || throw(
                ArgumentError(
                    "the forward solve stopped with $reason at t = " *
                        "$(_user_t(h.tdir, LibPETSc.TSGetSolveTime(pl, ts))), short of " *
                        "tspan's end; raise `maxiters`, in `solve` as well as here, or " *
                        "find out why the steps failed",
                ),
            )
            # PETSc reports TS_CONVERGED_TIME even when the state overflowed.
            _everywhere(comm, all(isfinite, _readvec!(zeros(n), pl, h.u))) || throw(
                ArgumentError(
                    "the forward solve ended on a state that is not finite, which `solve` " *
                        "reports as ReturnCode.Unstable, so there is no gradient to compute; " *
                        "shorten tspan or take smaller steps",
                ),
            )
            matched = falses(length(cost_t))
            for ks in values(adj.cost_at_step), k in ks
                matched[k] = true
            end
            k = findfirst(!, matched)
            k === nothing || throw(
                ArgumentError(
                    "cost time t[$k] = $(cost_t[k]) is not a time the solve stepped to; " *
                        "PETSc's adjoint has no derivative of interpolation, so with fixed " *
                        "steps every cost time must fall on a step, such as tspan[1] plus " *
                        "a multiple of dt",
                ),
            )

            # Set only now, so an adjoint started inside TSSolve fails instead of running
            # before the costs are recorded.
            code = ccall(
                _symbol(pl, :TSSetCostGradients), LibPETSc.PetscErrorCode,
                (LibPETSc.CTS, LibPETSc.PetscInt, Ptr{LibPETSc.CVec}, Ptr{LibPETSc.CVec}),
                ts, LibPETSc.PetscInt(1), pointer(adj.lamarr),
                np > 0 ? pointer(adj.muarr) : Ptr{LibPETSc.CVec}(C_NULL),
            )
            _check_code(code, "TSSetCostGradients")
            try
                _quiet_errors(h) do
                    LibPETSc.TSAdjointSolve(pl, ts)
                end
            catch
                ctx.err === nothing && adj.err === nothing && rethrow()
            end
            _throw_callback_error(ctx, adj, comm)
            reason = LibPETSc.TSGetConvergedReason(pl, ts)
            reason == LibPETSc.TS_CONVERGED_ITS || throw(
                ArgumentError(
                    "PETSc's adjoint solve stopped with $reason; for TSImplicit that is " *
                        "the transposed linear solve, so pass a solver that converges on " *
                        "it, such as `-ksp_type preonly -pc_type lu`, through " *
                        "PETScAdjoint's petsc_options",
                ),
            )
            du0 = _readvec!(zeros(n), pl, adj.lam)
            dp = np == 0 ? zeros(0) : comm === nothing ? _readvec!(zeros(np), pl, adj.mu) :
                _gathered(pl, adj.mu, np, comm)
        end
    finally
        _destroy!(h)
        adj === nothing || _destroy_adjoint!(adj)
    end
    if dgdp_discrete !== nothing
        gp = zeros(np)
        mine = comm === nothing ? dp : zeros(np)
        _checked_everywhere(comm) do
            for (step, ks) in adj.cost_at_step, k in ks
                no_start && k == 1 && continue
                fill!(gp, 0.0)
                dgdp_discrete(gp, copy(adj.u_at_step[step]), p, cost_t[k], k)
                mine .+= gp
            end
        end
        comm === nothing || (dp .+= MPI.Allreduce(mine, +, comm))
    end
    return _like(prob.u0, du0), has_p ? _like(p, dp)' : nothing
end

_like(x, g) = eltype(x) === Float32 ? Float32.(g) : g

_discrete_adjoint(prob, alg::AnyPETScTS, sensealg::PETScAdjoint; kwargs...) =
    _locked(() -> _discrete_adjoint_unlocked(prob, alg, sensealg; kwargs...))

function SciMLBase._concrete_solve_adjoint(
        ::SupportedProblem, ::AnyPETScTS, ::PETScAdjoint, u0, p,
        ::SciMLBase.ADOriginator, args...; kwargs...,
    )
    throw(
        ArgumentError(
            "PETScAdjoint is reached through `adjoint_sensitivities(sol, alg; sensealg = " *
                "PETScAdjoint(), t, dgdu_discrete, ...)`; differentiating `solve` with it " *
                "is not supported, since SciMLSensitivity's reverse-mode `solve` accepts " *
                "only its own adjoint types",
        ),
    )
end
