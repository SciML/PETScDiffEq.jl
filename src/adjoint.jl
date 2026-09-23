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
end

_load_jacobian!(::AdjointContext{<:Any, <:Any, <:Any, Matrix{Float64}}, A) = nothing

# PETSc's sparse matrix holds the prototype's pattern plus the diagonal, row by row.
function _load_jacobian!(adj::AdjointContext{<:Any, <:Any, <:Any, <:SparseMatrixCSC}, A)
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
        s::LibPETSc.PetscReal,
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
        adj.jac!(adj.J, adj.u, adj.p, s)
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
        s::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        A_ptr::LibPETSc.CMat,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    adj = unsafe_pointer_to_objref(ctx_ptr)::AdjointContext
    return _adjoint_paramjac_body!(adj, s, x_ptr, A_ptr)
end

function _adjoint_ijacobianp!(
        ::LibPETSc.CTS,
        s::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        ::LibPETSc.CVec,
        ::LibPETSc.PetscReal,
        A_ptr::LibPETSc.CMat,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    adj = unsafe_pointer_to_objref(ctx_ptr)::AdjointContext
    return _adjoint_paramjac_body!(adj, s, x_ptr, A_ptr)
end

# PETSc steps dv/ds = tdir * f(v, p, tdir * s), so an explicit method wants tdir * f_p
# and the residual dv/ds - tdir * f an implicit one solves has derivative -tdir * f_p.
function _adjoint_paramjac_body!(adj, s, x_ptr, A_ptr)
    pl = adj.petsclib
    try
        _readvec!(adj.u, pl, PETSc.VecPtr(pl, x_ptr, false))
        adj.paramjac!(adj.pJ, adj.u, adj.p, _user_t(adj.tdir, s))
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
        s::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    adj = unsafe_pointer_to_objref(ctx_ptr)::AdjointContext
    return _adjoint_record_body!(adj, Int(step), Float64(s), x_ptr)
end

# Cost times are taken in order and each only once, so the steps a memory trajectory
# recomputes during the adjoint, which it reports to the monitors again, record nothing.
function _adjoint_record_body!(adj, step, s, x_ptr)
    pl = adj.petsclib
    try
        # Steps are the step just taken apart, so a cost time within a small fraction of
        # it belongs to this step and cannot be nearer the previous one. PETSc adds each
        # step to its time, which rounds by up to an ulp per step, so the allowance grows
        # with the step count.
        tol = max(
            sqrt(eps(Float64)) * (s - adj.s_prev), (step + 100) * eps(max(1.0, abs(s))),
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
        ::LibPETSc.PetscReal,
        ::LibPETSc.CVec,
        ::LibPETSc.PetscInt,
        ::Ptr{LibPETSc.CVec},
        ::Ptr{LibPETSc.CVec},
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    adj = unsafe_pointer_to_objref(ctx_ptr)::AdjointContext
    return _adjoint_jump_body!(adj, Int(step))
end

# A memory trajectory reloads the stages for each adjoint step but not the state the
# monitor is handed, and at step 0 not the time either, so the cost is found by step
# number and evaluated on the state recorded going forward.
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
            adj.dgdu!(adj.g, adj.u, adj.p, adj.cost_t[k], k)
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

# Called from `__init__`, since `@cfunction` needs these functions defined where it appears.
function _init_adjoint_pointers!()
    ADJ_RHSJACOBIAN_PTR[] = @cfunction(
        _adjoint_rhsjacobian!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CMat,
            LibPETSc.CMat, Ptr{Cvoid},
        )
    )
    ADJ_RHSJACOBIANP_PTR[] = @cfunction(
        _adjoint_rhsjacobianp!,
        LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CMat, Ptr{Cvoid})
    )
    ADJ_IJACOBIANP_PTR[] = @cfunction(
        _adjoint_ijacobianp!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec,
            LibPETSc.PetscReal, LibPETSc.CMat, Ptr{Cvoid},
        )
    )
    ADJ_RECORD_PTR[] = @cfunction(
        _adjoint_record!,
        LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, LibPETSc.PetscInt, LibPETSc.PetscReal, LibPETSc.CVec, Ptr{Cvoid})
    )
    ADJ_JUMP_PTR[] = @cfunction(
        _adjoint_jump!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscInt, LibPETSc.PetscReal, LibPETSc.CVec,
            LibPETSc.PetscInt, Ptr{LibPETSc.CVec}, Ptr{LibPETSc.CVec}, Ptr{Cvoid},
        )
    )
    return nothing
end

# PETSc.jl's wrappers for these pin the callback context to `nothing`, hand PETSc an array
# that lives only for the call when PETSc keeps a pointer to it, or drop the value they
# fetch, so the symbols are called directly.
const ADJOINT_SYMBOLS = Dict{Symbol, Ptr{Cvoid}}()

_adjoint_symbol(petsclib, name::Symbol) = get!(ADJOINT_SYMBOLS, name) do
    Libdl.dlsym(Libdl.dlopen(petsclib.petsc_library), name)
end

function _check_code(code, name)
    iszero(code) || throw(ErrorException("$name failed with $code"))
    return nothing
end

function _exact_final_time(petsclib, ts)
    opt = Ref{LibPETSc.TSExactFinalTimeOption}()
    code = ccall(
        _adjoint_symbol(petsclib, :TSGetExactFinalTime), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Ptr{LibPETSc.TSExactFinalTimeOption}), ts, opt,
    )
    _check_code(code, "TSGetExactFinalTime")
    return opt[]
end

function _trajectory_type(petsclib, ts)
    tj = Ref{Ptr{Cvoid}}(C_NULL)
    code = ccall(
        _adjoint_symbol(petsclib, :TSGetTrajectory), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Ptr{Ptr{Cvoid}}), ts, tj,
    )
    _check_code(code, "TSGetTrajectory")
    tj[] == C_NULL && return "none"
    name = Ref{Ptr{Cchar}}(C_NULL)
    code = ccall(
        _adjoint_symbol(petsclib, :TSTrajectoryGetType), LibPETSc.PetscErrorCode,
        (Ptr{Cvoid}, LibPETSc.CTS, Ptr{Ptr{Cchar}}), tj[], ts, name,
    )
    _check_code(code, "TSTrajectoryGetType")
    return name[] == C_NULL ? "none" : unsafe_string(name[])
end

const _ADJOINT_TYPES = "TSRK, TSImplicit(\"beuler\") or TSImplicit(\"cn\")"

_adjoint_unsupported(::TSRK) = nothing
# The PETSc type is known only once the options are applied, so it is checked then.
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

# Saving and `sensealg` do not change the steps a solve takes, and the adjoint does its
# own saving.
const _ADJOINT_OWNED_KWARGS =
    (:saveat, :save_everystep, :save_start, :save_end, :dense, :extra_options, :sensealg)

# PETSc matches option names without regard to case, and a value can follow `=`.
_names_option(opt, name) =
    startswith(opt, "-") && lowercase(first(split(opt[2:end], "="))) == name

_unset(v) = _no_callback(v) || ((v isa Tuple || v isa AbstractArray) && isempty(v))

function _adjoint_solve_kwargs(prob, kwargs)
    given = hasproperty(prob, :kwargs) ? values(prob.kwargs) : NamedTuple()
    call = values(kwargs)
    merged = merge(given, call)
    for (key, why) in pairs(_ADJOINT_REFUSED_KWARGS)
        # `solve` combines a callback on the problem with one given to the call, where
        # any other keyword given to the call replaces the problem's.
        vals = key === :callback ? (get(given, key, nothing), get(call, key, nothing)) :
            (get(merged, key, nothing),)
        all(_unset, vals) || throw(ArgumentError(why))
    end
    dropped = (keys(_ADJOINT_REFUSED_KWARGS)..., _ADJOINT_OWNED_KWARGS...)
    return Base.structdiff(merged, NamedTuple{dropped})
end

function _check_adjoint_problem(prob, alg, sensealg, t, dgdu_discrete, dgdp_discrete)
    (prob isa SciMLBase.AbstractODEProblem && !(prob.f isa SciMLBase.SplitFunction)) ||
        throw(
        ArgumentError(
            "PETScAdjoint supports an ODEProblem, not a DAEProblem or SplitODEProblem",
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
    # PETSc's own differences never reach the adjoint, which multiplies by the Jacobian it
    # is given, so under `AutoFiniteDiff()` both Jacobians have to be the caller's.
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

function _throw_callback_error(ctx, adj)
    ctx.err === nothing || throw(ctx.err)
    adj.err === nothing || throw(adj.err)
    return nothing
end

function _destroy_adjoint!(adj::AdjointContext)
    (PETSc.finalized(adj.petsclib) || MPI.Finalized()) && return nothing
    for obj in (adj.jac_mat, adj.pmat, adj.lam, adj.mu)
        obj === nothing || PETSc.destroy(obj)
    end
    return nothing
end

const _ADJOINT_TRAJECTORY = ["-ts_save_trajectory", "1", "-ts_trajectory_type", "memory"]

function _discrete_adjoint_unlocked(
        prob, alg::AnyPETScTS, sensealg::PETScAdjoint;
        t = nothing, dgdu_discrete = nothing, dgdp_discrete = nothing, no_start = false,
        kwargs...,
    )
    solve_kwargs = _adjoint_solve_kwargs(prob, kwargs)
    has_p = _check_adjoint_problem(prob, alg, sensealg, t, dgdu_discrete, dgdp_discrete)
    p = prob.p
    np = has_p ? length(p) : 0
    cost_t = collect(Float64, t)

    h = _setup(
        prob, alg; solve_kwargs...,
        saveat = Float64[], save_everystep = false, save_start = true, save_end = true,
        dense = false, extra_options = vcat(_ADJOINT_TRAJECTORY, sensealg.petsc_options),
        jac_advice = _ADJOINT_JAC_ADVICE,
    )
    pl, ts, ctx = h.petsclib, h.ts, h.ctx
    n = length(h.u0)
    adj = nothing
    local du0, dp
    try
        cost_s = h.tdir .* cost_t
        implicit = _check_adjoint_ts(h, alg, cost_s)
        iip = SciMLBase.isinplace(prob)
        # A TSRK has no `autodiff` of its own and differentiates with ForwardDiff.
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
        rows = J isa SparseMatrixCSC ? _row_structure(J, n) :
            (Vector{LibPETSc.PetscInt}[], Vector{Int}[], Vector{Float64}[])
        adj = AdjointContext(
            pl, h.tdir, p, jac, J, rows...,
            np == 0 ? nothing : prob.f.paramjac === nothing ?
                _ad_paramjacobian(backend, f_ad, h.u0, p, user_t0, _ADJOINT_PARAMJAC_ADVICE) :
                _as_inplace_jac(prob.f.paramjac, iip),
            zeros(n, np),
            implicit ? -h.tdir : h.tdir, dgdu_discrete, Bool(no_start),
            cost_t, cost_s, sortperm(cost_s), 1, h.t0,
            Dict{Int, Vector{Int}}(), Dict{Int, Vector{Float64}}(),
            zeros(n), zeros(n), zeros(n), zeros(n), zeros(np),
            LibPETSc.CVec[], LibPETSc.CVec[], nothing, nothing, nothing, nothing, nothing,
        )
        adjptr = pointer_from_objref(adj)
        GC.@preserve ctx adj begin
            if !implicit
                adj.jac_mat = J isa SparseMatrixCSC ?
                    PETSc.MatSeqAIJWithArrays(pl, MPI.COMM_SELF, _jacobian_pattern(J, n)) :
                    PETSc.MatSeqDense(pl, J)
                code = ccall(
                    _adjoint_symbol(pl, :TSSetRHSJacobian), LibPETSc.PetscErrorCode,
                    (LibPETSc.CTS, LibPETSc.CMat, LibPETSc.CMat, Ptr{Cvoid}, Ptr{Cvoid}),
                    ts, adj.jac_mat.ptr, adj.jac_mat.ptr, ADJ_RHSJACOBIAN_PTR[], adjptr,
                )
                _check_code(code, "TSSetRHSJacobian")
            end
            if np > 0
                adj.pmat = PETSc.MatSeqDense(pl, adj.pJ)
                name = implicit ? :TSSetIJacobianP : :TSSetRHSJacobianP
                code = ccall(
                    _adjoint_symbol(pl, name), LibPETSc.PetscErrorCode,
                    (LibPETSc.CTS, LibPETSc.CMat, Ptr{Cvoid}, Ptr{Cvoid}),
                    ts, adj.pmat.ptr,
                    implicit ? ADJ_IJACOBIANP_PTR[] : ADJ_RHSJACOBIANP_PTR[], adjptr,
                )
                _check_code(code, String(name))
                adj.mu = PETSc.VecSeq(pl, adj.mu_buf)
                push!(adj.muarr, adj.mu.ptr)
            end
            adj.lam = PETSc.VecSeq(pl, adj.lam_buf)
            push!(adj.lamarr, adj.lam.ptr)
            LibPETSc.TSMonitorSet(pl, ts, ADJ_RECORD_PTR[], adjptr)
            code = ccall(
                _adjoint_symbol(pl, :TSAdjointMonitorSet), LibPETSc.PetscErrorCode,
                (LibPETSc.CTS, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                ts, ADJ_JUMP_PTR[], adjptr, C_NULL,
            )
            _check_code(code, "TSAdjointMonitorSet")

            try
                _quiet_errors(h) do
                    LibPETSc.TSSolve(pl, ts, h.u)
                end
            catch
                ctx.err === nothing && adj.err === nothing && rethrow()
            end
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
            # PETSc reports reaching the end even when the steps overflowed on the way.
            all(isfinite, _readvec!(zeros(n), pl, h.u)) || throw(
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

            # Given only now, so that an adjoint started from inside the forward solve
            # fails PETSc's own check for cost gradients instead of running before the
            # costs are recorded.
            code = ccall(
                _adjoint_symbol(pl, :TSSetCostGradients), LibPETSc.PetscErrorCode,
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
            _throw_callback_error(ctx, adj)
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
            dp = np > 0 ? _readvec!(zeros(np), pl, adj.mu) : zeros(0)
        end
    finally
        _destroy!(h)
        adj === nothing || _destroy_adjoint!(adj)
    end
    if dgdp_discrete !== nothing
        gp = zeros(np)
        for (step, ks) in adj.cost_at_step, k in ks
            no_start && k == 1 && continue
            fill!(gp, 0.0)
            dgdp_discrete(gp, copy(adj.u_at_step[step]), p, cost_t[k], k)
            dp .+= gp
        end
    end
    return du0, has_p ? dp' : nothing
end

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
