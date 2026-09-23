module PETScDiffEq

using ADTypes: ADTypes, AutoFiniteDiff, AutoForwardDiff
using DiffEqBase: DiffEqBase
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using LinearAlgebra: LinearAlgebra, mul!
using MPI: MPI
using PETSc: PETSc
using Libdl: Libdl
using PETSc.LibPETSc: LibPETSc
using SciMLBase: SciMLBase
using SciMLOperators: SciMLOperators
using SparseArrays: SparseArrays, SparseMatrixCSC, findnz, nonzeros, nzrange, rowvals,
    sparse
using SparseMatrixColorings: SparseMatrixColorings

export TSRK, TSRosW, TSImplicit, TSIRK, TSARKIMEX, TSDAE, TSMPRK, TSGeneric,
    PETScIntegrator, PETScAdjoint

abstract type PETScTSAlgorithm <: SciMLBase.AbstractODEAlgorithm end
abstract type PETScTSDAEAlgorithm <: SciMLBase.AbstractDAEAlgorithm end

const AnyPETScTS = Union{PETScTSAlgorithm, PETScTSDAEAlgorithm}

"""
    TSRK(subtype = "5dp", petsc_options = String[])

Explicit Runge-Kutta from PETSc's `TSRK`. `subtype` is a PETSc `TSRKType`
without its prefix, such as `"3bs"`, `"5dp"`, `"5f"` or `"5bs"`.

Adapts on its embedded error estimate, so `reltol` and `abstol` apply. PETSc
gives `"1fe"`, `"2b"`, `"3"` and `"4"` no such estimate, so those step at the
`dt` you give and warn if you pass a tolerance. Being explicit it never forms
a Jacobian and ignores an `ODEFunction`'s `jac`, and it cannot carry a mass
matrix.

`petsc_options` are command-line style tokens passed to PETSc for this solve,
for example `["-ts_adapt_type", "none"]`. They are parsed after the options
this package sets, so they win.
"""
struct TSRK <: PETScTSAlgorithm
    subtype::String
    petsc_options::Vector{String}
end

TSRK(subtype::AbstractString = "5dp", petsc_options::AbstractVector{<:AbstractString} = String[]) =
    TSRK(String(subtype), String[String(o) for o in petsc_options])

"""
    TSRosW(subtype = "ra34pw2", petsc_options = String[]; autodiff = AutoForwardDiff())

Rosenbrock-W from PETSc's `TSROSW`. `subtype` is a PETSc `TSRosWType` without
its prefix, such as `"2m"`, `"ra34pw2"` or `"r34prw"`.

Adapts on its embedded error estimate, except for `"theta1"` and `"theta2"`,
which PETSc gives none, so they step at the `dt` you give and warn if you pass
a tolerance. Linearly implicit, so it uses an `ODEFunction`'s `jac`, and it
accepts a mass matrix.

Without a `jac` the Jacobian comes from `autodiff`: ForwardDiff by default, colouring a
sparse `jac_prototype`, or `AutoFiniteDiff()` to have PETSc difference the step's own
equations, colouring a sparse prototype too.

PETSc's implementation assumes a right-hand side that does not depend on `t`.
When it does, `"2p"`, `"2m"`, `"ra3pw"`, `"ra34pw2"`, `"r34prw"` and `"assp3p3s1c"`
keep their order, and every other type, including all the fourth-order ones,
converges at first order with or without a `jac`. Carrying `t` as an extra state
whose derivative is 1 restores their order.

On a linear right-hand side that does not depend on `t`, `"ra3pw"`'s embedded error
estimate is zero with an exact Jacobian and far too small with a differenced one, so an adaptive solve
reports success with an error well above the tolerance. Step it with a fixed `dt` on
such problems.

`"assp3p3s1c"` needs a Jacobian, which `AutoFiniteDiff()` does not give it, and cannot
take a mass matrix, which PETSc leaves out of its explicit first stage. `"lassp3p4s2c"`, `"llssp3p4s2c"` and `"ark3"` are refused.
They end on an explicit stage: without a Jacobian PETSc stops and asks for one, and with
one it does not restore its Jacobian lag after that stage, so an adaptive solve fails
within its first two steps and a fixed-step solve diverges.
"""
struct TSRosW <: PETScTSAlgorithm
    subtype::String
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
end

TSRosW(
    subtype::AbstractString = "ra34pw2",
    petsc_options::AbstractVector{<:AbstractString} = String[];
    autodiff = AutoForwardDiff(),
) = TSRosW(String(subtype), String[String(o) for o in petsc_options], _check_autodiff(autodiff))

"""
    TSImplicit(subtype = "beuler"; order = nothing, autodiff = AutoForwardDiff())
    TSImplicit(subtype, theta; ...)
    TSImplicit(subtype, [theta,] petsc_options; ...)

Fully implicit methods from PETSc: `"beuler"`, `"cn"`, `"theta"` and `"bdf"`.
`theta` sets the parameter of the theta method, where `0.5` is Crank-Nicolson
and `1.0` is backward Euler.

`order` sets the BDF order, 1 through 6. PETSc's own default is 2, which on a
stiff problem can cost an order of magnitude in steps against a higher-order
method, so raise it when comparing against one.

Only `"bdf"` carries an embedded error estimate and adapts; the others step at
the `dt` you give and warn if you pass a tolerance. All of them use an
`ODEFunction`'s `jac` and accept a mass matrix, which makes a singular mass
matrix an index-1 differential-algebraic problem.

Without a `jac` the Jacobian comes from `autodiff`: ForwardDiff by default, colouring a
sparse `jac_prototype`, or `AutoFiniteDiff()` to have PETSc difference the step's own
equations, colouring a sparse prototype too.
"""
struct TSImplicit <: PETScTSAlgorithm
    subtype::String
    theta::Union{Nothing, Float64}
    order::Union{Nothing, Int}
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
end

function _bdf_order(subtype, order)
    order === nothing && return nothing
    subtype == "bdf" ||
        throw(ArgumentError("`order` applies to TSImplicit(\"bdf\"), not \"$subtype\""))
    1 <= order <= 6 || throw(ArgumentError("PETSc supports BDF orders 1 through 6"))
    return Int(order)
end

TSImplicit(
    subtype::AbstractString = "beuler"; order = nothing, autodiff = AutoForwardDiff(),
) = TSImplicit(
    String(subtype), nothing, _bdf_order(subtype, order), String[], _check_autodiff(autodiff),
)
TSImplicit(
    subtype::AbstractString, theta::Real; order = nothing, autodiff = AutoForwardDiff(),
) = TSImplicit(
    String(subtype), Float64(theta), _bdf_order(subtype, order), String[],
    _check_autodiff(autodiff),
)
TSImplicit(
    subtype::AbstractString, petsc_options::AbstractVector{<:AbstractString};
    order = nothing, autodiff = AutoForwardDiff(),
) = TSImplicit(
    String(subtype), nothing, _bdf_order(subtype, order),
    String[String(o) for o in petsc_options], _check_autodiff(autodiff),
)
TSImplicit(
    subtype::AbstractString, theta::Real,
    petsc_options::AbstractVector{<:AbstractString}; order = nothing,
    autodiff = AutoForwardDiff(),
) = TSImplicit(
    String(subtype), Float64(theta), _bdf_order(subtype, order),
    String[String(o) for o in petsc_options], _check_autodiff(autodiff),
)

"""
    TSIRK(nstages = 3, petsc_options = String[]; autodiff = AutoForwardDiff())

Gauss-Legendre implicit Runge-Kutta from PETSc's `TSIRK`, of order `2 *
nstages`: one stage is the implicit midpoint rule at order 2, two stages give
order 4 and three give order 6. Measured at each of those.

Fixed step: PETSc gives this family no embedded error estimate, so it steps at
the `dt` you give and warns if you pass a tolerance.

Needs a Jacobian, from the `ODEFunction`'s `jac` or from `autodiff`, and refuses
`AutoFiniteDiff()` rather than letting PETSc fail, since it solves all stages as
one coupled system whose matrix it cannot build from finite differences. That coupled matrix is a Kronecker product with the
Jacobian, which has no LU factorisation, so this algorithm defaults to
`-pc_type pbjacobi`; your own `petsc_options` are parsed afterwards and win.

A wrong Jacobian is not caught here. Where the other implicit families fail to
converge, this one reports success and returns a wrong answer, so check a
hand-written `jac` against a solve without one before trusting it.

A mass matrix is rejected. PETSc's coupled-stage matrix assumes `dF/du̇ = I`,
and with a non-identity mass matrix the answer drifts further from the true one
as `dt` shrinks instead of failing, which is worse than an error.
"""
struct TSIRK <: PETScTSAlgorithm
    nstages::Int
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
end

TSIRK(
    nstages::Integer = 3, petsc_options::AbstractVector{<:AbstractString} = String[];
    autodiff = AutoForwardDiff(),
) = TSIRK(Int(nstages), String[String(o) for o in petsc_options], _check_autodiff(autodiff))

"""
    TSDAE(subtype = "bdf", petsc_options = String[]; order = nothing, autodiff = AutoForwardDiff())

Fully implicit methods applied to a `DAEProblem`, whose residual `G(t, u, u') = 0`
is exactly the form PETSc's `IFunction` takes. `subtype` is `"beuler"`, `"cn"`,
`"theta"` or `"bdf"`, as for [`TSImplicit`](@ref).

A `DAEFunction`'s `jac(J, du, u, p, gamma, t)` is `gamma * dG/du' + dG/du`,
which is what PETSc's `IJacobian` wants whole, so it is passed straight through
and `gamma` is PETSc's shift. Without one the Jacobian comes from `autodiff`, as for
[`TSImplicit`](@ref).

`order` sets the BDF order, 1 through 6, and carries the same warning as
[`TSImplicit`](@ref): PETSc's own default is 2.

Only `"bdf"` adapts; the others step at the `dt` you give. `du0` is not used,
since PETSc derives the initial derivative itself.
"""
struct TSDAE <: PETScTSDAEAlgorithm
    subtype::String
    order::Union{Nothing, Int}
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
end

TSDAE(
    subtype::AbstractString = "bdf",
    petsc_options::AbstractVector{<:AbstractString} = String[];
    order = nothing, autodiff = AutoForwardDiff(),
) = TSDAE(
    String(subtype), _bdf_order(subtype, order), String[String(o) for o in petsc_options],
    _check_autodiff(autodiff),
)

"""
    TSARKIMEX(subtype = "3", petsc_options = String[]; autodiff = AutoForwardDiff())

Additive Runge-Kutta IMEX from PETSc's `TSARKIMEX`. `subtype` is a PETSc
`TSARKIMEXType` without its prefix, such as `"2e"`, `"3"`, `"4"` or `"5"`.

Takes a `SplitODEProblem` whose `f1` is integrated implicitly and whose `f2`
is integrated explicitly, and uses `f1`'s Jacobian when the problem carries
one. Without one, `f1`'s Jacobian comes from `autodiff`, as for
[`TSImplicit`](@ref). A plain `ODEProblem` is treated as fully implicit with the explicit part
left at zero, which is PETSc's own default. Adapts on its embedded error
estimate, except for `"prssp2"`, `"ars443"` and `"bpr3"`, which PETSc gives
none, so they step at the `dt` you give and warn if you pass a tolerance.

`"ars122"` needs a `SplitODEProblem`: it has an explicit first stage and is not stiffly
accurate, so PETSc cannot evaluate its first-stage slope when the whole problem is
implicit.

`"bpr3"` is refused on a `SplitODEProblem`, where it converges at first order. On a
plain `ODEProblem` PETSc does not use its explicit tableau, and it keeps order 3.
"""
struct TSARKIMEX <: PETScTSAlgorithm
    subtype::String
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
end

TSARKIMEX(
    subtype::AbstractString = "3",
    petsc_options::AbstractVector{<:AbstractString} = String[];
    autodiff = AutoForwardDiff(),
) = TSARKIMEX(
    String(subtype), String[String(o) for o in petsc_options], _check_autodiff(autodiff),
)

const _MPRK_TWO_WAY = ("2a22", "2a32", "p2", "p3")
const _MPRK_THREE_WAY = ("2a23", "2a33")

"""
    TSMPRK(slow, subtype = "p2", petsc_options = String[])
    TSMPRK(slow, medium, subtype = "2a23", petsc_options = String[])

PETSc's multirate partitioned Runge-Kutta. `slow` lists the indices of the state
that are integrated with the outer step; everything else is advanced on a smaller
one. Both parts come from the problem's own `f`, which is evaluated in full and
then read row by row, so nothing beyond the index list is asked of the caller.

Passing a `medium` set as well splits the state three ways, which is what PETSc's
`"2a23"` and `"2a33"` want. Whatever neither set names is the fast part.

Explicit, so a mass matrix and an analytic Jacobian are both refused. Without a
`medium` set `subtype` is one of `"2a22"`, `"2a32"`, `"p2"` or `"p3"`; with one it
is `"2a23"` or `"2a33"`.

The fast part is stepped on a shorter clock, so the step this tolerates is larger
than an explicit single-rate method's: on `u' = [-u1, -100u2]` with only the first
component slow, `"p3"` holds to `dt = 0.0425` against `TSRK("5dp")`'s `0.03`. The
ratio is fixed by the tableau rather than by the stiffness, so a separation much
wider than that is not something these methods can absorb.
"""
struct TSMPRK <: PETScTSAlgorithm
    slow::Vector{Int}
    medium::Vector{Int}
    subtype::String
    petsc_options::Vector{String}

    # The checks live here rather than in an outer constructor: the generated inner
    # one is the more specific method for the concrete types a default argument
    # expands to, so an outer one would be skipped exactly when it is not needed.
    function TSMPRK(
            slow::Vector{Int}, medium::Vector{Int}, subtype::String,
            petsc_options::Vector{String},
        )
        isempty(slow) && throw(ArgumentError("`slow` needs at least one index"))
        for (name, v) in (("slow", slow), ("medium", medium))
            all(i -> i >= 1, v) || throw(ArgumentError("`$name` indices start at 1"))
            length(unique(v)) == length(v) ||
                throw(ArgumentError("`$name` repeats an index"))
        end
        isempty(intersect(slow, medium)) ||
            throw(ArgumentError("`slow` and `medium` share an index"))
        wanted = isempty(medium) ? _MPRK_TWO_WAY : _MPRK_THREE_WAY
        subtype in wanted || throw(
            ArgumentError(
                isempty(medium) ?
                    "`$subtype` needs a `medium` split as well; without one use " *
                    join(map(t -> "\"$t\"", _MPRK_TWO_WAY), ", ") :
                    "`$subtype` takes only two splits, so leave `medium` out; with " *
                    "one use " * join(map(t -> "\"$t\"", _MPRK_THREE_WAY), " or "),
            ),
        )
        return new(sort(slow), sort(medium), subtype, petsc_options)
    end
end

TSMPRK(
    slow::AbstractVector{<:Integer},
    subtype::AbstractString = "p2",
    petsc_options::AbstractVector{<:AbstractString} = String[],
) = TSMPRK(
    Vector{Int}(slow), Int[], String(subtype),
    String[String(o) for o in petsc_options],
)

TSMPRK(
    slow::AbstractVector{<:Integer},
    medium::AbstractVector{<:Integer},
    subtype::AbstractString = "2a23",
    petsc_options::AbstractVector{<:AbstractString} = String[],
) = TSMPRK(
    Vector{Int}(slow), Vector{Int}(medium), String(subtype),
    String[String(o) for o in petsc_options],
)

"""
    TSGeneric(ts_type, petsc_options = String[]; explicit = false, autodiff = AutoForwardDiff())

Any other PETSc `TSType` by name. An implicit one such as `"alpha"` works with
the default; an explicit one such as `"euler"` or `"ssp"` needs
`explicit = true`, since PETSc then wants the right-hand side rather than the
implicit residual. The constructor refuses a type given the wrong `explicit`.
An explicit type also ignores a `jac` and rejects a mass matrix. An implicit one without
a `jac` gets its Jacobian from `autodiff`, as for [`TSImplicit`](@ref).

Whether the named type adapts is not known here, so no tolerance warning is
issued for it. Only `"euler"` and `"alpha"` have been run through this
package's own convergence tests.

`"alpha2"`, `"discgrad"`, `"eimex"`, `"mimex"` and `"mprk"` are refused: each is
driven through a PETSc setup call this package does not make, and without it they
crash or integrate to zero rather than saying anything.
"""
struct TSGeneric <: PETScTSAlgorithm
    ts_type::String
    explicit::Bool
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
end

# These PETSc types are driven through a setup call this package does not make, so
# PETSc reaches its own solve with half a problem: `alpha2`, `discgrad` and `mimex`
# take the process down with them, and `eimex` integrates to zero and reports success.
const _NEEDS_OTHER_SETUP = Dict(
    "alpha2" => "is for second-order systems and needs TSSetI2Function",
    "discgrad" => "needs TSDiscGradSetFormulation",
    "eimex" => "needs its own right-hand-side split, and integrates to zero without one",
    "mimex" => "needs TSRHSSplit to declare its slow and fast parts",
    "mprk" => "needs TSRHSSplit to declare its slow and fast parts",
    "pseudo" => "is pseudo-transient continuation toward a steady state and runs past the final time",
)

# Handed an implicit residual these integrate nothing. `euler`, `ssp` and `rk` say so
# through PETSc; `glee` returns the initial condition and reports success.
const _EXPLICIT_ONLY = ("euler", "glee", "rk", "ssp")

# These end on an explicit stage. Without a Jacobian PETSc stops and asks for one; with
# one it does not restore its Jacobian lag after that stage, so an adaptive solve fails
# within its first two steps and a fixed-step solve diverges.
const _ROSW_NO_STEP = ("lassp3p4s2c", "llssp3p4s2c", "ark3")

function TSGeneric(
        ts_type::AbstractString,
        petsc_options::AbstractVector{<:AbstractString} = String[];
        explicit::Bool = false, autodiff = AutoForwardDiff(),
    )
    t = String(ts_type)
    haskey(_NEEDS_OTHER_SETUP, t) && throw(
        ArgumentError("PETScDiffEq cannot drive `$t`, which $(_NEEDS_OTHER_SETUP[t])"),
    )
    !explicit && t in _EXPLICIT_ONLY && throw(
        ArgumentError("`$t` is an explicit PETSc type, so it needs `explicit = true`"),
    )
    return TSGeneric(
        t, explicit, String[String(o) for o in petsc_options], _check_autodiff(autodiff),
    )
end

_uses_ifunction(::TSRK) = false
_uses_ifunction(::TSRosW) = true
_uses_ifunction(::TSImplicit) = true
_uses_ifunction(::TSIRK) = true
_uses_ifunction(::TSDAE) = true
_uses_ifunction(::TSARKIMEX) = true
_uses_ifunction(::TSMPRK) = false
_uses_ifunction(alg::TSGeneric) = !alg.explicit

# PETSc registers these without embedded weights and gives them the `none` adaptor.
const _RK_NO_ESTIMATE = ("1fe", "2b", "3", "4")
const _ROSW_NO_ESTIMATE = ("theta1", "theta2")
const _ARKIMEX_NO_ESTIMATE = ("prssp2", "ars443", "bpr3")

# Only these PETSc TS families carry an embedded error estimate. The rest step
# at the requested dt and ignore any tolerance. `nothing` means the answer is
# not known, which is the case for an arbitrary TSGeneric type.
_adapts(alg::TSRK) = !(alg.subtype in _RK_NO_ESTIMATE)
_adapts(alg::TSRosW) = !(alg.subtype in _ROSW_NO_ESTIMATE)
_adapts(::TSIRK) = false
_adapts(alg::TSDAE) = alg.subtype == "bdf"
_adapts(alg::TSARKIMEX) = !(alg.subtype in _ARKIMEX_NO_ESTIMATE)
_adapts(alg::TSImplicit) = alg.subtype == "bdf"
_adapts(::TSMPRK) = false
_adapts(::TSGeneric) = nothing

# PETSc's interpolant is at least cubic for these. For the rest, where the problem gives a
# derivative, the state between step ends comes from a cubic Hermite interpolant.
const _RK_CUBIC_INTERP = ("5dp",)
const _ROSW_CUBIC_INTERP = ("ra34pw2", "lassp3p4s2c", "llssp3p4s2c", "ark3")
const _ARKIMEX_CUBIC_INTERP = ("4", "5")
# PETSc registers these with no interpolant at all.
const _ROSW_NO_INTERP = (
    "r34prw", "r3prl2", "rodas3", "rodaspr", "rodaspr2", "grk4t", "shamp4", "veldd4", "4l",
)
const _ARKIMEX_NO_INTERP = ("prssp2", "ars443", "bpr3")

# Whether the state between step ends comes from PETSc. BDF's interpolant is the
# polynomial through the history its steps are built from, so it keeps the method's order.
_petsc_interpolant(alg::TSRK) = alg.subtype in _RK_CUBIC_INTERP
_petsc_interpolant(alg::TSRosW) = alg.subtype in _ROSW_CUBIC_INTERP
_petsc_interpolant(alg::TSARKIMEX) = alg.subtype in _ARKIMEX_CUBIC_INTERP
_petsc_interpolant(alg::Union{TSImplicit, TSDAE}) = alg.subtype == "bdf"
_petsc_interpolant(::Union{TSIRK, TSMPRK, TSGeneric}) = false

# Whether PETSc can interpolate at all, which is all a mass matrix or a DAEProblem has.
# TSIRK hands back its output vector untouched rather than failing. `nothing` means the
# answer is not known yet.
_interpolates(::TSRK) = true
_interpolates(alg::TSRosW) = !(alg.subtype in _ROSW_NO_INTERP)
_interpolates(alg::TSARKIMEX) = !(alg.subtype in _ARKIMEX_NO_INTERP)
_interpolates(::Union{TSImplicit, TSDAE}) = true
_interpolates(::Union{TSIRK, TSMPRK}) = false
_interpolates(::TSGeneric) = nothing

# The orders PETSc registers each tableau with; GaussAdjoint sizes its quadrature from them.
const _RK_ORDER = Dict(
    "1fe" => 1, "2a" => 2, "2b" => 2, "3" => 3, "3bs" => 3, "4" => 4,
    "5f" => 5, "5dp" => 5, "5bs" => 5, "6vr" => 6, "7vr" => 7, "8vr" => 8,
)
const _ROSW_ORDER = Dict(
    "theta1" => 1, "theta2" => 2, "2p" => 2, "2m" => 2, "ra3pw" => 3, "ra34pw2" => 3,
    "r34prw" => 3, "r3prl2" => 3, "rodas3" => 3, "rodaspr" => 4, "rodaspr2" => 4,
    "sandu3" => 3, "assp3p3s1c" => 3, "lassp3p4s2c" => 3, "llssp3p4s2c" => 3, "ark3" => 3,
    "grk4t" => 4, "shamp4" => 4, "veldd4" => 4, "4l" => 4,
)
# PETSc registers `1bee` at 2, but it is backward Euler and converges at first order.
const _ARKIMEX_ORDER = Dict(
    "1bee" => 1, "ars122" => 2, "a2" => 2, "l2" => 2, "2c" => 2, "2d" => 2, "2e" => 2,
    "prssp2" => 2, "3" => 3, "ars443" => 3, "bpr3" => 3, "4" => 4, "5" => 5,
)
const _MPRK_ORDER = Dict(
    "2a22" => 2, "2a23" => 2, "2a32" => 2, "2a33" => 2, "p2" => 2, "p3" => 3,
)

function _order(table, family, subtype)
    haskey(table, subtype) ||
        throw(ArgumentError("no order is known for $family subtype \"$subtype\""))
    return table[subtype]
end

SciMLBase.alg_order(alg::TSRK) = _order(_RK_ORDER, "TSRK", alg.subtype)
SciMLBase.alg_order(alg::TSRosW) = _order(_ROSW_ORDER, "TSRosW", alg.subtype)
SciMLBase.alg_order(alg::TSARKIMEX) = _order(_ARKIMEX_ORDER, "TSARKIMEX", alg.subtype)
SciMLBase.alg_order(alg::TSMPRK) = _order(_MPRK_ORDER, "TSMPRK", alg.subtype)
SciMLBase.alg_order(alg::TSIRK) = 2 * alg.nstages
SciMLBase.alg_order(alg::TSImplicit) = _implicit_order(alg.subtype, alg.theta, alg.order)
SciMLBase.alg_order(alg::TSDAE) = _implicit_order(alg.subtype, nothing, alg.order)

function _implicit_order(subtype, theta, order)
    subtype == "beuler" && return 1
    subtype == "cn" && return 2
    subtype == "theta" && return theta === nothing || theta == 0.5 ? 2 : 1
    subtype == "bdf" && return something(order, 2)
    throw(ArgumentError("no order is known for implicit subtype \"$subtype\""))
end

mutable struct TSContext{F, F2, JAC, JBUF, P, T, V}
    petsclib::T
    f!::F
    f2!::F2
    jac!::JAC
    p::P
    du::Vector{Float64}
    u::Vector{Float64}
    mudot::Vector{Float64}
    resid::Vector{Float64}
    M::Union{Nothing, Matrix{Float64}}
    dae::Bool
    missing_diag::Vector{Int}
    W::Matrix{Float64}
    idx0::Vector{LibPETSc.PetscInt}
    row_cols0::Vector{Vector{LibPETSc.PetscInt}}
    row_src::Vector{Vector{Int}}
    row_buf::Vector{Vector{Float64}}
    J::JBUF
    ts::Vector{Float64}
    us::Vector{Vector{Float64}}
    dus::Vector{Vector{Float64}}
    saveat::Vector{Float64}
    saveat_idx::Int
    save_everystep::Bool
    save_start::Bool
    dense::Bool
    save_idxs::Union{Nothing, Vector{Int}}
    work::V
    hermite::Bool
    interpolates::Union{Nothing, Bool}
    alg_name::String
    # The start of the step a TSSolve monitor call ends.
    step_t::Float64
    step_u::Vector{Float64}
    # The end of the integrator's step as the step reached it, before any callback there.
    end_s::Float64
    end_u::Vector{Float64}
    # The derivatives at the two ends of the step being interpolated, each `nothing`
    # until something needs it.
    fstart::Union{Nothing, Vector{Float64}}
    fend::Union{Nothing, Vector{Float64}}
    # Whether a parameter may have changed since the end derivative was taken. Only the
    # next step's start derivative has to account for it.
    pdirty::Bool
    slow_idxs::Vector{Int}
    medium_idxs::Vector{Int}
    fast_idxs::Vector{Int}
    part_t::Float64
    part_u::Vector{Float64}
    part_valid::Bool
    # The smallest step the caller allows, 0.0 when none, and whether one was asked for.
    dtmin::Float64
    dt_too_small::Bool
    # The caller's own check on the state at each step's end, and whether it has fired.
    unstable::Any
    unstable_hit::Bool
    # The caller's time direction, which the checks above are given their times in.
    tdir::Float64
    # The caller's `isoutofdomain(u, p, t)`, or nothing.
    domain::Any
    nf::Int
    nf2::Int
    njacs::Int
    err::Union{Nothing, Any}
end

function _record!(ctx, t, x, du = nothing)
    full = Vector{Float64}(x)
    idxs = ctx.save_idxs
    push!(ctx.ts, Float64(t))
    # The derivative comes from the whole state even when only part is kept.
    if ctx.dense
        du === nothing && (du = _derivative(ctx, Float64(t), full))
        push!(ctx.dus, _select(du, idxs))
    end
    push!(ctx.us, _select(full, idxs))
    return du
end

# A point on the end of the step being interpolated, where dense output and the
# interpolant need the same derivative.
function _record_end!(ctx, t, x)
    du = _record!(ctx, t, x, ctx.fend)
    ctx.hermite && (ctx.fend = du)
    return nothing
end

# After each step TSSolve takes, the checks that end a solve early, as OrdinaryDiffEq makes
# them before its next step: the caller's floor, kept here because PETSc's own `dt_min`
# clamps the step and takes it whatever its error, and the caller's
# `unstable_check(dt, u, p, t)`, given the step about to be taken in the caller's time.
# Neither applies after a step that failed or after the last one, and a step PETSc has
# shortened to land on the final time does not count against the floor. TSSolve calls this
# before it decides whether to go on, so a reason set here ends the solve on this step.
function _post_step!(ts_ptr::LibPETSc.CTS)::LibPETSc.PetscErrorCode
    ctxptr = Ref{Ptr{Cvoid}}(C_NULL)
    get_ctx, get_solution = POST_STEP_FNS[]
    ccall(get_ctx, LibPETSc.PetscErrorCode, (LibPETSc.CTS, Ptr{Ptr{Cvoid}}), ts_ptr, ctxptr)
    ctx = unsafe_pointer_to_objref(ctxptr[])::TSContext
    ctx.err === nothing || return LibPETSc.PetscErrorCode(0)
    try
        pl = ctx.petsclib
        ts = LibPETSc.TS(ts_ptr, pl)
        Int(LibPETSc.TSGetConvergedReason(pl, ts)) < 0 && return LibPETSc.PetscErrorCode(0)
        s = Float64(LibPETSc.TSGetTime(pl, ts))
        smax = Float64(LibPETSc.TSGetMaxTime(pl, ts))
        s >= smax - _near(smax) && return LibPETSc.PetscErrorCode(0)
        hnext = Float64(LibPETSc.TSGetTimeStep(pl, ts))
        stop = false
        if ctx.dtmin > 0 && hnext < ctx.dtmin && s + hnext < smax - _near(smax)
            ctx.dt_too_small = stop = true
        end
        if !stop && ctx.unstable !== nothing
            x = Ref{LibPETSc.CVec}(C_NULL)
            ccall(
                get_solution, LibPETSc.PetscErrorCode, (LibPETSc.CTS, Ptr{LibPETSc.CVec}),
                ts_ptr, x,
            )
            u = _readvec!(ctx.u, pl, PETSc.VecPtr(pl, x[], false))
            ctx.unstable(ctx.tdir * hnext, u, ctx.p, _user_t(ctx.tdir, s)) &&
                (ctx.unstable_hit = stop = true)
        end
        stop && LibPETSc.TSSetConvergedReason(pl, ts, LibPETSc.TS_CONVERGED_USER)
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const POST_STEP_PTR = Ref{Ptr{Cvoid}}(C_NULL)

# The post-step callback carries no context of its own, so the context rides on the TS.
function _set_post_step!(pl, ts, ctxptr)
    lib = Libdl.dlopen(pl.petsc_library)
    POST_STEP_FNS[][1] == C_NULL && (
        POST_STEP_FNS[] = (
            Libdl.dlsym(lib, :TSGetApplicationContext), Libdl.dlsym(lib, :TSGetSolution),
        )
    )
    ccall(
        Libdl.dlsym(lib, :TSSetApplicationContext), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Ptr{Cvoid}), ts, ctxptr,
    )
    ccall(
        Libdl.dlsym(lib, :TSSetPostStep), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Ptr{Cvoid}), ts, POST_STEP_PTR[],
    )
    return nothing
end
const POST_STEP_FNS = Ref((C_NULL, C_NULL))

# Exactly the same time: accepted steps near a singularity can be far closer than any
# tolerance and are still points of the solution.
# The caller's floor on the step, which OrdinaryDiffEq applies only to an adaptive solve and
# which `force_dtmin` hands to PETSc instead.
_floor(dtmin, force_dtmin, adaptive) =
    force_dtmin || !adaptive || dtmin === nothing ? 0.0 : abs(Float64(dtmin))

# A rounding error at `t` itself. A stop just after the start of a long span is still ahead,
# where a tolerance scaled to the final time would count it as passed.
_near(t) = 100 * eps(max(one(Float64), abs(t)))

_last_recorded(ctx, t) = !isempty(ctx.ts) && ctx.ts[end] == t

_select(u, ::Nothing) = u
_select(u, idxs::Vector{Int}) = u[idxs]

function _derivative(ctx, t, u)
    du = similar(u)
    ctx.f!(du, u, ctx.p, t)
    if ctx.f2! !== nothing
        ctx.f2!(ctx.du, u, ctx.p, t)
        du .+= ctx.du
    end
    ctx.nf += 1
    ctx.f2! === nothing || (ctx.nf2 += 1)
    return du
end

_saved(ctx, u) = ctx.save_idxs === nothing ? Vector{Float64}(u) :
    Float64[u[i] for i in ctx.save_idxs]

_interp(ctx, ts, dus) = ctx.dense ? SciMLBase.HermiteInterpolation(ts, ctx.us, dus) :
    SciMLBase.LinearInterpolation(ts, ctx.us)

# The cubic Hermite interpolant at s on the step from (s0, u0) to (s1, u1), as SciMLBase
# writes the one dense output uses. Each end's derivative is kept until that end moves.
function _hermite!(out, ctx, s, s0, u0, s1, u1)
    ctx.fstart === nothing && (ctx.fstart = _derivative(ctx, s0, u0))
    ctx.fend === nothing && (ctx.fend = _derivative(ctx, s1, u1))
    f0, f1 = ctx.fstart, ctx.fend
    dt = s1 - s0
    Θ = (s - s0) / dt
    @. out = (1 - Θ) * u0 + Θ * u1 +
        Θ * (Θ - 1) * ((1 - 2Θ) * (u1 - u0) + (Θ - 1) * dt * f0 + Θ * dt * f1)
    return out
end

_no_interpolant(ctx) = ArgumentError(
    "`$(ctx.alg_name)` has no interpolant in PETSc, and a mass matrix or a DAEProblem " *
        "gives no derivative to build one from, so it has no state between step ends for " *
        "saveat, a ContinuousCallback or integrator(t); use a type that interpolates, such " *
        "as BDF, or keep saveat times on step ends and pass `rootfind = NoRootFind`",
)

# PETSc.jl does not carry PETSc's error codes; these are PETSC_ERR_SUP and
# PETSC_ERR_MAT_LU_ZRPVT.
const PETSC_ERR_SUP = 56
const PETSC_ERR_MAT_LU_ZRPVT = 71
const PETSC_ERR_FP = 72
# What a callback returns to PETSc when the user's code threw; the exception itself is what
# reaches the caller.
const CALLBACK_THREW = 1

# A dense LU factorization raises on a zero pivot whatever PETSc was told about failed
# steps, so a singular Newton matrix ends the solve with this error instead of a failed
# step. Where the options ask the linear solve to raise, it is left to raise.
_failed_step(e, h) =
    e isa LibPETSc.PetscError && !h.pivot_raises &&
    (e.code == PETSC_ERR_MAT_LU_ZRPVT || e.code == PETSC_ERR_FP)

# A step that stopped part way through adds nothing to PETSc's rejection counters, so the
# warning is the only account of why the solve stopped where it did.
function _warn_failed_step(alg, code)
    why = code == PETSC_ERR_FP ?
        "PETSc hit a floating point exception, an overflow or a NaN in the step or in " *
        "its error estimate" :
        "the LU factorization of its Newton matrix hit a zero pivot"
    @warn "`$(_warn_name(alg))` ends here because $why"
    return nothing
end

# Whether the options leave the linear or the nonlinear solve raising on a failure.
function _pivot_raises(pl, ts)
    lib = Libdl.dlopen(pl.petsc_library)
    snes, ksp = Ref{LibPETSc.CSNES}(C_NULL), Ref{LibPETSc.CKSP}(C_NULL)
    raises = Ref{LibPETSc.PetscBool}(LibPETSc.PETSC_FALSE)
    snes_raises = Ref{LibPETSc.PetscBool}(LibPETSc.PETSC_FALSE)
    ccall(
        Libdl.dlsym(lib, :TSGetSNES), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Ptr{LibPETSc.CSNES}), ts, snes,
    )
    ccall(
        Libdl.dlsym(lib, :SNESGetKSP), LibPETSc.PetscErrorCode,
        (LibPETSc.CSNES, Ptr{LibPETSc.CKSP}), snes[], ksp,
    )
    ccall(
        Libdl.dlsym(lib, :KSPGetErrorIfNotConverged), LibPETSc.PetscErrorCode,
        (LibPETSc.CKSP, Ptr{LibPETSc.PetscBool}), ksp[], raises,
    )
    ccall(
        Libdl.dlsym(lib, :SNESGetErrorIfNotConverged), LibPETSc.PetscErrorCode,
        (LibPETSc.CSNES, Ptr{LibPETSc.PetscBool}), snes[], snes_raises,
    )
    return raises[] == LibPETSc.PETSC_TRUE || snes_raises[] == LibPETSc.PETSC_TRUE
end

# With no `jac`, PETSc differences the step's equations to get their Jacobian. Handed a
# matrix with the pattern, it perturbs every column of one colour together, rather than
# each column in turn as it does on the dense matrix it makes for itself.
function _colour_jacobian!(pl, ts, mat)
    lib = Libdl.dlopen(pl.petsc_library)
    snes = Ref{LibPETSc.CSNES}(C_NULL)
    ccall(
        Libdl.dlsym(lib, :TSGetSNES), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Ptr{LibPETSc.CSNES}), ts, snes,
    )
    code = ccall(
        Libdl.dlsym(lib, :SNESSetJacobian), LibPETSc.PetscErrorCode,
        (LibPETSc.CSNES, LibPETSc.CMat, LibPETSc.CMat, Ptr{Cvoid}, Ptr{Cvoid}),
        snes[], mat, mat, Libdl.dlsym(lib, :SNESComputeJacobianDefaultColor), C_NULL,
    )
    code == 0 || throw(LibPETSc.PetscError(code))
    return nothing
end

_fd_pattern(jac_prototype, M, n) =
    _jacobian_pattern(SparseMatrixCSC{Float64, Int}(jac_prototype), n, M)

# PETSc applies the last setting of an option, matches its name without regard to case, and
# reads a bare flag as true. PETSc 3.22 has no getter for this one, so it is read here.
function _option_flag(opts, name)
    value = false
    for (i, opt) in enumerate(opts)
        _names_option(opt, name) || continue
        setting = occursin("=", opt) ? last(split(opt, "="; limit = 2)) :
            i < length(opts) && !startswith(opts[i + 1], "-") ? opts[i + 1] : "true"
        value = lowercase(setting) in ("1", "true", "yes", "on")
    end
    return value
end

# Whether a zero pivot or an overflow is being turned into a retcode, set on each push.
# PETSc runs one task at a time, so one flag serves.
const QUIET_FAILED_STEPS = Ref(true)

# An exception the user's own code threw is not printed, since the exception reaches the
# caller, and a zero pivot or overflow that becomes a retcode is not printed either. Nothing
# printed also keeps PETSc from opening the next error's traceback as one that followed it.
# Every other code goes on to `traceback`.
function _zero_pivot_handler(
        comm::MPI.API.MPI_Comm, line::Cint, fun::Ptr{Cchar}, file::Ptr{Cchar},
        n::LibPETSc.PetscErrorCode, p::Cint, mess::Ptr{Cchar}, traceback::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    n == CALLBACK_THREW && return n
    QUIET_FAILED_STEPS[] && (n == PETSC_ERR_MAT_LU_ZRPVT || n == PETSC_ERR_FP) && return n
    return ccall(
        traceback, LibPETSc.PetscErrorCode,
        (
            MPI.API.MPI_Comm, Cint, Ptr{Cchar}, Ptr{Cchar}, LibPETSc.PetscErrorCode, Cint,
            Ptr{Cchar}, Ptr{Cvoid},
        ),
        comm, line, fun, file, n, p, mess, C_NULL,
    )
end

const ZERO_PIVOT_HANDLER_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const ERROR_HANDLER_FNS = Ref((C_NULL, C_NULL, C_NULL))

# Runs `f` with the handler above in front of PETSc's. This wraps every step, so the
# symbols are looked up once.
function _quiet_errors(f, h)
    QUIET_FAILED_STEPS[] = !h.pivot_raises
    if ERROR_HANDLER_FNS[][1] == C_NULL
        lib = Libdl.dlopen(h.petsclib.petsc_library)
        ERROR_HANDLER_FNS[] = (
            Libdl.dlsym(lib, :PetscPushErrorHandler), Libdl.dlsym(lib, :PetscPopErrorHandler),
            Libdl.dlsym(lib, :PetscTraceBackErrorHandler),
        )
    end
    push, pop, traceback = ERROR_HANDLER_FNS[]
    ccall(
        push, LibPETSc.PetscErrorCode, (Ptr{Cvoid}, Ptr{Cvoid}),
        ZERO_PIVOT_HANDLER_PTR[], traceback,
    )
    try
        return f()
    finally
        ccall(pop, LibPETSc.PetscErrorCode, ())
    end
end

# PETSc's interpolant at s, left in `ctx.work`, or `nothing` where PETSc has none.
function _petsc_interpolate!(ctx, ts, s)
    pl = ctx.petsclib
    ctx.interpolates === false && return nothing
    ctx.interpolates === true && (LibPETSc.TSInterpolate(pl, ts, s, ctx.work); return ctx.work)
    # Whether this type interpolates is only known by asking, and PETSc prints a traceback
    # before refusing, so it is asked with printing switched off. A type that registers an
    # interpolant which writes nothing answers without refusing, so the vector is filled
    # first with a value PETSc has to overwrite for the answer to be its own.
    PETSc.withlocalarray!(w -> fill!(w, NaN), ctx.work; read = false, write = true)
    lib = Libdl.dlopen(pl.petsc_library)
    ccall(
        Libdl.dlsym(lib, :PetscPushErrorHandler), LibPETSc.PetscErrorCode,
        (Ptr{Cvoid}, Ptr{Cvoid}), Libdl.dlsym(lib, :PetscReturnErrorHandler), C_NULL,
    )
    try
        LibPETSc.TSInterpolate(pl, ts, s, ctx.work)
    catch e
        e isa LibPETSc.PetscError && e.code == PETSC_ERR_SUP || rethrow()
        ctx.interpolates = false
        return nothing
    finally
        ccall(Libdl.dlsym(lib, :PetscPopErrorHandler), LibPETSc.PetscErrorCode, ())
    end
    written = PETSc.withlocalarray!(
        w -> any(!isnan, w), ctx.work; read = true, write = false,
    )
    ctx.interpolates = written
    return written ? ctx.work : nothing
end

_mass(ctx, i, j) = ctx.M === nothing ? (i == j ? 1.0 : 0.0) : ctx.M[i, j]

# A DAE Jacobian is `shift * dG/du_dot + dG/du`, which is what PETSc wants
# whole, so it is filled in place of the `shift * M - J` an ODE builds.
function _call_jac!(ctx, xdot_ptr, shift, t)
    if ctx.dae
        _readvec!(ctx.mudot, ctx.petsclib, PETSc.VecPtr(ctx.petsclib, xdot_ptr, false))
        ctx.jac!(ctx.J, ctx.mudot, ctx.u, ctx.p, Float64(shift), Float64(t))
    else
        ctx.jac!(ctx.J, ctx.u, ctx.p, t)
    end
    return nothing
end

function _fill_rows!(ctx, shift, n)
    @inbounds for i in 1:n
        cols = ctx.row_cols0[i]
        src = ctx.row_src[i]
        buf = ctx.row_buf[i]
        for k in eachindex(cols)
            j = Int(cols[k]) + 1
            jv = src[k] == 0 ? 0.0 : ctx.J.nzval[src[k]]
            buf[k] = ctx.dae ? jv : shift * _mass(ctx, i, j) - jv
        end
    end
    return nothing
end

# PETSc stores a SeqAIJ row by ascending column, which is the order
# `_row_structure` builds, so the per-row buffers concatenate straight into the
# matrix's own value array and no per-row call is needed.
function _setrows!(ctx, A, n)
    vals = LibPETSc.MatSeqAIJGetArray(ctx.petsclib, A)
    try
        k = 1
        @inbounds for i in 1:n
            buf = ctx.row_buf[i]
            for v in buf
                vals[k] = v
                k += 1
            end
        end
    finally
        LibPETSc.MatSeqAIJRestoreArray(ctx.petsclib, A, vals)
    end
    return nothing
end

# The shift lands on the diagonal and on every entry of the mass matrix, so those get a
# slot even where the prototype has none, with no Jacobian entry behind it.
function _row_structure(J::SparseMatrixCSC, n, M = nothing)
    cols = [Int[] for _ in 1:n]
    src = [Int[] for _ in 1:n]
    for j in 1:n, k in J.colptr[j]:(J.colptr[j + 1] - 1)
        i = J.rowval[k]
        push!(cols[i], j)
        push!(src[i], k)
    end
    shifted = [CartesianIndex(i, i) for i in 1:n]
    M === nothing || append!(shifted, findall(!iszero, M))
    for ij in shifted
        i, j = ij[1], ij[2]
        j in cols[i] && continue
        push!(cols[i], j)
        push!(src[i], 0)
    end
    for i in 1:n
        perm = sortperm(cols[i])
        cols[i] = cols[i][perm]
        src[i] = src[i][perm]
    end
    cols0 = Vector{LibPETSc.PetscInt}[LibPETSc.PetscInt[c - 1 for c in cols[i]] for i in 1:n]
    buf = [zeros(length(cols[i])) for i in 1:n]
    return cols0, src, buf
end

function _setblock!(ctx, A, n)
    return LibPETSc.MatSetValues(
        ctx.petsclib, A, LibPETSc.PetscInt(n), ctx.idx0,
        LibPETSc.PetscInt(n), ctx.idx0, vec(ctx.W), LibPETSc.INSERT_VALUES,
    )
end

_stored(A::SparseMatrixCSC, i, j) = any(==(i), @view A.rowval[A.colptr[j]:(A.colptr[j + 1] - 1)])

_as_inplace(f, iip::Bool) = iip ? f : (du, u, p, t) -> (du .= f(u, p, t); nothing)
_as_inplace_jac(j, iip::Bool) = iip ? j :
    (J, u, p, t) -> (_copy_jac!(J, j(u, p, t)); nothing)

# PETSc only steps forward in time, so a reversed span is stepped in s = -t, where
# v(s) = u(-s) has dv/ds = -f(v, p, -s) and a residual G(t, u, u') becomes
# G(-s, v, -dv/ds). The derivative buffer is negated in place and restored rather
# than copied, since these run on every function and Jacobian evaluation.
_reverse_rhs(f) = (du, u, p, s) -> (f(du, u, p, _user_t(-1.0, s)); du .*= -1; nothing)
_reverse_jac(j) =
    (J, u, p, s) -> (j(J, u, p, _user_t(-1.0, s)); LinearAlgebra.rmul!(J, -1); nothing)
_reverse_residual(g) =
    (r, dv, u, p, s) -> (dv .*= -1; g(r, dv, u, p, _user_t(-1.0, s)); dv .*= -1; nothing)
_reverse_dae_jac(j) =
    (J, dv, u, p, gamma, s) -> (dv .*= -1; j(J, dv, u, p, -gamma, _user_t(-1.0, s)); dv .*= -1; nothing)

# The user's time for PETSc's s. Negating s = 0 gives -0.0, and `isless(-0.0, 0.0)`
# holds, so a stop or a saved time at zero would be missed; adding zero gives +0.0.
_user_t(tdir, s) = tdir * s + 0.0

_copy_jac!(J::AbstractMatrix, A) = (copyto!(J, A); nothing)

# The sparse buffer's nonzero positions were captured at setup, so writing an
# entry the prototype never declared would silently misplace every later one.
function _setstored!(J::SparseMatrixCSC, i, j, v)
    r = J.colptr[j]:(J.colptr[j + 1] - 1)
    k = findfirst(==(i), @view J.rowval[r])
    k === nothing && throw(
        ArgumentError(
            "the Jacobian has an entry at ($i, $j) that the jac_prototype does not " *
                "declare; add it to the prototype",
        ),
    )
    J.nzval[first(r) + k - 1] = v
    return nothing
end

function _copy_jac!(J::SparseMatrixCSC, A::SparseMatrixCSC)
    fill!(J.nzval, 0.0)
    rows, vals = rowvals(A), nonzeros(A)
    @inbounds for j in axes(A, 2), k in nzrange(A, j)
        _setstored!(J, rows[k], j, vals[k])
    end
    return nothing
end

function _copy_jac!(J::SparseMatrixCSC, A::AbstractMatrix)
    fill!(J.nzval, 0.0)
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        iszero(A[i, j]) || _setstored!(J, i, j, A[i, j])
    end
    return nothing
end

# PETSc's own array accessor picks a device type and allocates on every call,
# which on a small right-hand side costs more than the derivative it is fetching.
function _readvec!(dest::Vector{Float64}, pl, v)
    a = LibPETSc.VecGetArrayRead(pl, v)
    try
        copyto!(dest, a)
    finally
        LibPETSc.VecRestoreArrayRead(pl, v, a)
    end
    return dest
end

function _writevec!(pl, v, src::Vector{Float64})
    a = LibPETSc.VecGetArrayWrite(pl, v)
    try
        copyto!(a, src)
    finally
        LibPETSc.VecRestoreArrayWrite(pl, v, a)
    end
    return nothing
end

# The context reaches these callbacks as an untyped pointer, so each one hands
# straight off to a body that compiles for the concrete context type.
function _rhs!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _rhs_body!(ctx, t, x_ptr, f_ptr)
end

function _rhs_body!(ctx, t, x_ptr, f_ptr)
    pl = ctx.petsclib
    try
        _readvec!(ctx.u, pl, PETSc.VecPtr(pl, x_ptr, false))
        ctx.f!(ctx.du, ctx.u, ctx.p, t)
        _writevec!(pl, PETSc.VecPtr(pl, f_ptr, false), ctx.du)
        ctx.nf += 1
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const RHS_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _split_rhs!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _split_rhs_body!(ctx, t, x_ptr, f_ptr)
end

function _split_rhs_body!(ctx, t, x_ptr, f_ptr)
    pl = ctx.petsclib
    try
        _readvec!(ctx.u, pl, PETSc.VecPtr(pl, x_ptr, false))
        ctx.f2!(ctx.du, ctx.u, ctx.p, t)
        _writevec!(pl, PETSc.VecPtr(pl, f_ptr, false), ctx.du)
        ctx.nf2 += 1
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const SPLIT_RHS_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _ifunction!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        xdot_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _ifunction_body!(ctx, t, x_ptr, xdot_ptr, f_ptr)
end

function _ifunction_body!(ctx, t, x_ptr, xdot_ptr, f_ptr)
    pl = ctx.petsclib
    try
        _readvec!(ctx.u, pl, PETSc.VecPtr(pl, x_ptr, false))
        udot = _readvec!(ctx.mudot, pl, PETSc.VecPtr(pl, xdot_ptr, false))
        if ctx.dae
            ctx.f!(ctx.resid, udot, ctx.u, ctx.p, t)
        else
            ctx.f!(ctx.du, ctx.u, ctx.p, t)
            if ctx.M === nothing
                @. ctx.resid = udot - ctx.du
            else
                mul!(ctx.resid, ctx.M, udot)
                @. ctx.resid = ctx.resid - ctx.du
            end
        end
        _writevec!(pl, PETSc.VecPtr(pl, f_ptr, false), ctx.resid)
        ctx.nf += 1
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

# MPRK asks for each part of the split separately. PETSc hands over the whole state
# and wants back only that part's entries, so the user's `f` is evaluated in full and
# the requested rows are copied out.
function _mprk_part!(ctx, t, x_ptr, f_ptr, idxs)
    pl = ctx.petsclib
    try
        _readvec!(ctx.u, pl, PETSc.VecPtr(pl, x_ptr, false))
        # PETSc asks for each part of the same stage in turn, so the whole
        # right-hand side only has to be evaluated for the first of them.
        if !(ctx.part_valid && ctx.part_t == t && ctx.part_u == ctx.u)
            ctx.f!(ctx.du, ctx.u, ctx.p, t)
            ctx.nf += 1
            ctx.part_t = t
            copyto!(ctx.part_u, ctx.u)
            ctx.part_valid = true
        end
        sub = PETSc.VecPtr(pl, f_ptr, false)
        vals = LibPETSc.VecGetArrayWrite(pl, sub)
        try
            @inbounds for (k, i) in enumerate(idxs)
                vals[k] = ctx.du[i]
            end
        finally
            LibPETSc.VecRestoreArrayWrite(pl, sub, vals)
        end
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

function _mprk_slow!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _mprk_part!(ctx, t, x_ptr, f_ptr, ctx.slow_idxs)
end

function _mprk_medium!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _mprk_part!(ctx, t, x_ptr, f_ptr, ctx.medium_idxs)
end

function _mprk_fast!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _mprk_part!(ctx, t, x_ptr, f_ptr, ctx.fast_idxs)
end

const MPRK_SLOW_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const MPRK_MEDIUM_PTR = Ref{Ptr{Cvoid}}(C_NULL)
const MPRK_FAST_PTR = Ref{Ptr{Cvoid}}(C_NULL)
# PETSc.jl wraps `TSRHSSplitSetRHSFunction` with an opaque function type and no room
# for a context, so the symbol is called directly the way PETSc.jl itself does for
# `TSSetRHSFunction`.
const TSRHSSPLIT_SET_RHS = Ref{Ptr{Cvoid}}(C_NULL)

const IFUNCTION_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _ijacobian!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        xdot_ptr::LibPETSc.CVec,
        shift::LibPETSc.PetscReal,
        A_ptr::LibPETSc.CMat,
        B_ptr::LibPETSc.CMat,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _ijacobian_body!(ctx, t, x_ptr, xdot_ptr, shift, A_ptr, B_ptr)
end

function _ijacobian_body!(ctx, t, x_ptr, xdot_ptr, shift, A_ptr, B_ptr)
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    A = LibPETSc.PetscMat(A_ptr, ctx.petsclib)
    B = LibPETSc.PetscMat(B_ptr, ctx.petsclib)
    try
        _readvec!(ctx.u, ctx.petsclib, x)
        _call_jac!(ctx, xdot_ptr, shift, t)
        ctx.njacs += 1
        n = length(ctx.u)
        # One batched MatSetValues beats n^2 single-entry ccalls by orders of
        # magnitude. PETSc reads the block row-major, so entry (i,j) is stored
        # at W[j,i] and `vec` then yields the order PETSc wants.
        @inbounds for j in 1:n, i in 1:n
            ctx.W[j, i] = ctx.dae ? ctx.J[i, j] : shift * _mass(ctx, i, j) - ctx.J[i, j]
        end
        _setblock!(ctx, B, n)
        PETSc.assemble!(B)
        # Under `-snes_mf_operator` the operator is PETSc's matrix-free one, which takes no
        # values and is assembled only to pick up the new state.
        B.ptr == A.ptr || PETSc.assemble!(A)
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const IJACOBIAN_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _sparse_ijacobian!(
        ::LibPETSc.CTS,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        xdot_ptr::LibPETSc.CVec,
        shift::LibPETSc.PetscReal,
        A_ptr::LibPETSc.CMat,
        B_ptr::LibPETSc.CMat,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _sparse_ijacobian_body!(ctx, t, x_ptr, xdot_ptr, shift, A_ptr, B_ptr)
end

function _sparse_ijacobian_body!(ctx, t, x_ptr, xdot_ptr, shift, A_ptr, B_ptr)
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    A = LibPETSc.PetscMat(A_ptr, ctx.petsclib)
    B = LibPETSc.PetscMat(B_ptr, ctx.petsclib)
    try
        _readvec!(ctx.u, ctx.petsclib, x)
        _call_jac!(ctx, xdot_ptr, shift, t)
        ctx.njacs += 1
        n = length(ctx.u)
        # One MatSetValues per row rather than one per stored entry. The row
        # structure is the prototype's pattern unioned with the diagonal, which
        # is what was preallocated, and it never changes.
        _fill_rows!(ctx, shift, n)
        _setrows!(ctx, B, n)
        PETSc.assemble!(B)
        B.ptr == A.ptr || PETSc.assemble!(A)
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const SPARSE_IJACOBIAN_PTR = Ref{Ptr{Cvoid}}(C_NULL)

function _monitor!(
        ts_ptr::LibPETSc.CTS,
        step::LibPETSc.PetscInt,
        t::LibPETSc.PetscReal,
        x_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _monitor_body!(ctx, ts_ptr, step, t, x_ptr)
end

function _monitor_body!(ctx, ts_ptr, step, t, x_ptr)
    # The time that failed lies before the step PETSc can interpolate in.
    ctx.err === nothing || return LibPETSc.PetscErrorCode(0)
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    try
        ts = LibPETSc.TS(ts_ptr, ctx.petsclib)
        tol = 100 * eps(max(one(Float64), abs(Float64(t))))
        # Whether a saveat point sits on this step's end, which then needs no second point.
        landed = false
        while ctx.saveat_idx <= length(ctx.saveat) &&
                ctx.saveat[ctx.saveat_idx] <= Float64(t) + tol
            want = ctx.saveat[ctx.saveat_idx]
            # Before any step has been taken there is nothing to interpolate
            # from, and the incoming vector is already the initial state.
            if step == 0 || abs(want - Float64(t)) <= tol
                _record_end!(ctx, want, _readvec!(ctx.u, ctx.petsclib, x))
                landed = true
            elseif ctx.hermite
                # `-ts_exact_final_time interpolate` steps past tf, then reports its own
                # state at tf, which is the one the solve ends on, in a call of its own.
                tmax = Float64(LibPETSc.TSGetMaxTime(ctx.petsclib, ts))
                want >= tmax - tol && Float64(t) > tmax + tol && break
                u1 = _readvec!(ctx.u, ctx.petsclib, x)
                _record!(
                    ctx, want,
                    _hermite!(similar(u1), ctx, want, ctx.step_t, ctx.step_u, Float64(t), u1),
                )
            elseif _petsc_interpolate!(ctx, ts, want) === nothing
                ctx.err = _no_interpolant(ctx)
                # A monitor that fails makes PETSc print a traceback, so the solve
                # ends after its next step instead.
                LibPETSc.TSSetMaxSteps(ctx.petsclib, ts, step + 1)
                return LibPETSc.PetscErrorCode(0)
            else
                _record!(ctx, want, _readvec!(ctx.u, ctx.petsclib, ctx.work))
            end
            ctx.saveat_idx += 1
        end
        # The start, and every step's end when every step is saved. A step PETSc cannot
        # take brings the monitor back to the time it last reported, and one time is worth
        # one point.
        if (step == 0 ? ctx.save_start : ctx.save_everystep) && !landed &&
                !_last_recorded(ctx, Float64(t))
            _record!(ctx, t, _readvec!(ctx.u, ctx.petsclib, x))
        end
        # The step that ends here is where the next one starts.
        if ctx.hermite && ctx.saveat_idx <= length(ctx.saveat)
            ctx.step_t = Float64(t)
            _readvec!(ctx.step_u, ctx.petsclib, x)
            ctx.fstart = ctx.pdirty ? nothing : ctx.fend
            ctx.fend = nothing
            ctx.pdirty = false
        end

    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
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
    SPLIT_RHS_PTR[] = @cfunction(
        _split_rhs!,
        LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec, Ptr{Cvoid})
    )
    POST_STEP_PTR[] = @cfunction(_post_step!, LibPETSc.PetscErrorCode, (LibPETSc.CTS,))
    MONITOR_PTR[] = @cfunction(
        _monitor!,
        LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, LibPETSc.PetscInt, LibPETSc.PetscReal, LibPETSc.CVec, Ptr{Cvoid})
    )
    IFUNCTION_PTR[] = @cfunction(
        _ifunction!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec,
            LibPETSc.CVec, Ptr{Cvoid},
        )
    )
    IJACOBIAN_PTR[] = @cfunction(
        _ijacobian!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec,
            LibPETSc.PetscReal, LibPETSc.CMat, LibPETSc.CMat, Ptr{Cvoid},
        )
    )
    SPARSE_IJACOBIAN_PTR[] = @cfunction(
        _sparse_ijacobian!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec,
            LibPETSc.PetscReal, LibPETSc.CMat, LibPETSc.CMat, Ptr{Cvoid},
        )
    )
    MPRK_SLOW_PTR[] = @cfunction(
        _mprk_slow!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec,
            Ptr{Cvoid},
        )
    )
    MPRK_MEDIUM_PTR[] = @cfunction(
        _mprk_medium!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec,
            Ptr{Cvoid},
        )
    )
    MPRK_FAST_PTR[] = @cfunction(
        _mprk_fast!,
        LibPETSc.PetscErrorCode,
        (
            LibPETSc.CTS, LibPETSc.PetscReal, LibPETSc.CVec, LibPETSc.CVec,
            Ptr{Cvoid},
        )
    )
    ZERO_PIVOT_HANDLER_PTR[] = @cfunction(
        _zero_pivot_handler,
        LibPETSc.PetscErrorCode,
        (
            MPI.API.MPI_Comm, Cint, Ptr{Cchar}, Ptr{Cchar}, LibPETSc.PetscErrorCode, Cint,
            Ptr{Cchar}, Ptr{Cvoid},
        )
    )
    _init_adjoint_pointers!()
    return nothing
end

# `LibPETSc.PetscInt` is a fixed `Int64` in PETSc.jl rather than a property of the
# library that got loaded, so on a platform offering only 32-bit-index builds the
# index vectors handed to PETSc would be the wrong width and nothing would say so.
function _split_rhs_symbol(petsclib)
    TSRHSSPLIT_SET_RHS[] == C_NULL && (
        TSRHSSPLIT_SET_RHS[] = Libdl.dlsym(
            Libdl.dlopen(petsclib.petsc_library), :TSRHSSplitSetRHSFunction,
        )
    )
    return TSRHSSPLIT_SET_RHS[]
end

# One part of the split: the rows `idxs` own, filled by `fptr`.
function _set_split!(petsclib, ts, name, idxs, fptr, ctxptr)
    n = LibPETSc.PetscInt(length(idxs))
    is = LibPETSc.ISCreateGeneral(
        petsclib, MPI.COMM_SELF, n,
        LibPETSc.PetscInt[i - 1 for i in idxs], LibPETSc.PETSC_COPY_VALUES,
    )
    LibPETSc.TSRHSSplitSetIS(petsclib, ts, name, is)
    code = ccall(
        _split_rhs_symbol(petsclib), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Cstring, LibPETSc.CVec, Ptr{Cvoid}, Ptr{Cvoid}),
        ts, name, C_NULL, fptr, ctxptr,
    )
    iszero(code) ||
        throw(ErrorException("TSRHSSplitSetRHSFunction(\"$name\") failed with $code"))
    return nothing
end

function _check_inttype(petsclib)
    PETSc.inttype(petsclib) === LibPETSc.PetscInt || error(
        "PETScDiffEq needs a PETSc built with $(LibPETSc.PetscInt) indices, but the " *
            "library it loaded uses $(PETSc.inttype(petsclib))",
    )
    return nothing
end

# `maxiters = typemax(Int)` is how SciML spells "no limit", which does not fit a
# PETSc built with 32-bit indices.
_maxsteps(maxiters) = LibPETSc.PetscInt(min(maxiters, typemax(LibPETSc.PetscInt)))

function _jacobian_pattern(jac_prototype::SparseMatrixCSC, n::Integer, M = nothing)
    rows, cols, _ = findnz(jac_prototype)
    all_rows = vcat(rows, 1:n)
    all_cols = vcat(cols, 1:n)
    if M !== nothing
        for ij in findall(!iszero, M)
            push!(all_rows, ij[1])
            push!(all_cols, ij[2])
        end
    end
    return sparse(all_rows, all_cols, ones(length(all_rows)), n, n)
end

function _cstr(f::F, s::AbstractString) where {F}
    str = String(s)
    return GC.@preserve str f(Base.unsafe_convert(Ptr{Cchar}, str))
end

_ts_type(::TSRK) = "rk"
_ts_type(::TSRosW) = "rosw"
_ts_type(alg::TSImplicit) = alg.subtype
_ts_type(::TSIRK) = "irk"
_ts_type(alg::TSDAE) = alg.subtype
_ts_type(::TSARKIMEX) = "arkimex"
_ts_type(::TSMPRK) = "mprk"
_ts_type(alg::TSGeneric) = alg.ts_type

_warn_name(alg::Union{TSRK, TSRosW, TSARKIMEX}) = "$(_ts_type(alg)) $(alg.subtype)"
_warn_name(alg) = _ts_type(alg)

# What PETSc runs, named as `_warn_name` names an algorithm.
function _running_name(petsclib, ts)
    type = LibPETSc.TSGetType(petsclib, ts)
    type == "rk" && return "rk $(LibPETSc.TSRKGetType(petsclib, ts))"
    type == "rosw" && return "rosw $(LibPETSc.TSRosWGetType(petsclib, ts))"
    type == "arkimex" && return "arkimex $(LibPETSc.TSARKIMEXGetType(petsclib, ts))"
    return type
end

_set_subtype!(petsclib, ts, alg::TSRK) =
    _cstr(p -> LibPETSc.TSRKSetType(petsclib, ts, p), alg.subtype)
_set_subtype!(petsclib, ts, alg::TSRosW) =
    _cstr(p -> LibPETSc.TSRosWSetType(petsclib, ts, p), alg.subtype)
function _set_subtype!(petsclib, ts, alg::TSImplicit)
    if alg.subtype == "theta" && alg.theta !== nothing
        LibPETSc.TSThetaSetTheta(petsclib, ts, alg.theta)
    end
    if alg.order !== nothing
        LibPETSc.TSBDFSetOrder(petsclib, ts, LibPETSc.PetscInt(alg.order))
    end
    return nothing
end
function _set_subtype!(petsclib, ts, alg::TSIRK)
    LibPETSc.TSIRKSetNumStages(petsclib, ts, LibPETSc.PetscInt(alg.nstages))
    # The tableau is built from the stage count, so it needs rebuilding
    # whenever that count changes.
    _cstr(p -> LibPETSc.TSIRKSetType(petsclib, ts, p), "gauss")
    return nothing
end
_set_subtype!(petsclib, ts, alg::TSARKIMEX) =
    _cstr(p -> LibPETSc.TSARKIMEXSetType(petsclib, ts, p), alg.subtype)
function _set_subtype!(petsclib, ts, alg::TSDAE)
    if alg.order !== nothing
        LibPETSc.TSBDFSetOrder(petsclib, ts, LibPETSc.PetscInt(alg.order))
    end
    return nothing
end
_set_subtype!(petsclib, ts, ::TSMPRK) = nothing
_set_subtype!(petsclib, ts, ::TSGeneric) = nothing

_default_options(::AnyPETScTS) = String[]
# A Kronecker-product coupled-stage matrix has no LU factorisation, so PETSc's
# default preconditioner cannot be set up for it.
_default_options(::TSIRK) = ["-pc_type", "pbjacobi"]
# Without `-ts_use_splitrhsfunction` PETSc takes its other path, evaluating the whole
# right-hand side and slicing it, and never calls the per-part functions below.
_default_options(alg::TSMPRK) =
    ["-ts_mprk_type", alg.subtype, "-ts_use_splitrhsfunction", "true"]

const UNSUPPORTED_KWARGS = (
    :internalnorm, :calck, :alias_u0, :sensealg,
    :controller, :qmax, :qmin, :gamma, :beta1, :beta2,
)

mutable struct TSHandles{CTX, T}
    ctx::CTX
    petsclib::T
    ts::Any
    u::Any
    jac_mat::Any
    fd_mat::Any
    # The evaluations of `f` an automatic-differentiation Jacobian makes, or `nothing`.
    ad_calls::Union{Nothing, Base.RefValue{Int}}
    opts::Any
    t0::Float64
    tf::Float64
    tdir::Float64
    u0::Vector{Float64}
    maxiters::Int
    save_start::Bool
    save_end::Bool
    pivot_raises::Bool
    # The PETSc error code a step stopped on, or 0 where none did.
    stopped::Int
    tolvecs::Vector{Any}
    tolbufs::Vector{Vector{Float64}}
    destroyed::Bool
end

# Julia runs finalizers after `atexit` hooks, by which point MPI has shut down and
# freeing a PETSc object aborts the process. Handles are tracked weakly so that an
# integrator dropped part-way is still torn down while PETSc is alive.
const LIVE_HANDLES = WeakKeyDict{Any, Nothing}()
const EXIT_CLEANUP_ARMED = Ref(false)

# PETSc.jl registers its own teardown when it initializes, which is later than this
# module's `__init__`, and exit hooks run newest first. Arming here rather than in
# `__init__` is what puts this one ahead of PETSc's.
function _arm_exit_cleanup!()
    EXIT_CLEANUP_ARMED[] && return nothing
    EXIT_CLEANUP_ARMED[] = true
    atexit(_destroy_live_handles!)
    return nothing
end

function _destroy_live_handles!()
    for h in collect(keys(LIVE_HANDLES))
        _destroy!(h)
    end
    return nothing
end

# PETSc and MPI, which runs at THREAD_SERIALIZED, are shared by the whole process, and PETSc's
# options stack is global, so every entry point that reaches them runs one task at a time.
const PETSC_LOCK = ReentrantLock()
_locked(f) = lock(f, PETSC_LOCK)

# A finalizer must not wait on a lock, so one that finds it taken tries again later.
function _finalize!(h::TSHandles)
    if islocked(PETSC_LOCK) || !trylock(PETSC_LOCK)
        finalizer(_finalize!, h)
        return nothing
    end
    try
        _destroy!(h)
    finally
        unlock(PETSC_LOCK)
    end
    return nothing
end

function _destroy!(h::TSHandles)
    h.destroyed && return nothing
    h.destroyed = true
    # Once PETSc has finalized (at process exit) its objects are already gone
    # and calling into it reaches MPI after MPI has shut down, so an integrator
    # collected late must not try to free anything.
    (PETSc.finalized(h.petsclib) || MPI.Finalized()) && return nothing
    h.opts === nothing || PETSc.destroy(h.opts)
    h.jac_mat === nothing || PETSc.destroy(h.jac_mat)
    h.fd_mat === nothing || PETSc.destroy(h.fd_mat)
    for v in h.tolvecs
        v.ptr == C_NULL || PETSc.destroy(v)
    end
    h.ctx.work.ptr == C_NULL || PETSc.destroy(h.ctx.work)
    h.u === nothing || PETSc.destroy(h.u)
    h.ts === nothing || LibPETSc.TSDestroy(h.petsclib, h.ts)
    return nothing
end

_tolscalar(tol, default) = tol === nothing || tol isa AbstractVector ? default : Float64(tol)

function _check_tol(tol, n, name)
    tol isa AbstractVector || return nothing
    length(tol) == n ||
        throw(ArgumentError("`$name` has length $(length(tol)), but the state has $n"))
    all(t -> t >= 0, tol) || throw(ArgumentError("`$name` has a negative entry"))
    return nothing
end

# PETSc prefers the per-component vector when one is attached, so the scalar beside it is
# only the fallback.
function _set_tolerances!(h::TSHandles, abstol, reltol)
    pl, n = h.petsclib, length(h.u0)
    novec = LibPETSc.PetscVec{typeof(pl)}()
    avec = _tolvec(h, pl, abstol, n, "abstol")
    rvec = _tolvec(h, pl, reltol, n, "reltol")
    LibPETSc.TSSetTolerances(
        pl, h.ts, _tolscalar(abstol, 1.0e-6), avec === nothing ? novec : avec,
        _tolscalar(reltol, 1.0e-3), rvec === nothing ? novec : rvec,
    )
    return nothing
end

function _tolvec(h::TSHandles, petsclib, tol, n, name)
    tol isa AbstractVector || return nothing
    # PETSc borrows this array rather than copying it, and reads it on every
    # adaptive step, so it has to outlive the handle.
    buf = Vector{Float64}(collect(tol))
    push!(h.tolbufs, buf)
    v = PETSc.VecSeq(petsclib, buf)
    push!(h.tolvecs, v)
    return v
end

# Hairer and Wanner's starting step as OrdinaryDiffEq takes it for an in-place f, in the
# user's time before the right-hand side is wrapped for PETSc. Where OrdinaryDiffEq falls
# back to its floor nextfloat(max(dtmin, eps(t0))) or ends up with NaN, this returns the
# small default instead.
function _initial_dt(f1, f2, u0, p, t0, tdir, order, abstol, reltol, dtmin, dtmax)
    dtmin_floor = nextfloat(max(dtmin, eps(t0)))
    smalldt = max(dtmin_floor, 1.0e-6)
    isempty(u0) && return smalldt
    norm = DiffEqBase.ODE_DEFAULT_NORM
    function rhs(u, t)
        du = similar(u)
        f1(du, u, p, t)
        if f2 !== nothing
            extra = similar(u)
            f2(extra, u, p, t)
            du .+= extra
        end
        return du
    end
    sk = abstol .+ abs.(u0) .* reltol
    f0 = rhs(u0, t0)
    all(isfinite, f0) || return smalldt
    d0 = norm(u0 ./ sk, t0)
    d1 = norm(f0 ./ sk, t0)
    isnan(d1) && return smalldt
    dt0 = min(d0 < 1.0e-5 || d1 < 1.0e-5 ? smalldt : (d0 / d1) / 100, dtmax)
    dt0 < 10 * eps(Float64) && return smalldt
    f_next = rhs(u0 .+ (tdir * dt0) .* f0, t0 + tdir * dt0)
    f0 == f_next && return min(max(dtmin_floor, 100 * dt0), dtmax)
    d2 = norm((f_next .- f0) ./ sk, t0) / dt0
    m = max(d1, d2)
    dt1 = m <= 1.0e-15 ? max(1.0e-6, dt0 * 1.0e-3) : 10.0^(-(2 + log10(m)) / order)
    dt = max(dtmin_floor, min(100 * dt0, dt1, dtmax))
    return isfinite(dt) && dt > 0 ? dt : smalldt
end

const SupportedProblem = Union{SciMLBase.AbstractODEProblem, SciMLBase.AbstractDAEProblem}

function _setup(
        prob::SupportedProblem,
        alg::AnyPETScTS;
        dt = nothing,
        reltol = nothing,
        abstol = nothing,
        maxiters = 1000000,
        adaptive = true,
        dtmin = nothing,
        dtmax = nothing,
        force_dtmin = false,
        unstable_check = nothing,
        isoutofdomain = nothing,
        saveat = Float64[],
        save_everystep = nothing,
        save_start = nothing,
        save_end = nothing,
        save_on = true,
        dense = nothing,
        save_idxs = nothing,
        tstops = (),
        extra_options = String[],
        jac_advice = nothing,
        kwargs...,
    )
    for key in UNSUPPORTED_KWARGS
        if haskey(kwargs, key)
            @warn "PETScDiffEq does not support `$key` and is ignoring it"
        end
    end
    prob.u0 isa AbstractVector{<:Real} ||
        throw(ArgumentError("PETScDiffEq requires a real AbstractVector u0"))
    dt_given = dt !== nothing
    if !dt_given && !(adaptive && _adapts(alg) === true)
        throw(
            ArgumentError(
                "PETScDiffEq needs `dt` unless the solve adapts on an error estimate it " *
                    "knows about, which `$(_ts_type(alg))` with `adaptive = $adaptive` does not",
            ),
        )
    end
    is_split = prob.f isa SciMLBase.SplitFunction
    is_split && !(alg isa TSARKIMEX) &&
        throw(ArgumentError("PETScDiffEq only supports SplitODEProblem with TSARKIMEX"))
    is_dae = prob isa SciMLBase.AbstractDAEProblem
    mass_matrix = is_dae ? nothing : prob.f.mass_matrix
    has_mass = !(mass_matrix === nothing || mass_matrix == LinearAlgebra.I)
    if has_mass && !_uses_ifunction(alg)
        throw(
            ArgumentError(
                "PETScDiffEq cannot apply a mass matrix with an explicit algorithm; " *
                    "use an implicit one such as TSImplicit or TSRosW",
            ),
        )
    end
    if has_mass && alg isa TSIRK
        throw(
            ArgumentError(
                "PETScDiffEq does not support a mass matrix with TSIRK; PETSc's " *
                    "coupled-stage matrix assumes dF/du_dot = I, and the answer drifts " *
                    "further from the true one as dt shrinks rather than failing",
            ),
        )
    end
    if has_mass && is_split
        throw(ArgumentError("PETScDiffEq does not support a mass matrix on a SplitODEProblem"))
    end

    t0, tf = Float64(prob.tspan[1]), Float64(prob.tspan[2])
    t0 == tf && throw(ArgumentError("PETScDiffEq requires tspan[1] != tspan[2]"))
    # Everything handed to PETSc from here on is in its forward-running time s = tdir * t.
    tdir = t0 < tf ? 1.0 : -1.0
    t0, tf = tdir * t0, tdir * tf

    u0 = Vector{Float64}(vec(prob.u0))
    n = length(u0)

    slow_idxs, medium_idxs, fast_idxs = if alg isa TSMPRK
        named = vcat(alg.slow, alg.medium)
        maximum(named) <= n || throw(
            ArgumentError("`slow` or `medium` names index $(maximum(named)), but the state has $n"),
        )
        rest = setdiff(1:n, named)
        isempty(rest) && throw(
            ArgumentError("`slow` and `medium` cover the whole state, leaving nothing fast"),
        )
        alg.slow, alg.medium, rest
    else
        (Int[], Int[], Int[])
    end

    petsclib = PETSc.getlib(PetscScalar = Float64)
    _check_inttype(petsclib)
    PETSc.initialized(petsclib) || PETSc.initialize(petsclib)
    _arm_exit_cleanup!()

    iip = SciMLBase.isinplace(prob)
    # SciMLBase wraps the functions for the problem's own eltype, and PETSc's double build
    # calls them with Float64 arrays, so a problem in another eltype gets them unwrapped
    # and is solved in Float64, as a whole-number state is.
    unwrap = eltype(prob.u0) === Float64 ? identity : SciMLBase.unwrapped_f
    f1 = unwrap(is_split ? prob.f.f1.f : prob.f.f)
    is_dae && !iip &&
        throw(ArgumentError("PETScDiffEq requires an in-place DAEProblem residual"))
    f2 = is_split ? unwrap(prob.f.f2.f) : nothing
    for g in (f1, f2)
        # An AbstractSciMLOperator ignores the f(du,u,p,t) call this package
        # makes, leaving the derivative buffer untouched rather than erroring.
        g isa SciMLOperators.AbstractSciMLOperator && throw(
            ArgumentError(
                "PETScDiffEq does not support an operator-valued right-hand side; " *
                    "supply a function f!(du, u, p, t)",
            ),
        )
    end
    f1 = is_dae ? f1 : _as_inplace(f1, iip)
    f2 = f2 === nothing ? nothing : _as_inplace(f2, iip)
    # Without a `jac` one is built with `autodiff`, unless that asks PETSc to difference.
    builds_jac = _uses_ifunction(alg) && prob.f.jac === nothing && !_petsc_differences(alg)
    has_jac = _uses_ifunction(alg) && (prob.f.jac !== nothing || builds_jac)
    if alg isa TSIRK && !has_jac
        throw(
            ArgumentError(
                "TSIRK needs a Jacobian; give the ODEFunction a `jac` or leave `autodiff` " *
                    "at a backend other than `AutoFiniteDiff()`, since PETSc builds its " *
                    "coupled-stage matrix from one and has no finite-difference fallback for it",
            ),
        )
    end
    if alg isa TSRosW && alg.subtype in _ROSW_NO_STEP
        throw(
            ArgumentError(
                "TSRosW(\"$(alg.subtype)\") cannot be used: without a `jac` PETSc stops " *
                    "and asks for one, and with one it does not restore its Jacobian lag " *
                    "after the explicit last stage, so an adaptive solve fails within its " *
                    "first two steps and a fixed-step solve diverges; use another TSRosW type",
            ),
        )
    end
    if alg isa TSRosW && alg.subtype == "assp3p3s1c" && has_mass
        throw(
            ArgumentError(
                "TSRosW(\"assp3p3s1c\") cannot take a mass matrix; PETSc leaves the mass " *
                    "matrix out of its explicit first stage, so the solve reports success " *
                    "with an error that does not shrink with dt",
            ),
        )
    end
    if alg isa TSRosW && alg.subtype == "assp3p3s1c" && !has_jac
        throw(
            ArgumentError(
                "TSRosW(\"assp3p3s1c\") needs a Jacobian; give the ODEFunction a `jac` " *
                    "or leave `autodiff` at a backend other than `AutoFiniteDiff()`, since " *
                    "PETSc asks for one at the start of every step and has no " *
                    "finite-difference fallback there",
            ),
        )
    end
    if alg isa TSARKIMEX && alg.subtype == "ars122" && !is_split
        throw(
            ArgumentError(
                "TSARKIMEX(\"ars122\") needs a SplitODEProblem; it has an explicit first " *
                    "stage and is not stiffly accurate, so PETSc cannot evaluate its " *
                    "first-stage slope when the whole problem is implicit",
            ),
        )
    end
    if alg isa TSARKIMEX && alg.subtype == "bpr3" && is_split
        throw(
            ArgumentError(
                "TSARKIMEX(\"bpr3\") converges at first order on a SplitODEProblem; solve " *
                    "a plain ODEProblem with it, or use another TSARKIMEX type",
            ),
        )
    end
    ad_calls = builds_jac ? Ref(0) : nothing
    jac_fn = if !has_jac
        nothing
    elseif builds_jac
        # Dual numbers need the function itself, not the wrapper SciMLBase made for Float64.
        f_ad = SciMLBase.unwrapped_f(is_split ? prob.f.f1.f : prob.f.f)
        user_t0 = Float64(prob.tspan[1])
        advice = something(jac_advice, is_dae ? _DAE_ADVICE : _ODE_ADVICE)
        is_dae ?
            _ad_dae_jacobian(
                _autodiff(alg), f_ad, prob.f.jac_prototype, u0, prob.p, user_t0, ad_calls,
                advice,
            ) :
            _ad_jacobian(
                _autodiff(alg), _as_inplace(f_ad, iip), prob.f.jac_prototype, u0, prob.p,
                user_t0, ad_calls, advice,
            )
    else
        is_dae ? unwrap(prob.f.jac) : _as_inplace_jac(unwrap(prob.f.jac), iip)
    end
    _check_tol(abstol, n, "abstol")
    _check_tol(reltol, n, "reltol")
    if !dt_given
        user_t0 = Float64(prob.tspan[1])
        est_dtmin = dtmin === nothing ? 0.0 : abs(Float64(dtmin))
        # With no explicit derivative to estimate from, start small as OrdinaryDiffEq does.
        dt = if is_dae
            1.0e-6 * abs(tf - t0)
        elseif has_mass
            max(nextfloat(max(est_dtmin, eps(user_t0))), 1.0e-6)
        else
            # The abstol/reltol keywords as this function hands them to PETSc below, which
            # keeps its own 1e-4 for both when neither is given. Tolerances set through
            # petsc_options are not seen here.
            est_abstol = something(abstol, reltol === nothing ? 1.0e-4 : 1.0e-6)
            est_reltol = something(reltol, abstol === nothing ? 1.0e-4 : 1.0e-3)
            user_dtmax = dtmax === nothing || isinf(dtmax) ? Inf : abs(Float64(dtmax))
            first_stop = minimum(
                (abs(Float64(s) - user_t0) for s in tstops if tdir * (Float64(s) - user_t0) > 0);
                init = Inf,
            )
            _initial_dt(
                f1, f2, u0, prob.p, user_t0, tdir, SciMLBase.alg_order(alg),
                est_abstol, est_reltol, est_dtmin, min(user_dtmax, first_stop, abs(tf - t0)),
            )
        end
    end
    if tdir < 0
        f1 = is_dae ? _reverse_residual(f1) : _reverse_rhs(f1)
        f2 = f2 === nothing ? nothing : _reverse_rhs(f2)
        jac_fn = jac_fn === nothing ? nothing :
            (is_dae ? _reverse_dae_jac(jac_fn) : _reverse_jac(jac_fn))
    end
    jac_prototype = has_jac ? prob.f.jac_prototype : nothing
    uses_sparse_jac = jac_prototype isa SparseMatrixCSC
    J0 = if !has_jac
        zeros(0, 0)
    elseif uses_sparse_jac
        SparseMatrixCSC{Float64, Int}(jac_prototype)
    else
        zeros(n, n)
    end
    saveat_times = saveat isa Number ?
        collect(Float64, t0:abs(Float64(saveat)):tf) :
        sort!(tdir .* Vector{Float64}(collect(saveat)))
    filter!(t -> t0 - eps(tf) <= t <= tf + eps(tf), saveat_times)
    # OrdinaryDiffEq's defaults: a saveat keeps only its own points, and an end is kept
    # when saveat names it. A save flag given explicitly wins over saveat.
    endtol = 100 * eps(max(one(Float64), abs(tf)))
    no_saveat = !(saveat isa Number) && isempty(saveat)
    save_everystep = save_on && something(save_everystep, no_saveat)
    save_start = something(
        save_start, save_everystep || no_saveat || saveat isa Number ||
            any(t -> abs(t - t0) <= endtol, saveat_times),
    )
    save_end = something(
        save_end, save_everystep || no_saveat || saveat isa Number ||
            any(t -> abs(t - tf) <= endtol, saveat_times),
    )
    save_on || empty!(saveat_times)
    save_start || filter!(t -> abs(t - t0) > endtol, saveat_times)
    save_end || filter!(t -> abs(t - tf) > endtol, saveat_times)
    M = has_mass ? Matrix{Float64}(mass_matrix) : nothing
    missing_diag = uses_sparse_jac ?
        [i for i in 1:n if !_stored(J0, i, i)] : Int[]
    W0 = has_jac && !uses_sparse_jac ? zeros(n, n) : zeros(0, 0)
    idx0 = has_jac && !uses_sparse_jac ?
        LibPETSc.PetscInt[i - 1 for i in 1:n] : LibPETSc.PetscInt[]
    row_cols0, row_src, row_buf = uses_sparse_jac ? _row_structure(J0, n, M) :
        (Vector{LibPETSc.PetscInt}[], Vector{Int}[], Vector{Float64}[])
    kept = if save_idxs === nothing
        nothing
    else
        v = save_idxs isa Integer ? [Int(save_idxs)] : Vector{Int}(collect(save_idxs))
        isempty(v) && throw(ArgumentError("`save_idxs` must name at least one component"))
        all(i -> 1 <= i <= n, v) || throw(
            ArgumentError("`save_idxs` has an index outside 1:$n"),
        )
        v
    end
    dense_out = dense === nothing ?
        (save_everystep && no_saveat && !has_mass && !is_dae) : Bool(dense)
    if dense_out && has_mass
        throw(
            ArgumentError(
                "PETScDiffEq cannot produce dense output with a mass matrix; " *
                    "the saved derivative would have to be M \\ f(u)",
            ),
        )
    end
    if dense_out && is_dae
        throw(
            ArgumentError(
                "PETScDiffEq cannot produce dense output for a DAEProblem; a residual " *
                    "gives no derivative to build a Hermite interpolant from",
            ),
        )
    end
    ctx = TSContext(
        petsclib, f1, f2, jac_fn, prob.p,
        similar(u0), similar(u0), similar(u0), similar(u0), M, is_dae, missing_diag, W0,
        idx0,
        row_cols0, row_src, row_buf, J0,
        Float64[], Vector{Float64}[], Vector{Float64}[],
        saveat_times, 1, save_everystep, save_start, dense_out, kept,
        PETSc.VecSeq(petsclib, n),
        !has_mass && !is_dae && !_petsc_interpolant(alg), _interpolates(alg), _warn_name(alg),
        NaN, similar(u0), t0, copy(u0), nothing, nothing, false,
        slow_idxs, medium_idxs, fast_idxs,
        NaN, similar(u0), false,
        _floor(dtmin, force_dtmin, adaptive && _adapts(alg) !== false), false,
        unstable_check, false, tdir, isoutofdomain,
        0, 0, 0, nothing,
    )
    h = TSHandles(
        ctx, petsclib, nothing, nothing, nothing, nothing, ad_calls, nothing,
        t0, tf, tdir, u0, Int(maxiters), save_start, save_end, false, 0,
        Any[], Vector{Float64}[], false,
    )
    finalizer(_finalize!, h)
    LIVE_HANDLES[h] = nothing

    try
        h.ts = LibPETSc.TSCreate(petsclib, MPI.COMM_SELF)
        ts = h.ts
        LibPETSc.TSSetProblemType(petsclib, ts, LibPETSc.TS_NONLINEAR)
        LibPETSc.TSSetType(petsclib, ts, _ts_type(alg))
        _set_subtype!(petsclib, ts, alg)

        h.u = PETSc.VecSeq(petsclib, n)
        u = h.u
        PETSc.withlocalarray!(u; read = false, write = true) do ua
            copyto!(ua, u0)
        end
        LibPETSc.TSSetSolution(petsclib, ts, u)

        ctxptr = pointer_from_objref(ctx)
        GC.@preserve ctx begin
            if _uses_ifunction(alg)
                LibPETSc.TSSetIFunction(petsclib, ts, nothing, IFUNCTION_PTR[], ctxptr)
            else
                LibPETSc.TSSetRHSFunction(petsclib, ts, nothing, RHS_PTR[], ctxptr)
            end
            # MPRK steps the whole system as well as each part, so it needs the
            # plain right-hand side above in addition to these.
            if alg isa TSMPRK
                _set_split!(petsclib, ts, "slow", slow_idxs, MPRK_SLOW_PTR[], ctxptr)
                isempty(medium_idxs) || _set_split!(
                    petsclib, ts, "medium", medium_idxs, MPRK_MEDIUM_PTR[], ctxptr,
                )
                _set_split!(petsclib, ts, "fast", fast_idxs, MPRK_FAST_PTR[], ctxptr)
            end
            if is_split
                LibPETSc.TSSetRHSFunction(petsclib, ts, nothing, SPLIT_RHS_PTR[], ctxptr)
            end
            if has_jac && uses_sparse_jac
                pattern = _jacobian_pattern(J0, n, M)
                h.jac_mat = PETSc.MatSeqAIJWithArrays(petsclib, MPI.COMM_SELF, pattern)
                LibPETSc.TSSetIJacobian(
                    petsclib, ts, h.jac_mat, h.jac_mat, SPARSE_IJACOBIAN_PTR[], ctxptr,
                )
            elseif has_jac
                # A dense matrix gets LAPACK's pivoting LU, as PETSc's own does.
                h.jac_mat = PETSc.MatSeqDense(petsclib, zeros(n, n))
                LibPETSc.TSSetIJacobian(
                    petsclib, ts, h.jac_mat, h.jac_mat, IJACOBIAN_PTR[], ctxptr,
                )
            elseif _uses_ifunction(alg) && prob.f.jac_prototype isa SparseArrays.AbstractSparseMatrix
                h.fd_mat = PETSc.MatSeqAIJWithArrays(
                    petsclib, MPI.COMM_SELF, _fd_pattern(prob.f.jac_prototype, M, n),
                )
                _colour_jacobian!(petsclib, ts, h.fd_mat)
            end
            LibPETSc.TSMonitorSet(petsclib, ts, MONITOR_PTR[], ctxptr)
            if ctx.dtmin > 0 || ctx.unstable !== nothing
                _set_post_step!(petsclib, ts, ctxptr)
            end
            LibPETSc.TSSetTime(petsclib, ts, t0)
            # PETSc's floor clamps only the steps its adaptor chooses, not the one given.
            LibPETSc.TSSetTimeStep(
                petsclib, ts,
                force_dtmin && dtmin !== nothing ? max(abs(Float64(dt)), abs(Float64(dtmin))) :
                    abs(Float64(dt)),
            )
            LibPETSc.TSSetMaxTime(petsclib, ts, tf)
            LibPETSc.TSSetMaxSteps(petsclib, ts, _maxsteps(maxiters))
            LibPETSc.TSSetExactFinalTime(
                petsclib, ts, LibPETSc.TS_EXACTFINALTIME_MATCHSTEP,
            )
            if (reltol !== nothing || abstol !== nothing) && _adapts(alg) === false
                @warn "`$(_warn_name(alg))` has no embedded error estimate in PETSc, so " *
                    "it steps at the requested dt and ignores reltol/abstol"
            end
            # SciML's defaults, as its other wrappers use, rather than PETSc's own 1e-4 for
            # both, so an unset tolerance is the one `integrator.opts` reports.
            _set_tolerances!(h, something(abstol, 1.0e-6), something(reltol, 1.0e-3))
            # A step PETSc cannot take is reported through the retcode rather
            # than raised, which leaves argument errors still raising.
            effective_options = ["-ts_error_if_step_fails", "false"]
            append!(effective_options, _default_options(alg))
            # PETSc's sparse factorisations do not pivot, and an algebraic row of an index-1
            # system has a zero on the diagonal, so rows are swapped to move it off.
            (h.jac_mat !== nothing && uses_sparse_jac || h.fd_mat !== nothing) &&
                append!(effective_options, ["-pc_factor_nonzeros_along_diagonal"])
            adaptive || append!(effective_options, ["-ts_adapt_type", "none"])
            # Told to keep going below the floor, the floor is PETSc's to clamp with,
            # since it takes the clamped step whatever its error.
            forced = force_dtmin && dtmin !== nothing && dtmin != 0
            forced &&
                append!(effective_options, ["-ts_adapt_dt_min", string(abs(Float64(dtmin)))])
            # With force_dtmin the floor wins over a smaller dtmax, as in OrdinaryDiffEq.
            (dtmax === nothing || isinf(dtmax)) || append!(
                effective_options,
                [
                    "-ts_adapt_dt_max",
                    string(_above(abs(Float64(dtmax)), forced ? abs(Float64(dtmin)) : 0.0)),
                ],
            )
            append!(effective_options, alg.petsc_options)
            append!(effective_options, extra_options)
            if !isempty(effective_options)
                parsed = PETSc.parse_options(effective_options)
                h.opts = PETSc.Options(petsclib; parsed...)
                push!(h.opts)
                try
                    LibPETSc.TSSetFromOptions(petsclib, ts)
                finally
                    pop!(h.opts)
                end
            else
                LibPETSc.TSSetFromOptions(petsclib, ts)
            end
            # An option can change the type, so the constructor's refusals are applied again
            # to the type PETSc will run.
            chosen = LibPETSc.TSGetType(petsclib, ts)
            !(alg isa TSMPRK) && haskey(_NEEDS_OTHER_SETUP, chosen) && throw(
                ArgumentError(
                    "PETScDiffEq cannot drive `$chosen`, which $(_NEEDS_OTHER_SETUP[chosen])",
                ),
            )
            _uses_ifunction(alg) && chosen in _EXPLICIT_ONLY && throw(
                ArgumentError(
                    "`$chosen` is an explicit PETSc type, so it needs " *
                        "`TSGeneric(\"$chosen\"; explicit = true)` rather than an option",
                ),
            )
            # IRK builds its coupled-stage matrix from an AIJ Jacobian and takes no other,
            # whichever way it was asked for.
            if chosen == "irk" && has_jac && !uses_sparse_jac
                PETSc.destroy(h.jac_mat)
                h.jac_mat = PETSc.MatSeqAIJ(petsclib, n, n, n)
                LibPETSc.TSSetIJacobian(
                    petsclib, ts, h.jac_mat, h.jac_mat, IJACOBIAN_PTR[], ctxptr,
                )
            end
            # The options have reached the linear solve by here.
            h.pivot_raises = _pivot_raises(petsclib, ts) ||
                _option_flag(effective_options, "ts_error_if_step_fails")
            if !dt_given &&
                    LibPETSc.TSAdaptGetType(petsclib, LibPETSc.TSGetAdapt(petsclib, ts)) == "none"
                throw(
                    ArgumentError(
                        "PETScDiffEq needs `dt` here: PETSc will step this solve at a fixed " *
                            "size, since the method has no embedded error estimate or " *
                            "`-ts_adapt_type none` is set",
                    ),
                )
            end
            # An option can pick another type or subtype than the one named, and then, as
            # for TSGeneric, whether PETSc interpolates is only known by asking.
            running = _running_name(petsclib, ts)
            if running != ctx.alg_name
                ctx.hermite = !has_mass && !is_dae
                ctx.interpolates = nothing
                ctx.alg_name = running
            end
        end
    catch
        _destroy!(h)
        rethrow()
    end
    return h
end

function _read_stats(h::TSHandles)
    pl, ts = h.petsclib, h.ts
    return (
        reason = LibPETSc.TSGetConvergedReason(pl, ts),
        nsteps = Int(LibPETSc.TSGetStepNumber(pl, ts)),
        nreject = Int(LibPETSc.TSGetStepRejections(pl, ts)),
        nnonliniter = Int(LibPETSc.TSGetSNESIterations(pl, ts)),
        nnonlinfail = Int(LibPETSc.TSGetSNESFailures(pl, ts)),
    )
end

function _assemble(prob, alg, h::TSHandles, tend, uend, st)
    ctx = h.ctx
    tf, t0, tol = h.tf, h.t0, 100 * eps(max(one(Float64), abs(h.tf)))
    # `-ts_exact_final_time interpolate` steps past tf and then interpolates
    # back, so the monitor reports an overshoot point mid-sequence, not last.
    keep = findall(t -> t <= tf + tol, ctx.ts)
    if length(keep) != length(ctx.ts)
        ctx.ts = ctx.ts[keep]
        ctx.us = ctx.us[keep]
        ctx.dense && (ctx.dus = ctx.dus[keep])
    end
    if h.save_end && (
            isempty(ctx.ts) || ctx.ts[end] < tend - tol ||
                (ctx.ts[end] <= tend + tol && ctx.us[end] != _saved(ctx, uend))
        )
        # A step that stopped part way through leaves the time already recorded, with the
        # state it had then, so the end replaces that point rather than repeating its time.
        if !isempty(ctx.ts) && abs(ctx.ts[end] - tend) <= tol
            pop!(ctx.ts)
            pop!(ctx.us)
            ctx.dense && pop!(ctx.dus)
        end
        _record!(ctx, tend, uend)
    end
    if !h.save_end && !isempty(ctx.ts) && abs(ctx.ts[end] - tf) <= tol
        pop!(ctx.ts)
        pop!(ctx.us)
        ctx.dense && pop!(ctx.dus)
    end
    if !h.save_start && length(ctx.ts) > 1 && abs(ctx.ts[1] - t0) <= tol
        popfirst!(ctx.ts)
        popfirst!(ctx.us)
        ctx.dense && popfirst!(ctx.dus)
    end

    finite = all(isfinite, uend)
    # PETSc's own reason says what stopped a solve short of the final time.
    retcode = if !finite || h.stopped == PETSC_ERR_FP || ctx.unstable_hit
        SciMLBase.ReturnCode.Unstable
    elseif tend >= tf - tol
        SciMLBase.ReturnCode.Success
    elseif st.nsteps >= h.maxiters
        SciMLBase.ReturnCode.MaxIters
    elseif st.reason == LibPETSc.TS_DIVERGED_NONLINEAR_SOLVE
        SciMLBase.ReturnCode.ConvergenceFailure
    elseif ctx.dt_too_small || st.reason == LibPETSc.TS_DIVERGED_STEP_REJECTED
        SciMLBase.ReturnCode.DtLessThanMin
    else
        SciMLBase.ReturnCode.Failure
    end
    # A method that solves implicitly calls the Jacobian on its first step.
    # One that never calls it is not solving implicitly, whatever it reports. A Jacobian
    # this package built is not asked about, since options such as `-snes_mf` leave any
    # Jacobian unused on purpose.
    if h.jac_mat !== nothing && h.ad_calls === nothing && st.nsteps > 0 && ctx.njacs == 0
        @warn "`$(_ts_type(alg))` took $(st.nsteps) steps without ever calling the " *
            "Jacobian this package gave PETSc, so it is not solving implicitly and the " *
            "result should not be trusted; an explicit PETSc type needs `explicit = true`"
    end
    stats = SciMLBase.DEStats(
        _nf(h), ctx.nf2, -1, -1, ctx.njacs, st.nnonliniter, st.nnonlinfail, -1, -1, -1,
        st.nsteps, st.nreject, 0.0,
    )
    ts, dus = _user_time(h)
    return SciMLBase.build_solution(
        prob, alg, ts, ctx.us; retcode = retcode, stats = stats,
        dense = ctx.dense, interp = _interp(ctx, ts, dus),
    )
end

function _solve_unlocked(
        prob::SupportedProblem, alg::AnyPETScTS;
        callback = nothing, tstops = (), d_discontinuities = (), kwargs...,
    )
    # PETSc's MPRK step never shortens itself onto the final time, so it runs
    # through the integrator, which shortens the last step for it.
    # A step outside the caller's domain is taken again smaller, which only the
    # integrator's own loop can do.
    if !_no_callback(callback) || !isempty(tstops) || !isempty(d_discontinuities) ||
            alg isa TSMPRK || get(kwargs, :isoutofdomain, nothing) !== nothing
        return SciMLBase.solve!(
            SciMLBase.__init(
                prob, alg; callback = callback, tstops = tstops,
                d_discontinuities = d_discontinuities, kwargs...,
            ),
        )
    end
    h = _setup(prob, alg; kwargs...)
    ctx, pl = h.ctx, h.petsclib
    tend, uend, st = h.t0, copy(h.u0), nothing
    try
        GC.@preserve ctx begin
            try
                _quiet_errors(h) do
                    LibPETSc.TSSolve(pl, h.ts, h.u)
                end
            catch e
                # A callback that threw reports failure to PETSc, which raises a
                # PetscError here. The user's own exception is the useful one.
                ctx.err === nothing && !_failed_step(e, h) && rethrow()
                h.stopped = e.code
            end
        end
        ctx.err === nothing || throw(ctx.err)
        h.stopped == 0 || _warn_failed_step(alg, h.stopped)
        # PETSc records the solve time only as TSSolve returns, so a raised step leaves
        # the time and state at the last finished one.
        tend = Float64(
            h.stopped == 0 ? LibPETSc.TSGetSolveTime(pl, h.ts) : LibPETSc.TSGetTime(pl, h.ts),
        )
        st = _read_stats(h)
        uend = _readvec!(similar(h.u0), pl, h.u)
    finally
        _destroy!(h)
    end
    return _assemble(prob, alg, h, tend, uend, st)
end

SciMLBase.__solve(prob::SupportedProblem, alg::AnyPETScTS; kwargs...) =
    _locked(() -> _solve_unlocked(prob, alg; kwargs...))

# Callbacks written against OrdinaryDiffEq reach for `integrator.opts` and set
# tolerances or a step cap mid-solve, so writes here reach PETSc.
mutable struct PETScIntegratorOpts{H}
    h::H
    adaptive::Bool
    abstol::Any
    reltol::Any
    dtmin::Float64
    dtmax::Float64
    verbose::Bool
    force_dtmin::Bool
end

function _setopt_unlocked(o::PETScIntegratorOpts, name::Symbol, v)
    setfield!(o, name, name in (:dtmin, :dtmax) ? Float64(v) : v)
    h = getfield(o, :h)
    (h === nothing || h.destroyed) && return v
    pl = h.petsclib
    if name === :abstol || name === :reltol
        _set_tolerances!(h, getfield(o, :abstol), getfield(o, :reltol))
    elseif name === :dtmin && getfield(o, :force_dtmin)
        # Under force_dtmin the floor is PETSc's clamp, and moving it keeps it so.
        adapt = LibPETSc.TSGetAdapt(pl, h.ts)
        _, hi = LibPETSc.TSAdaptGetStepLimits(pl, adapt)
        lo = abs(getfield(o, :dtmin))
        LibPETSc.TSAdaptSetStepLimits(pl, adapt, lo, _above(hi, lo))
    elseif name === :dtmin
        # The floor is kept here rather than by PETSc, whose own floor takes the step anyway.
        h.ctx.dtmin = _floor(getfield(o, :dtmin), false, getfield(o, :adaptive))
    elseif name === :dtmax
        adapt = LibPETSc.TSGetAdapt(pl, h.ts)
        lo, _ = LibPETSc.TSAdaptGetStepLimits(pl, adapt)
        hi = abs(getfield(o, :dtmax))
        LibPETSc.TSAdaptSetStepLimits(pl, adapt, lo, isfinite(hi) ? _above(hi, lo) : 1.0e308)
    end
    return v
end

Base.setproperty!(o::PETScIntegratorOpts, name::Symbol, v) =
    _locked(() -> _setopt_unlocked(o, name, v))

"""
    PETScIntegrator

The integrator `SciMLBase.init` returns for a PETSc TS algorithm. Step it with
`step!`, run it to the end with `solve!`, stop it early with `terminate!` and
restart it with `reinit!`. Between steps `u`, `uprev`, `t`, `tprev` and `dt`
are readable, and `add_tstop!` schedules a time to land on exactly.
"""
mutable struct PETScIntegrator{Alg, P, H, Pr, CB, CC} <:
    SciMLBase.AbstractODEIntegrator{Alg, true, Vector{Float64}, Float64}
    alg::Alg
    u::Vector{Float64}
    uprev::Vector{Float64}
    t::Float64
    tprev::Float64
    dt::Float64
    tdir::Float64
    p::P
    h::H
    prob::Pr
    callbacks::CB
    continuous::CC
    f::Any
    opts::Any
    ucache::Vector{Float64}
    tmp1::Vector{Float64}
    tmp2::Vector{Float64}
    event_t::Vector{Vector{Float64}}
    event_residual::Vector{Vector{Float64}}
    kwargs::Any
    tstops::Vector{Float64}
    tstops_cache::Vector{Float64}
    # The d_discontinuities of this run, in the caller's time, which are also in `tstops`,
    # and those given to `init`, which `reinit!` goes back to.
    d_discontinuities::Vector{Float64}
    d_discontinuities_cache::Vector{Float64}
    dtcache::Float64
    sol::Any
    finished::Bool
    derivative_discontinuity::Bool
end

# `solve` can pass an empty `CallbackSet` where no callback was given.
_no_callback(cb) = cb === nothing || (
    cb isa SciMLBase.CallbackSet &&
        isempty(cb.discrete_callbacks) && isempty(cb.continuous_callbacks)
)

_split_callbacks(::Nothing) = ((), ())
_split_callbacks(cb::SciMLBase.DiscreteCallback) = ((cb,), ())
_split_callbacks(cb::SciMLBase.ContinuousCallback) = ((), (cb,))
function _split_callbacks(cb::SciMLBase.CallbackSet)
    for c in cb.continuous_callbacks
        _check_continuous(c)
    end
    return (cb.discrete_callbacks, cb.continuous_callbacks)
end
_split_callbacks(cb::SciMLBase.AbstractContinuousCallback) = (_check_continuous(cb); ((), (cb,)))
_check_continuous(::SciMLBase.ContinuousCallback) = nothing
_check_continuous(::SciMLBase.VectorContinuousCallback) = nothing
_check_continuous(cb) = throw(
    ArgumentError(
        "PETScDiffEq does not support $(typeof(cb).name.name); use a ContinuousCallback, " *
            "a VectorContinuousCallback or a DiscreteCallback",
    ),
)

function _discontinuity_unlocked(integ::PETScIntegrator, bool::Bool)
    integ.derivative_discontinuity = bool
    bool && (integ.h.ctx.pdirty = true)
    return nothing
end

SciMLBase.derivative_discontinuity!(integ::PETScIntegrator, bool::Bool) =
    _locked(() -> _discontinuity_unlocked(integ, bool))

SciMLBase.get_dt(integ::PETScIntegrator) = integ.dt
function _proposed_dt_unlocked(integ::PETScIntegrator)
    integ.finished && return abs(integ.dt)
    return Float64(LibPETSc.TSGetTimeStep(integ.h.petsclib, integ.h.ts))
end

SciMLBase.get_proposed_dt(integ::PETScIntegrator) = _locked(() -> _proposed_dt_unlocked(integ))
function _set_proposed_dt_unlocked(integ::PETScIntegrator, dt)
    integ.finished || LibPETSc.TSSetTimeStep(integ.h.petsclib, integ.h.ts, abs(Float64(dt)))
    return nothing
end

SciMLBase.set_proposed_dt!(integ::PETScIntegrator, dt) =
    _locked(() -> _set_proposed_dt_unlocked(integ, dt))
_make_opts(h, kwargs) = PETScIntegratorOpts(
    h, get(kwargs, :adaptive, true) === true,
    get(kwargs, :abstol, 1.0e-6), get(kwargs, :reltol, 1.0e-3),
    Float64(something(get(kwargs, :dtmin, nothing), 0.0)),
    Float64(something(get(kwargs, :dtmax, nothing), Inf)),
    get(kwargs, :verbose, true) === true, get(kwargs, :force_dtmin, false) === true,
)

SciMLBase.isadaptive(integ::PETScIntegrator) =
    getfield(integ.opts, :adaptive) && _adapts(integ.alg) !== false

# Interpolation inside the step just taken, the only one with both ends at hand.
(integ::PETScIntegrator)(t::Number) = copy(_state_at(integ, Float64(t)))
(integ::PETScIntegrator)(t::Number, ::Type{Val{0}}) = copy(_state_at(integ, Float64(t)))
(integ::PETScIntegrator)(out::AbstractArray, t) =
    copyto!(out, _state_at(integ, Float64(t)))
(integ::PETScIntegrator)(out::AbstractArray, t, ::Type{Val{0}}) =
    copyto!(out, _state_at(integ, Float64(t)))

SciMLBase.get_du(integ::PETScIntegrator) = integ.tdir .* _derivative(integ.h.ctx, integ.tdir * integ.t, integ.u)
function SciMLBase.get_du!(out, integ::PETScIntegrator)
    copyto!(out, integ.tdir .* _derivative(integ.h.ctx, integ.tdir * integ.t, integ.u))
    return out
end
SciMLBase.get_tmp_cache(integ::PETScIntegrator) = (integ.tmp1, integ.tmp2)
DiffEqBase.get_tstops(integ::PETScIntegrator) = integ.tstops
DiffEqBase.get_tstops_array(integ::PETScIntegrator) = integ.tstops
DiffEqBase.get_tstops_max(integ::PETScIntegrator) = last(integ.tstops)

function _set_u_unlocked(integ::PETScIntegrator, u)
    copyto!(integ.u, u)
    integ.finished && return nothing
    PETSc.withlocalarray!(
        ua -> copyto!(ua, integ.u), integ.h.u; read = false, write = true,
    )
    LibPETSc.TSRestartStep(integ.h.petsclib, integ.h.ts)
    return nothing
end

SciMLBase.set_u!(integ::PETScIntegrator, u) = _locked(() -> _set_u_unlocked(integ, u))

function _set_t_unlocked(integ::PETScIntegrator, t)
    integ.t = Float64(t)
    _end_step_here!(integ)
    integ.finished || LibPETSc.TSSetTime(integ.h.petsclib, integ.h.ts, integ.tdir * Float64(t))
    return nothing
end

SciMLBase.set_t!(integ::PETScIntegrator, t) = _locked(() -> _set_t_unlocked(integ, t))

function SciMLBase.add_saveat!(integ::PETScIntegrator, t)
    t = Float64(t)
    ctx = integ.h.ctx
    s = integ.tdir * t
    tol = 100 * eps(max(one(Float64), abs(integ.h.tf)))
    s < integ.tdir * integ.t - tol &&
        throw(ArgumentError("cannot add a saveat at $t, behind the current time $(integ.t)"))
    i = searchsortedfirst(ctx.saveat, s)
    (i <= length(ctx.saveat) && ctx.saveat[i] == s) || insert!(ctx.saveat, i, s)
    i < ctx.saveat_idx && (ctx.saveat_idx += 1)
    return nothing
end

# The state at an earlier time inside the step just taken.
function _change_t_unlocked(
        integ::PETScIntegrator, t, modify_save_endpoint::Type{Val{T}} = Val{false},
    ) where {T}
    integ.finished && return nothing
    copyto!(integ.u, _state_at(integ, Float64(t)))
    integ.t = Float64(t)
    _end_step_here!(integ)
    PETSc.withlocalarray!(
        ua -> copyto!(ua, integ.u), integ.h.u; read = false, write = true,
    )
    LibPETSc.TSSetTime(integ.h.petsclib, integ.h.ts, integ.tdir * Float64(t))
    LibPETSc.TSRestartStep(integ.h.petsclib, integ.h.ts)
    return nothing
end

SciMLBase.change_t_via_interpolation!(
    integ::PETScIntegrator, t, modify_save_endpoint::Type{Val{T}} = Val{false},
) where {T} = _locked(() -> _change_t_unlocked(integ, t, modify_save_endpoint))

# Returns `(saved, savedexactly)`: whether any point was saved, and whether one was saved at
# the current time. The current time is saved only when every step is, and not twice.
function _savevalues_unlocked(integ::PETScIntegrator, force_save = false)
    integ.finished && return (false, false)
    ctx = integ.h.ctx
    n = length(ctx.ts)
    _save_step!(integ, integ.t, false)
    s = integ.tdir * integ.t
    if force_save || (ctx.save_everystep && !_last_recorded(ctx, s))
        _record!(ctx, s, integ.u)
    end
    saved = length(ctx.ts) > n
    return (saved, saved && ctx.ts[end] == s)
end

SciMLBase.savevalues!(integ::PETScIntegrator, force_save = false) =
    _locked(() -> _savevalues_unlocked(integ, force_save))

# SciMLBase's version steps again once the span is done, which a finished integrator refuses.
function SciMLBase.step!(integ::PETScIntegrator, dt, stop_at_tdt = false)
    integ.tdir * dt < 0 && throw(ArgumentError("cannot step backward in time"))
    next_t = integ.t + dt
    tf = _user_t(integ.tdir, integ.h.tf)
    stop_at_tdt && integ.tdir * next_t < integ.tdir * tf && SciMLBase.add_tstop!(integ, next_t)
    while !integ.finished && integ.tdir * integ.t < integ.tdir * next_t
        SciMLBase.step!(integ)
    end
    return nothing
end

function _state_at_unlocked(integ::PETScIntegrator, t::Float64)
    # A rounding error from an end of the step is that end, so `integ(integ.t - integ.dt)`
    # is the step's start.
    tol = 100 * eps(max(one(Float64), abs(integ.t)))
    abs(t - integ.t) <= tol && return integ.u
    abs(t - integ.tprev) <= tol && return integ.uprev
    # Only the step just taken has both ends at hand.
    integ.tdir * integ.tprev <= integ.tdir * t <= integ.tdir * integ.t || throw(
        ArgumentError(
            "PETScDiffEq can only interpolate inside the step just taken, " *
                "$(integ.tprev) to $(integ.t), but $t was asked for",
        ),
    )
    return _interpolate!(integ, integ.tdir * t)
end

_state_at(integ::PETScIntegrator, t::Float64) = _locked(() -> _state_at_unlocked(integ, t))

# The state at PETSc's time s inside the step just taken, in `integ.ucache`.
function _interpolate!(integ::PETScIntegrator, s::Float64)
    h = integ.h
    ctx = h.ctx
    if !ctx.hermite
        integ.finished && return _interpolate_finished!(integ, s)
        v = _petsc_interpolate!(ctx, h.ts, s)
        v === nothing && throw(_no_interpolant(ctx))
        return _readvec!(integ.ucache, h.petsclib, v)
    end
    return _hermite!(
        integ.ucache, ctx, s, integ.tdir * integ.tprev, integ.uprev, ctx.end_s, ctx.end_u,
    )
end

# PETSc's interpolant goes with the TS, which a finished integrator has freed. The last
# step is then answered by the cubic Hermite interpolant the solution's dense output uses.
function _interpolate_finished!(integ::PETScIntegrator, s::Float64)
    ctx = integ.h.ctx
    (ctx.M === nothing && !ctx.dae) || throw(
        ArgumentError(
            "the integrator has finished and freed PETSc's interpolant, and a mass matrix " *
                "or a DAEProblem gives no derivative to build another from; interpolate " *
                "before the last step completes, or use the solution's saveat times",
        ),
    )
    return _hermite!(
        integ.ucache, ctx, s, integ.tdir * integ.tprev, integ.uprev,
        integ.tdir * integ.t, integ.u,
    )
end

# An affect! may change the parameters, so the derivatives of the step just taken are
# fixed before one runs, while the parameters are still those the step was taken with.
function _pin_step!(integ::PETScIntegrator)
    ctx = integ.h.ctx
    ctx.hermite || return nothing
    ctx.fstart === nothing &&
        (ctx.fstart = _derivative(ctx, integ.tdir * integ.tprev, integ.uprev))
    ctx.fend === nothing && (ctx.fend = _derivative(ctx, ctx.end_s, ctx.end_u))
    return nothing
end

# The integrator's step now ends at its current time and state.
function _end_step_here!(integ::PETScIntegrator)
    ctx = integ.h.ctx
    ctx.hermite || return nothing
    ctx.end_s = integ.tdir * integ.t
    copyto!(ctx.end_u, integ.u)
    ctx.fend = nothing
    return nothing
end

_ncond(::SciMLBase.ContinuousCallback) = 1
_ncond(cb::SciMLBase.VectorContinuousCallback) = cb.len

function _fill_conditions!(out, integ::PETScIntegrator, cb::SciMLBase.ContinuousCallback, t)
    out[1] = cb.condition(_state_at(integ, t), t, integ)
    return out
end

function _fill_conditions!(
        out, integ::PETScIntegrator, cb::SciMLBase.VectorContinuousCallback, t,
    )
    cb.condition(out, _state_at(integ, t), t, integ)
    return out
end

# SciMLBase's rule: a crossing counts only if the handler for its direction
# exists, and a condition that starts at zero is not a crossing. A vector
# callback has one handler for every direction, so only the zero rule applies.
function _is_event(prev, next, cb::SciMLBase.ContinuousCallback)
    return (
        (prev < 0 && cb.affect! !== nothing) || (prev > 0 && cb.affect_neg! !== nothing)
    ) && prev * next <= 0
end
_is_event(prev, next, ::SciMLBase.VectorContinuousCallback) = prev != 0 && prev * next <= 0

# DiffEqBase's root finder, the one OrdinaryDiffEq uses, run to the precision of the time
# type. `buf` is this search's own; the caller's condition values are never written.
function _event_root(integ::PETScIntegrator, cb, lo, hi, i::Int, buf)
    condition(t, _) = _fill_conditions!(buf, integ, cb, t)[i]
    return DiffEqBase.find_root(condition, (lo, hi), cb.rootfind)
end

# What gets handed to the user: the crossing direction for a scalar callback,
# and SciMLBase's per-component mask for a vector one, where +1 is a crossing
# from negative to positive and -1 the other way.
_crossing(::SciMLBase.ContinuousCallback, s0, keep, m) = s0[1]
function _crossing(::SciMLBase.VectorContinuousCallback, s0, keep, m)
    mask = zeros(Int8, m)
    for i in keep
        mask[i] = s0[i] < 0 ? Int8(1) : Int8(-1)
    end
    return mask
end

function _find_event(integ::PETScIntegrator, cb, k::Int)
    t0, t1 = integ.tprev, integ.t
    integ.tdir * (t1 - t0) > 0 || return nothing
    m = _ncond(cb)
    s0 = Vector{Float64}(undef, m)
    sk = Vector{Float64}(undef, m)
    buf = Vector{Float64}(undef, m)
    _fill_conditions!(s0, integ, cb, t0)
    # A component that fired at t0, and whose condition the affect! left within
    # `cb.abstol` of its value at the root, still sits on that root. Its sign is read
    # and its search starts a nudge past t0, so the root it fired on stays behind it.
    ev, residual = integ.event_t[k], integ.event_residual[k]
    nudged = [ev[i] == t0 && abs(s0[i] - residual[i]) <= cb.abstol for i in 1:m]
    start = fill(t0, m)
    if any(nudged)
        tn = t0 + (t1 - t0) * Float64(cb.repeat_nudge)
        _fill_conditions!(buf, integ, cb, tn)
        for i in 1:m
            nudged[i] && (s0[i] = buf[i]; start[i] = tn)
        end
    end
    past(i, t) = integ.tdir * t > integ.tdir * start[i]
    if cb.rootfind === SciMLBase.NoRootFind
        _fill_conditions!(sk, integ, cb, t1)
        hit = [i for i in 1:m if _is_event(s0[i], sk[i], cb)]
        return isempty(hit) ? nothing : (t1, _crossing(cb, s0, hit, m))
    end
    lo = t0
    n = max(cb.interp_points, 1)
    for k in 1:n
        tk = k == n ? t1 : t0 + (t1 - t0) * (k / n)
        _fill_conditions!(sk, integ, cb, tk)
        hit = [i for i in 1:m if past(i, tk) && _is_event(s0[i], sk[i], cb)]
        if !isempty(hit)
            roots = [
                sk[i] == 0 ? tk :
                    _event_root(integ, cb, past(i, lo) ? lo : start[i], tk, i, buf)
                    for i in hit
            ]
            first_root = roots[argmin(integ.tdir .* roots)]
            # Roots this close to the first are one crossing seen through rounding.
            together = 10 * eps(first_root)
            keep = [
                hit[j] for j in eachindex(hit)
                    if integ.tdir * roots[j] <= integ.tdir * first_root + together
            ]
            return (first_root, _crossing(cb, s0, keep, m))
        end
        for i in 1:m
            past(i, tk) && (s0[i] = sk[i])
        end
        lo = tk
    end
    return nothing
end

_mark_fired!(ev, ::SciMLBase.ContinuousCallback, _, t) = (ev[1] = t; nothing)
function _mark_fired!(ev, ::SciMLBase.VectorContinuousCallback, mask, t)
    for i in eachindex(mask)
        mask[i] == 0 || (ev[i] = t)
    end
    return nothing
end

function _fire!(integ::PETScIntegrator, cb::SciMLBase.ContinuousCallback, prevsign)
    if prevsign < 0
        cb.affect! === nothing ? (integ.derivative_discontinuity = false) : cb.affect!(integ)
    else
        cb.affect_neg! === nothing ? (integ.derivative_discontinuity = false) :
            cb.affect_neg!(integ)
    end
    return nothing
end

function _fire!(integ::PETScIntegrator, cb::SciMLBase.VectorContinuousCallback, mask)
    cb.affect! === nothing ? (integ.derivative_discontinuity = false) :
        cb.affect!(integ, mask)
    return nothing
end

# The state is put back into PETSc's own vector and the stepper restarted, so a
# multistep method drops history taken across the event.
function _rollback!(integ::PETScIntegrator, t::Float64, dt::Float64, interpolate::Bool)
    h = integ.h
    pl = h.petsclib
    if interpolate && t != integ.t
        copyto!(integ.u, _interpolate!(integ, integ.tdir * t))
        integ.t = t
        _end_step_here!(integ)
    end
    integ.t = t
    # The step that just ended keeps the derivatives it was taken with. An affect! may
    # have changed the parameters, which only the next step's start derivative answers to.
    h.ctx.pdirty = true
    PETSc.withlocalarray!(ua -> copyto!(ua, integ.u), h.u; read = false, write = true)
    LibPETSc.TSSetTime(pl, h.ts, integ.tdir * t)
    LibPETSc.TSSetTimeStep(pl, h.ts, integ.tdir * dt)
    LibPETSc.TSRestartStep(pl, h.ts)
    integ.dt = integ.t - integ.tprev
    return nothing
end

# Returns whether an event was applied, in which case the step ends at the root
# rather than where PETSc stopped.
function _apply_continuous_callbacks!(integ::PETScIntegrator, dt::Float64)
    isempty(integ.continuous) && return false
    best, best_cb, best_crossing, best_k = nothing, nothing, nothing, 0
    for (k, cb) in enumerate(integ.continuous)
        found = _find_event(integ, cb, k)
        found === nothing && continue
        if best === nothing || integ.tdir * found[1] < integ.tdir * best
            best, best_cb, best_crossing, best_k = found[1], cb, found[2], k
        end
    end
    best === nothing && return false
    ctx = integ.h.ctx
    _save_step!(integ, best, false; slack = 0.0)
    _rollback!(integ, best, dt, true)
    # A repeat is judged against the condition at the root before the affect! runs. An
    # event found without root finding is at no root, and is judged against zero.
    residual = integ.event_residual[best_k]
    best_cb.rootfind === SciMLBase.NoRootFind ? fill!(residual, 0.0) :
        _fill_conditions!(residual, integ, best_cb, integ.t)
    best_cb.save_positions[1] && _record!(ctx, integ.tdir * integ.t, integ.u)
    integ.derivative_discontinuity = true
    _pin_step!(integ)
    _fire!(integ, best_cb, best_crossing)
    integ.finished && return true
    _rollback!(integ, integ.t, dt, false)
    _mark_fired!(integ.event_t[best_k], best_cb, best_crossing, integ.t)
    best_cb.save_positions[2] && _record!(ctx, integ.tdir * integ.t, integ.u)
    return true
end

function _apply_callbacks!(integ::PETScIntegrator)
    h = integ.h
    ctx = h.ctx
    for cb in integ.callbacks
        integ.finished && return nothing
        cb.condition(integ.u, integ.t, integ) || continue
        # An affect! is assumed to change the state unless it says otherwise
        # through derivative_discontinuity!(integ, false).
        integ.derivative_discontinuity = true
        _pin_step!(integ)
        cb.affect!(integ)
        integ.finished && return nothing
        if integ.derivative_discontinuity
            # The changed state has to reach PETSc's own solution vector, and a
            # multistep method must drop history taken before the jump.
            PETSc.withlocalarray!(ua -> copyto!(ua, integ.u), h.u; read = false, write = true)
            LibPETSc.TSRestartStep(h.petsclib, h.ts)
            ctx.pdirty = true
        end
        cb.save_positions[2] && _record!(ctx, integ.tdir * integ.t, integ.u)
    end
    return nothing
end

# An initialize may change the state, and PETSc steps from its own vector rather
# than from `integ.u`.
function _initialize_callbacks!(integ::PETScIntegrator, initialize_save::Bool)
    h = integ.h
    cbs = (integ.callbacks..., integ.continuous...)
    before = copy(integ.u)
    for cb in cbs
        cb.initialize(cb, integ.u, integ.t, integ)
    end
    integ.derivative_discontinuity = false
    integ.u == before && return nothing
    copyto!(integ.uprev, integ.u)
    PETSc.withlocalarray!(ua -> copyto!(ua, integ.u), h.u; read = false, write = true)
    LibPETSc.TSRestartStep(h.petsclib, h.ts)
    initialize_save && any(cb -> cb.save_positions[2], cbs) &&
        _record!(h.ctx, h.tdir * integ.t, integ.u)
    return nothing
end

function _init_unlocked(
        prob::SupportedProblem, alg::AnyPETScTS;
        callback = nothing, tstops = (), d_discontinuities = (), kwargs...,
    )
    tstops, d_discontinuities = _times(tstops), _times(d_discontinuities)
    stops_given = vcat(tstops, d_discontinuities)
    callbacks, continuous = _split_callbacks(callback)
    h = _setup(prob, alg; tstops = stops_given, kwargs...)
    LibPETSc.TSSetUp(h.petsclib, h.ts)
    _initial_save!(h)
    stops = _tstops(stops_given, h)
    dt0 = h.tdir * Float64(LibPETSc.TSGetTimeStep(h.petsclib, h.ts))
    integ = PETScIntegrator(
        alg, copy(h.u0), copy(h.u0), _user_t(h.tdir, h.t0), _user_t(h.tdir, h.t0),
        dt0, h.tdir,
        prob.p, h, prob, callbacks, continuous, prob.f, _make_opts(h, kwargs),
        copy(h.u0), similar(h.u0), similar(h.u0),
        Vector{Float64}[fill(NaN, _ncond(cb)) for cb in continuous],
        Vector{Float64}[fill(NaN, _ncond(cb)) for cb in continuous], NamedTuple(kwargs),
        stops, tstops, d_discontinuities, d_discontinuities, dt0,
        _initial_solution(prob, alg, h), false, false,
    )
    _initialize_callbacks!(integ, true)
    _past_discontinuity!(integ)
    return integ
end

SciMLBase.__init(prob::SupportedProblem, alg::AnyPETScTS; kwargs...) =
    _locked(() -> _init_unlocked(prob, alg; kwargs...))

# The queue holds `tdir * t` for each stop, the final time included, in increasing order and
# without repeats. step! drops each stop the integrator reaches, and terminate! empties it.
# A step of an adaptive solve that leaves the caller's domain is undone and taken again at a
# fifth of its size, OrdinaryDiffEq's default qmin. With `force_dtmin` a step that would fall
# below `dtmin` is taken at `dtmin` whatever the domain says, as OrdinaryDiffEq takes it;
# otherwise the solve ends where it last was in the domain, with DtLessThanMin below the
# caller's floor and Unstable below a rounding error of the span. The step before the undone
# one stays the step just taken. PETSc's rejection counter does not see these.
function _reject_out_of_domain!(integ::PETScIntegrator, before)
    h = integ.h
    ctx, pl = h.ctx, h.petsclib
    nstep, integ.dt, integ.dtcache, ctx.pdirty, outer = before
    taken = abs(integ.t - integ.tprev)
    smaller = taken / 5
    integ.t = integ.tprev
    copyto!(integ.u, integ.uprev)
    PETSc.withlocalarray!(ua -> copyto!(ua, integ.u), h.u; read = false, write = true)
    LibPETSc.TSSetTime(pl, h.ts, integ.tdir * integ.t)
    LibPETSc.TSSetStepNumber(pl, h.ts, LibPETSc.PetscInt(nstep))
    floor = abs(Float64(something(get(integ.kwargs, :dtmin, nothing), 0.0)))
    if get(integ.kwargs, :force_dtmin, false) === true && smaller < floor
        LibPETSc.TSSetTimeStep(pl, h.ts, min(floor, taken))
        LibPETSc.TSRestartStep(pl, h.ts)
        domain, ctx.domain = ctx.domain, nothing
        try
            return _step_unlocked(integ, outer)
        finally
            ctx.domain = domain
        end
    end
    below_floor = ctx.dtmin > 0 && smaller < ctx.dtmin
    if below_floor || smaller < 100 * eps(max(abs(integ.t), abs(h.t0), abs(h.tf)))
        below_floor ? (ctx.dt_too_small = true) : (ctx.unstable_hit = true)
        integ.tprev = outer[1]
        copyto!(integ.uprev, outer[2])
        ctx.fstart = nothing
        _finish!(integ)
        return nothing
    end
    LibPETSc.TSSetTimeStep(pl, h.ts, smaller)
    LibPETSc.TSRestartStep(pl, h.ts)
    return _step_unlocked(integ, outer)
end

# PETSc wants a largest step strictly above the smallest; where the floor wins, it pins the
# step there.
_above(hi, lo) = hi > lo ? hi : nextfloat(lo)

function _take_written_state!(integ::PETScIntegrator)
    h = integ.h
    _readvec!(integ.ucache, h.petsclib, h.u) == integ.u && return nothing
    PETSc.withlocalarray!(ua -> copyto!(ua, integ.u), h.u; read = false, write = true)
    LibPETSc.TSRestartStep(h.petsclib, h.ts)
    h.ctx.pdirty = true
    return nothing
end

# A single time is a list of one, as OrdinaryDiffEq takes it.
_times(ts) = ts isa Number ? [Float64(ts)] : Vector{Float64}(collect(Float64, ts))

# SciML's d_discontinuities are right-continuous: the step onto `t_d` ends in the old regime,
# and the next one starts a ULP past it, where the right-hand side is in the new one. An entry
# at the start moves the start the same way.
function _past_discontinuity!(integ::PETScIntegrator)
    integ.t in integ.d_discontinuities || return nothing
    h = integ.h
    s = nextfloat(integ.tdir * integ.t)
    integ.t = _user_t(integ.tdir, s)
    LibPETSc.TSSetTime(h.petsclib, h.ts, s)
    LibPETSc.TSRestartStep(h.petsclib, h.ts)
    h.ctx.pdirty = true
    return nothing
end

function _tstops(tstops, h::TSHandles)
    stops = sort!(unique!(h.tdir .* _times(tstops)))
    filter!(s -> h.t0 < s < h.tf, stops)
    return push!(stops, h.tf)
end

function _add_tstop_unlocked(integ::PETScIntegrator, t)
    t = Float64(t)
    s = integ.tdir * t
    s < integ.tdir * integ.t &&
        throw(ArgumentError("cannot add a tstop at $t, behind the current time $(integ.t)"))
    s > integ.h.tf && throw(
        ArgumentError(
            "cannot add a tstop at $t, beyond the final time $(integ.tdir * integ.h.tf); " *
                "PETScDiffEq cannot integrate past the problem's tspan",
        ),
    )
    i = searchsortedfirst(integ.tstops, s)
    (i <= length(integ.tstops) && integ.tstops[i] == s) || insert!(integ.tstops, i, s)
    return nothing
end

SciMLBase.add_tstop!(integ::PETScIntegrator, t) = _locked(() -> _add_tstop_unlocked(integ, t))
SciMLBase.has_tstop(integ::PETScIntegrator) = !isempty(integ.tstops)
# Both report the queue key, `integ.tdir * t`, which is what generic callback code
# compares against.
SciMLBase.first_tstop(integ::PETScIntegrator) = integ.tstops[1]
SciMLBase.pop_tstop!(integ::PETScIntegrator) = popfirst!(integ.tstops)

function _initial_save!(h::TSHandles)
    ctx = h.ctx
    tol = 100 * eps(max(one(Float64), abs(h.tf)))
    landed = false
    while ctx.saveat_idx <= length(ctx.saveat) && ctx.saveat[ctx.saveat_idx] <= h.t0 + tol
        _record_end!(ctx, ctx.saveat[ctx.saveat_idx], h.u0)
        ctx.saveat_idx += 1
        landed = true
    end
    h.save_start && !landed && _record_end!(ctx, h.t0, h.u0)
    return nothing
end

# Recorded times and derivatives are in PETSc's forward-running time.
_user_time(h::TSHandles) = h.tdir > 0 ? (h.ctx.ts, h.ctx.dus) :
    (_user_t.(h.tdir, h.ctx.ts), [-d for d in h.ctx.dus])

_nf(h::TSHandles) = h.ctx.nf + (h.ad_calls === nothing ? 0 : h.ad_calls[])

function _initial_solution(prob, alg, h::TSHandles)
    ts, dus = _user_time(h)
    return SciMLBase.build_solution(
        prob, alg, ts, h.ctx.us; retcode = SciMLBase.ReturnCode.Default,
        dense = h.ctx.dense, interp = _interp(h.ctx, ts, dus), stats = SciMLBase.DEStats(0),
    )
end

# The counters a callback can read part way through the solve, as in OrdinaryDiffEq.
function _live_stats!(integ::PETScIntegrator)
    stats = integ.sol.stats
    stats === nothing && return nothing
    ctx, st = integ.h.ctx, _read_stats(integ.h)
    stats.nf, stats.nf2, stats.njacs = _nf(integ.h), ctx.nf2, ctx.njacs
    stats.nnonliniter, stats.nnonlinconvfail = st.nnonliniter, st.nnonlinfail
    stats.naccept, stats.nreject = st.nsteps, st.nreject
    return nothing
end

# A reinitialised integrator gets fresh PETSc objects built from the keywords
# given to `init`, so its solve is identical to a fresh one and it works even
# after the previous solve released them.
function _reinit_unlocked(
        integ::PETScIntegrator, u0 = integ.prob.u0;
        t0 = integ.prob.tspan[1], tf = integ.prob.tspan[2],
        erase_sol = true, saveat = nothing, tstops = integ.tstops_cache,
        d_discontinuities = integ.d_discontinuities_cache,
        reinit_callbacks = true, initialize_save = true,
    )
    tstops, d_discontinuities = _times(tstops), _times(d_discontinuities)
    old = integ.h
    prob = SciMLBase.remake(integ.prob; u0 = u0, tspan = (t0, tf))
    setup_kwargs = saveat === nothing ? integ.kwargs : merge(integ.kwargs, (saveat = saveat,))
    h = _setup(prob, integ.alg; tstops = vcat(tstops, d_discontinuities), setup_kwargs...)
    LibPETSc.TSSetUp(h.petsclib, h.ts)
    if !erase_sol
        append!(h.ctx.ts, old.ctx.ts)
        append!(h.ctx.us, old.ctx.us)
        if h.ctx.dense
            if old.ctx.dense
                append!(h.ctx.dus, old.ctx.dus)
            else
                # A Hermite interpolant needs a derivative for every point it
                # holds, and the kept run saved none.
                for (t, u) in zip(old.ctx.ts, old.ctx.us)
                    push!(h.ctx.dus, _derivative(h.ctx, t, u))
                end
            end
        end
    end
    initialize_save && _initial_save!(h)
    _destroy!(old)
    integ.h = h
    integ.u = copy(h.u0)
    integ.uprev = copy(h.u0)
    integ.f = integ.prob.f
    integ.opts = _make_opts(h, integ.kwargs)
    integ.ucache = copy(h.u0)
    integ.tmp1 = similar(h.u0)
    integ.tmp2 = similar(h.u0)
    integ.tdir = h.tdir
    integ.t = _user_t(h.tdir, h.t0)
    integ.tprev = _user_t(h.tdir, h.t0)
    integ.dt = h.tdir * Float64(LibPETSc.TSGetTimeStep(h.petsclib, h.ts))
    integ.dtcache = integ.dt
    integ.tstops = _tstops(vcat(tstops, d_discontinuities), h)
    integ.d_discontinuities = d_discontinuities
    integ.finished = false
    for ev in integ.event_t
        fill!(ev, NaN)
    end
    integ.derivative_discontinuity = false
    integ.sol = _initial_solution(integ.prob, integ.alg, h)
    reinit_callbacks && _initialize_callbacks!(integ, initialize_save)
    _past_discontinuity!(integ)
    return nothing
end

SciMLBase.reinit!(integ::PETScIntegrator, u0 = integ.prob.u0; kwargs...) =
    _locked(() -> _reinit_unlocked(integ, u0; kwargs...))

@static if isdefined(SciMLBase, :has_reinit)
    SciMLBase.has_reinit(::PETScIntegrator) = true
end

function _finish!(integ::PETScIntegrator, retcode = nothing)
    integ.finished && return nothing
    h = integ.h
    for cb in (integ.callbacks..., integ.continuous...)
        cb.finalize(cb, integ.u, integ.t, integ)
    end
    st = _read_stats(h)
    sol = _assemble(integ.prob, integ.alg, h, integ.tdir * integ.t, copy(integ.u), st)
    integ.sol = retcode === nothing ? sol : SciMLBase.solution_new_retcode(sol, retcode)
    integ.finished = true
    _destroy!(h)
    return nothing
end

function _terminate_unlocked(
        integ::PETScIntegrator, retcode = SciMLBase.ReturnCode.Terminated,
    )
    empty!(integ.tstops)
    _finish!(integ, retcode)
    return nothing
end

SciMLBase.terminate!(integ::PETScIntegrator, retcode = SciMLBase.ReturnCode.Terminated) =
    _locked(() -> _terminate_unlocked(integ, retcode))

# `slack` lets a point a rounding error past `upto` count as reached. Before an event it is
# 0, so a point just after the root waits for the state the event leaves.
function _save_step!(
        integ::PETScIntegrator, upto::Float64, endpoint::Bool;
        slack = 100 * eps(max(one(Float64), abs(integ.h.tf))),
    )
    h = integ.h
    ctx = h.ctx
    tol = 100 * eps(max(one(Float64), abs(h.tf)))
    landed = false
    while ctx.saveat_idx <= length(ctx.saveat) &&
            ctx.saveat[ctx.saveat_idx] <= integ.tdir * upto + slack
        want = ctx.saveat[ctx.saveat_idx]
        if abs(want - integ.tdir * integ.t) <= tol
            _record_end!(ctx, want, integ.u)
            landed = true
        else
            _record!(ctx, want, _interpolate!(integ, want))
        end
        ctx.saveat_idx += 1
    end
    endpoint && ctx.save_everystep && !landed &&
        _record_end!(ctx, integ.tdir * upto, integ.u)
    return nothing
end

# `outer` is the start of the last step taken, kept across the retries of an undone step.
function _step_unlocked(integ::PETScIntegrator, outer = nothing)
    integ.finished && throw(
        ArgumentError(
            "this integrator has finished at t = $(integ.t) and cannot step further; " *
                "call reinit! to restart it",
        ),
    )
    h = integ.h
    ctx, pl = h.ctx, h.petsclib
    # The cap is on steps taken, so a cap already reached takes none.
    if Int(LibPETSc.TSGetStepNumber(pl, h.ts)) >= h.maxiters
        _finish!(integ)
        return nothing
    end
    # A state written to `integ.u` since the last step, as OrdinaryDiffEq allows, is the
    # one the step starts from; PETSc steps from its own vector.
    outer === nothing && _take_written_state!(integ)
    # PETSc's proposal for the next step; `integ.dt` is the step last taken.
    dtprev = integ.tdir * Float64(LibPETSc.TSGetTimeStep(pl, h.ts))
    before = (
        Int(LibPETSc.TSGetStepNumber(pl, h.ts)), integ.dt, integ.dtcache, ctx.pdirty,
        outer === nothing && ctx.domain !== nothing ? (integ.tprev, copy(integ.uprev)) : outer,
    )
    copyto!(integ.uprev, integ.u)
    integ.tprev = integ.t
    # The last end's derivative starts this step only if nothing has moved that end since.
    unmoved = integ.tdir * integ.t == ctx.end_s && integ.u == ctx.end_u
    ctx.fstart = unmoved && !ctx.pdirty ? ctx.fend : nothing
    ctx.pdirty = false
    tol = 100 * eps(max(one(Float64), abs(h.tf)))
    while !isempty(integ.tstops) && integ.tstops[1] <= integ.tdir * integ.t + _near(integ.t)
        popfirst!(integ.tstops)
    end
    # PETSc lands on its max time exactly but keeps the shortened step
    # afterwards; the cached dt is the last one chosen with no stop in the way.
    stop = !isempty(integ.tstops) && integ.tstops[1] < h.tf - tol ? integ.tstops[1] : nothing
    target = stop === nothing ? h.tf : stop
    LibPETSc.TSSetMaxTime(pl, h.ts, target)
    # TSAdaptChoose rejects a step that reaches past the max time, so the step
    # onto the target is shortened here rather than by PETSc.
    if Float64(LibPETSc.TSGetTimeStep(pl, h.ts)) > target - integ.tdir * integ.t
        LibPETSc.TSSetTimeStep(pl, h.ts, target - integ.tdir * integ.t)
    end
    h.stopped = 0
    GC.@preserve ctx begin
        try
            _quiet_errors(h) do
                LibPETSc.TSStep(pl, h.ts)
            end
        catch e
            ctx.err === nothing && !_failed_step(e, h) && rethrow()
            h.stopped = e.code
        end
    end
    if ctx.err !== nothing
        err = ctx.err
        _finish!(integ)
        throw(err)
    end
    h.stopped == 0 || _warn_failed_step(integ.alg, h.stopped)
    integ.t = _user_t(integ.tdir, Float64(LibPETSc.TSGetTime(pl, h.ts)))
    # A step PETSc could not take, or raised part-way through, returns without advancing
    # the clock, which would otherwise spin a `while !done` loop forever.
    if integ.tdir * integ.t <= integ.tdir * integ.tprev
        _finish!(integ)
        return nothing
    end
    if stop === nothing
        integ.dtcache = integ.tdir * Float64(LibPETSc.TSGetTimeStep(pl, h.ts))
        # Steps summed onto the final time can fall a rounding error short of it.
        if integ.tdir * integ.t != h.tf && integ.tdir * integ.t >= h.tf - tol
            integ.t = _user_t(integ.tdir, h.tf)
            LibPETSc.TSSetTime(pl, h.ts, h.tf)
        end
    elseif integ.tdir * integ.t >= stop - tol
        integ.t = _user_t(integ.tdir, stop)
        LibPETSc.TSSetTime(pl, h.ts, stop)
        LibPETSc.TSSetTimeStep(pl, h.ts, integ.tdir * integ.dtcache)
    end
    integ.dt = integ.t - integ.tprev
    _readvec!(integ.u, pl, h.u)
    if ctx.domain !== nothing && SciMLBase.isadaptive(integ) &&
            ctx.domain(integ.u, ctx.p, integ.t)
        return _reject_out_of_domain!(integ, before)
    end
    _end_step_here!(integ)
    fired = _apply_continuous_callbacks!(integ, dtprev)
    integ.finished && return nothing
    fired || _save_step!(integ, integ.t, true)
    # Discrete callbacks run with the stop they landed on still at the head of the queue.
    _apply_callbacks!(integ)
    while !isempty(integ.tstops) && integ.tstops[1] <= integ.tdir * integ.t + _near(integ.t)
        popfirst!(integ.tstops)
    end
    integ.finished && return nothing
    # The same checks the solve path makes after a step, on the step about to be taken.
    if integ.tdir * integ.t < h.tf - tol
        hnext = Float64(LibPETSc.TSGetTimeStep(pl, h.ts))
        limit = isempty(integ.tstops) ? h.tf : min(integ.tstops[1], h.tf)
        if ctx.dtmin > 0 && hnext < ctx.dtmin && integ.tdir * integ.t + hnext < limit - tol
            ctx.dt_too_small = true
        elseif ctx.unstable !== nothing &&
                ctx.unstable(integ.tdir * hnext, integ.u, ctx.p, integ.t)
            ctx.unstable_hit = true
        end
    end
    if !all(isfinite, integ.u) || integ.tdir * integ.t >= h.tf - tol ||
            ctx.unstable_hit ||
            ctx.dt_too_small || Int(LibPETSc.TSGetStepNumber(pl, h.ts)) >= h.maxiters
        _finish!(integ)
    else
        _past_discontinuity!(integ)
        _live_stats!(integ)
    end
    return nothing
end

SciMLBase.step!(integ::PETScIntegrator) = _locked(() -> _step_unlocked(integ))

function _solve_integrator_unlocked(integ::PETScIntegrator)
    while !integ.finished
        SciMLBase.step!(integ)
    end
    return integ.sol
end

SciMLBase.solve!(integ::PETScIntegrator) = _locked(() -> _solve_integrator_unlocked(integ))

SciMLBase.done(integ::PETScIntegrator) = integ.finished

include("autodiff.jl")
include("adjoint.jl")

end
