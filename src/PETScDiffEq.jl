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

include("petsc_compat.jl")

export TSRK, TSRosW, TSImplicit, TSIRK, TSARKIMEX, TSDAE, TSMPRK, TSGeneric,
    PETScIntegrator, PETScAdjoint

abstract type PETScTSAlgorithm <: SciMLBase.AbstractODEAlgorithm end
abstract type PETScTSDAEAlgorithm <: SciMLBase.AbstractDAEAlgorithm end

const AnyPETScTS = Union{PETScTSAlgorithm, PETScTSDAEAlgorithm}

"""
    TSRK(subtype = "5dp", petsc_options = String[]; comm = MPI.COMM_SELF)

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

A `comm` other than `MPI.COMM_SELF` runs the solve distributed over it, with `u0`
holding this rank's rows; see the MPI section of the documentation.
"""
struct TSRK <: PETScTSAlgorithm
    subtype::String
    petsc_options::Vector{String}
    comm::MPI.Comm
end

TSRK(
    subtype::AbstractString = "5dp",
    petsc_options::AbstractVector{<:AbstractString} = String[];
    comm::MPI.Comm = MPI.COMM_SELF,
) = TSRK(String(subtype), String[String(o) for o in petsc_options], comm)

"""
    TSRosW(subtype = "ra34pw2", petsc_options = String[]; autodiff = AutoForwardDiff(), comm = MPI.COMM_SELF)

Rosenbrock-W from PETSc's `TSROSW`. `subtype` is a PETSc `TSRosWType` without
its prefix, such as `"2m"`, `"ra34pw2"` or `"r34prw"`.

Adapts on its embedded error estimate, except for `"theta1"` and `"theta2"`,
which PETSc gives none, so they step at the `dt` you give and warn if you pass
a tolerance. Linearly implicit, so it uses an `ODEFunction`'s `jac`, and it
accepts a mass matrix.

Without a `jac` the Jacobian comes from `autodiff`: ForwardDiff by default, colouring a
sparse `jac_prototype`, or `AutoFiniteDiff()` to have PETSc difference the step's own
equations, colouring a sparse prototype too. For a complex state `f` has to be holomorphic,
and one that is not is refused under ForwardDiff.

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

A `comm` other than `MPI.COMM_SELF` runs the solve distributed over it, as for [`TSRK`](@ref).
There `autodiff` defaults to `AutoFiniteDiff()`, and a `jac` fills this rank's rows of a
sparse `jac_prototype` whose columns are global; see the MPI section of the documentation.
"""
struct TSRosW <: PETScTSAlgorithm
    subtype::String
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
    comm::MPI.Comm
end

TSRosW(
    subtype::AbstractString = "ra34pw2",
    petsc_options::AbstractVector{<:AbstractString} = String[];
    comm::MPI.Comm = MPI.COMM_SELF, autodiff = _default_autodiff(comm),
) = TSRosW(
    String(subtype), String[String(o) for o in petsc_options], _check_autodiff(autodiff), comm,
)

"""
    TSImplicit(subtype = "beuler"; order = nothing, autodiff = AutoForwardDiff(), comm = MPI.COMM_SELF)
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
equations, colouring a sparse prototype too. For a complex state `f` has to be holomorphic,
since PETSc's Newton iteration takes a complex Jacobian, and one that is not is refused
under ForwardDiff.

A `comm` other than `MPI.COMM_SELF` runs the solve distributed over it, as for [`TSRK`](@ref).
There `autodiff` defaults to `AutoFiniteDiff()`, and a `jac` fills this rank's rows of a
sparse `jac_prototype` whose columns are global; see the MPI section of the documentation.
"""
struct TSImplicit <: PETScTSAlgorithm
    subtype::String
    theta::Union{Nothing, Float64}
    order::Union{Nothing, Int}
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
    comm::MPI.Comm
end

function _bdf_order(subtype, order)
    order === nothing && return nothing
    subtype == "bdf" ||
        throw(ArgumentError("`order` applies to TSImplicit(\"bdf\"), not \"$subtype\""))
    1 <= order <= 6 || throw(ArgumentError("PETSc supports BDF orders 1 through 6"))
    return Int(order)
end

TSImplicit(
    subtype::AbstractString = "beuler"; order = nothing, comm::MPI.Comm = MPI.COMM_SELF,
    autodiff = _default_autodiff(comm),
) = TSImplicit(
    String(subtype), nothing, _bdf_order(subtype, order), String[], _check_autodiff(autodiff),
    comm,
)
TSImplicit(
    subtype::AbstractString, theta::Real; order = nothing, comm::MPI.Comm = MPI.COMM_SELF,
    autodiff = _default_autodiff(comm),
) = TSImplicit(
    String(subtype), Float64(theta), _bdf_order(subtype, order), String[],
    _check_autodiff(autodiff), comm,
)
TSImplicit(
    subtype::AbstractString, petsc_options::AbstractVector{<:AbstractString};
    order = nothing, comm::MPI.Comm = MPI.COMM_SELF, autodiff = _default_autodiff(comm),
) = TSImplicit(
    String(subtype), nothing, _bdf_order(subtype, order),
    String[String(o) for o in petsc_options], _check_autodiff(autodiff), comm,
)
TSImplicit(
    subtype::AbstractString, theta::Real,
    petsc_options::AbstractVector{<:AbstractString}; order = nothing,
    comm::MPI.Comm = MPI.COMM_SELF, autodiff = _default_autodiff(comm),
) = TSImplicit(
    String(subtype), Float64(theta), _bdf_order(subtype, order),
    String[String(o) for o in petsc_options], _check_autodiff(autodiff), comm,
)

"""
    TSIRK(nstages = 3, petsc_options = String[]; autodiff = AutoForwardDiff(), comm = MPI.COMM_SELF)

Gauss-Legendre implicit Runge-Kutta from PETSc's `TSIRK`, of order `2 *
nstages`: one stage is the implicit midpoint rule at order 2, two stages give
order 4 and three give order 6. Measured at each of those.

Fixed step: PETSc gives this family no embedded error estimate, so it steps at
the `dt` you give and warns if you pass a tolerance.

Needs a Jacobian, from the `ODEFunction`'s `jac` or from `autodiff`, and refuses
`AutoFiniteDiff()` rather than letting PETSc fail, since it solves all stages as
one coupled system whose matrix it cannot build from finite differences. That coupled matrix is a Kronecker product with the
Jacobian, which has no LU factorisation, so this algorithm defaults to
`-pc_type pbjacobi` unless your own `petsc_options` set a `-pc_type`. So does `"irk"`
picked by [`TSGeneric`](@ref) or by `-ts_type`.

A wrong Jacobian is not caught here. Where the other implicit families fail to
converge, this one reports success and returns a wrong answer, so check a
hand-written `jac` against a solve without one before trusting it.

A mass matrix is rejected. PETSc's coupled-stage matrix assumes `dF/du̇ = I`,
and with a non-identity mass matrix the answer drifts further from the true one
as `dt` shrinks instead of failing, which is worse than an error.

A `comm` other than `MPI.COMM_SELF` runs the solve distributed over it, as for [`TSRK`](@ref).
There it needs a `jac`, filling this rank's rows of a sparse `jac_prototype` whose columns are
global, and each rank has to hold PETSc's own share of the state, which splits it evenly with
the first ranks taking one row more; see the MPI section of the documentation.
"""
struct TSIRK <: PETScTSAlgorithm
    nstages::Int
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
    comm::MPI.Comm
end

TSIRK(
    nstages::Integer = 3, petsc_options::AbstractVector{<:AbstractString} = String[];
    comm::MPI.Comm = MPI.COMM_SELF, autodiff = _default_autodiff(comm),
) = TSIRK(
    Int(nstages), String[String(o) for o in petsc_options], _check_autodiff(autodiff), comm,
)

"""
    TSDAE(subtype = "bdf", petsc_options = String[]; order = nothing, autodiff = AutoForwardDiff(), comm = MPI.COMM_SELF)

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

A `comm` other than `MPI.COMM_SELF` runs the solve distributed over it, as for [`TSRK`](@ref).
There `autodiff` defaults to `AutoFiniteDiff()`, and a `jac` fills this rank's rows of a
sparse `jac_prototype` whose columns are global; see the MPI section of the documentation.
"""
struct TSDAE <: PETScTSDAEAlgorithm
    subtype::String
    order::Union{Nothing, Int}
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
    comm::MPI.Comm
end

TSDAE(
    subtype::AbstractString = "bdf",
    petsc_options::AbstractVector{<:AbstractString} = String[];
    order = nothing, comm::MPI.Comm = MPI.COMM_SELF, autodiff = _default_autodiff(comm),
) = TSDAE(
    String(subtype), _bdf_order(subtype, order), String[String(o) for o in petsc_options],
    _check_autodiff(autodiff), comm,
)

"""
    TSARKIMEX(subtype = "3", petsc_options = String[]; autodiff = AutoForwardDiff(), comm = MPI.COMM_SELF)

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

A `comm` other than `MPI.COMM_SELF` runs the solve distributed over it, as for [`TSRK`](@ref).
There `autodiff` defaults to `AutoFiniteDiff()`, and a `jac` fills this rank's rows of a
sparse `jac_prototype` whose columns are global; see the MPI section of the documentation.
"""
struct TSARKIMEX <: PETScTSAlgorithm
    subtype::String
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
    comm::MPI.Comm
end

TSARKIMEX(
    subtype::AbstractString = "3",
    petsc_options::AbstractVector{<:AbstractString} = String[];
    comm::MPI.Comm = MPI.COMM_SELF, autodiff = _default_autodiff(comm),
) = TSARKIMEX(
    String(subtype), String[String(o) for o in petsc_options], _check_autodiff(autodiff), comm,
)

const _MPRK_TWO_WAY = ("2a22", "2a32", "p2", "p3")
const _MPRK_THREE_WAY = ("2a23", "2a33")

"""
    TSMPRK(slow, subtype = "p2", petsc_options = String[]; comm = MPI.COMM_SELF)
    TSMPRK(slow, medium, subtype = "2a23", petsc_options = String[]; comm = MPI.COMM_SELF)

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
    comm::MPI.Comm

    function TSMPRK(
            slow::Vector{Int}, medium::Vector{Int}, subtype::String,
            petsc_options::Vector{String}, comm::MPI.Comm,
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
        return new(sort(slow), sort(medium), subtype, petsc_options, comm)
    end
end

TSMPRK(
    slow::AbstractVector{<:Integer},
    subtype::AbstractString = "p2",
    petsc_options::AbstractVector{<:AbstractString} = String[];
    comm::MPI.Comm = MPI.COMM_SELF,
) = TSMPRK(
    Vector{Int}(slow), Int[], String(subtype),
    String[String(o) for o in petsc_options], comm,
)

TSMPRK(
    slow::AbstractVector{<:Integer},
    medium::AbstractVector{<:Integer},
    subtype::AbstractString = "2a23",
    petsc_options::AbstractVector{<:AbstractString} = String[];
    comm::MPI.Comm = MPI.COMM_SELF,
) = TSMPRK(
    Vector{Int}(slow), Vector{Int}(medium), String(subtype),
    String[String(o) for o in petsc_options], comm,
)

"""
    TSGeneric(ts_type, petsc_options = String[]; explicit = false, autodiff = AutoForwardDiff(), comm = MPI.COMM_SELF)

Any other PETSc `TSType` by name. An implicit one such as `"alpha"` works with
the default; an explicit one such as `"euler"` or `"ssp"` needs
`explicit = true`, since PETSc then wants the right-hand side rather than the
implicit residual. The constructor refuses a type given the wrong `explicit`.
An explicit type also ignores a `jac` and rejects a mass matrix. An implicit one without
a `jac` gets its Jacobian from `autodiff`, as for [`TSImplicit`](@ref).

Whether the named type adapts is not known here, so no tolerance warning is
issued for it. Only `"euler"` and `"alpha"` have been run through this
package's own convergence tests.

`"alpha2"`, `"basicsymplectic"`, `"discgrad"`, `"eimex"`, `"mimex"` and `"mprk"` are
refused: each is driven through a PETSc setup call this package does not make, and
without it they crash or integrate to zero rather than saying anything.

An explicit type takes a `comm` other than `MPI.COMM_SELF` as [`TSRK`](@ref) does.
"""
struct TSGeneric <: PETScTSAlgorithm
    ts_type::String
    explicit::Bool
    petsc_options::Vector{String}
    autodiff::ADTypes.AbstractADType
    comm::MPI.Comm
end

const _NEEDS_OTHER_SETUP = Dict(
    "alpha2" => "is for second-order systems and needs TSSetI2Function",
    "basicsymplectic" => "needs TSRHSSplitSetIS to declare its position and momentum parts",
    "discgrad" => "needs TSDiscGradSetFormulation",
    "eimex" => "needs its own right-hand-side split, and integrates to zero without one",
    "mimex" => "needs TSRHSSplit to declare its slow and fast parts",
    "mprk" => "needs TSRHSSplit to declare its slow and fast parts",
    "pseudo" => "is pseudo-transient continuation toward a steady state and runs past the final time",
)

const _EXPLICIT_ONLY = ("euler", "glee", "rk", "ssp")

const _ROSW_NO_STEP = ("lassp3p4s2c", "llssp3p4s2c", "ark3")

function TSGeneric(
        ts_type::AbstractString,
        petsc_options::AbstractVector{<:AbstractString} = String[];
        explicit::Bool = false, autodiff = AutoForwardDiff(), comm::MPI.Comm = MPI.COMM_SELF,
    )
    t = String(ts_type)
    haskey(_NEEDS_OTHER_SETUP, t) && throw(
        ArgumentError("PETScDiffEq cannot drive `$t`, which $(_NEEDS_OTHER_SETUP[t])"),
    )
    !explicit && t in _EXPLICIT_ONLY && throw(
        ArgumentError("`$t` is an explicit PETSc type, so it needs `explicit = true`"),
    )
    explicit && t == "irk" && throw(
        ArgumentError("`irk` is an implicit PETSc type, so it cannot take `explicit = true`"),
    )
    return TSGeneric(
        t, explicit, String[String(o) for o in petsc_options], _check_autodiff(autodiff), comm,
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

const _RK_NO_ESTIMATE = ("1fe", "2b", "3", "4")
const _ROSW_NO_ESTIMATE = ("theta1", "theta2")
const _ARKIMEX_NO_ESTIMATE = ("prssp2", "ars443", "bpr3")

_adapts(alg::TSRK) = !(alg.subtype in _RK_NO_ESTIMATE)
_adapts(alg::TSRosW) = !(alg.subtype in _ROSW_NO_ESTIMATE)
_adapts(::TSIRK) = false
_adapts(alg::TSDAE) = alg.subtype == "bdf"
_adapts(alg::TSARKIMEX) = !(alg.subtype in _ARKIMEX_NO_ESTIMATE)
_adapts(alg::TSImplicit) = alg.subtype == "bdf"
_adapts(::TSMPRK) = false
_adapts(::TSGeneric) = nothing

const _RK_CUBIC_INTERP = ("5dp",)
const _ROSW_CUBIC_INTERP = ("ra34pw2", "lassp3p4s2c", "llssp3p4s2c", "ark3")
const _ARKIMEX_CUBIC_INTERP = ("4", "5")
const _ROSW_NO_INTERP = (
    "r34prw", "r3prl2", "rodas3", "rodaspr", "rodaspr2", "grk4t", "shamp4", "veldd4", "4l",
)
const _ARKIMEX_NO_INTERP = ("prssp2", "ars443", "bpr3")

_petsc_interpolant(alg::TSRK) = alg.subtype in _RK_CUBIC_INTERP
_petsc_interpolant(alg::TSRosW) = alg.subtype in _ROSW_CUBIC_INTERP
_petsc_interpolant(alg::TSARKIMEX) = alg.subtype in _ARKIMEX_CUBIC_INTERP
_petsc_interpolant(alg::Union{TSImplicit, TSDAE}) = alg.subtype == "bdf"
_petsc_interpolant(::Union{TSIRK, TSMPRK, TSGeneric}) = false

# TSIRK's TSInterpolate leaves the output untouched instead of failing.
_interpolates(::TSRK) = true
_interpolates(alg::TSRosW) = !(alg.subtype in _ROSW_NO_INTERP)
_interpolates(alg::TSARKIMEX) = !(alg.subtype in _ARKIMEX_NO_INTERP)
_interpolates(::Union{TSImplicit, TSDAE}) = true
_interpolates(::Union{TSIRK, TSMPRK}) = false
_interpolates(::TSGeneric) = nothing

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
# PETSc registers `1bee` at 2, but it is backward Euler.
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

SciMLBase.allowscomplex(::AnyPETScTS) = true

function _implicit_order(subtype, theta, order)
    subtype == "beuler" && return 1
    subtype == "cn" && return 2
    subtype == "theta" && return theta === nothing || theta == 0.5 ? 2 : 1
    subtype == "bdf" && return something(order, 2)
    throw(ArgumentError("no order is known for implicit subtype \"$subtype\""))
end

struct COOJacobian{S}
    src::Vector{Int}
    mass::Vector{S}
    vals::Vector{S}
end

mutable struct TSContext{R, S, U, F, F2, JAC, JBUF, P, L, V}
    petsclib::L
    f!::F
    f2!::F2
    jac!::JAC
    p::P
    du::Vector{S}
    u::Vector{S}
    mudot::Vector{S}
    resid::Vector{S}
    M::Union{Nothing, Matrix{S}, LinearAlgebra.Diagonal{S, Vector{S}}}
    dae::Bool
    missing_diag::Vector{Int}
    W::Matrix{S}
    idx0::Vector{LibPETSc.PetscInt}
    row_cols0::Vector{Vector{LibPETSc.PetscInt}}
    row_src::Vector{Vector{Int}}
    row_buf::Vector{Vector{S}}
    J::JBUF
    ts::Vector{R}
    us::Vector{Vector{U}}
    dus::Vector{Vector{U}}
    user_ts::Union{Nothing, Vector{R}}
    user_dus::Union{Nothing, Vector{Vector{U}}}
    saveat::Vector{R}
    saveat_idx::Int
    save_everystep::Bool
    save_start::Bool
    dense::Bool
    save_idxs::Union{Nothing, Vector{Int}}
    work::V
    hermite::Bool
    interpolates::Union{Nothing, Bool}
    alg_name::String
    step_t::R
    step_u::Vector{S}
    end_s::R
    end_u::Vector{S}
    fstart::Union{Nothing, Vector{S}}
    fend::Union{Nothing, Vector{S}}
    pdirty::Bool
    slow_idxs::Vector{Int}
    medium_idxs::Vector{Int}
    fast_idxs::Vector{Int}
    part_t::R
    part_u::Vector{S}
    part_valid::Bool
    dtmin::R
    dt_too_small::Bool
    unstable::Any
    unstable_hit::Bool
    tdir::R
    domain::Any
    nf::Int
    nf2::Int
    njacs::Int
    err::Union{Nothing, Any}
    comm::Union{Nothing, MPI.Comm}
    retry_fp::Bool
    workvec::Ptr{Cvoid}
    nreject::Int
    halt_nonfinite::Bool
    coo::Union{Nothing, COOJacobian{S}}
end

_distributed(alg::AnyPETScTS) = alg.comm != MPI.COMM_SELF

_everywhere(::Nothing, b::Bool) = b
_everywhere(comm::MPI.Comm, b::Bool) = MPI.Allreduce(b, &, comm)
_anywhere(::Nothing, b::Bool) = b
_anywhere(comm::MPI.Comm, b::Bool) = MPI.Allreduce(b, |, comm)

_remote_error() = ErrorException("the solve raised an error on another rank of its communicator")

function _throw_anywhere(comm, err)
    _anywhere(comm, err !== nothing) && throw(something(err, _remote_error()))
    return nothing
end

function _threw!(ctx::TSContext)
    _anywhere(ctx.comm, ctx.err !== nothing) || return false
    ctx.err === nothing && (ctx.err = _remote_error())
    return true
end

function _throw_if_threw!(ctx::TSContext)
    _threw!(ctx) && throw(ctx.err)
    return nothing
end

# A rank whose `f` throws keeps making the collective calls, returning NaN until the ranks agree.
function _call!(f, ctx::TSContext, out, args...)
    ctx.comm === nothing && return f(out, args...)
    try
        f(out, args...)
    catch e
        ctx.err === nothing && (ctx.err = e)
        fill!(out, NaN)
    end
    return nothing
end

_call_f!(ctx::TSContext, du, u, t) = _call!(ctx.f!, ctx, du, u, ctx.p, t)

_guard_f(f, ::Nothing, _) = f
_guard_f(f, ::MPI.Comm, box) = function (du, u, p, t)
    try
        f(du, u, p, t)
    catch e
        box[] === nothing && (box[] = e)
        fill!(du, NaN)
    end
    return nothing
end

function _asked(f, ctx::TSContext)
    try
        return f()::Bool
    catch e
        ctx.err === nothing && (ctx.err = e)
        return false
    end
end

_predicate(f, ctx::TSContext) = ctx.comm === nothing ? f() : _anywhere(ctx.comm, _asked(f, ctx))

function _checked_everywhere(f, comm)
    comm === nothing && return f()
    value = err = nothing
    try
        value = f()
    catch e
        err = e
    end
    _throw_anywhere(comm, err)
    return value
end

function _checked_anywhere(f, comm)
    comm === nothing && return f()
    value, err = false, nothing
    try
        value = f()::Bool
    catch e
        err = e
    end
    threw, value = MPI.Allreduce([err !== nothing, value], |, comm)
    threw && throw(something(err, _remote_error()))
    return value
end

_smallest(::Nothing, x) = x
_smallest(comm::MPI.Comm, x) = MPI.Allreduce(x, min, comm)

function _record!(ctx::TSContext{R, S}, t, x, du = nothing) where {R, S}
    full = Vector{S}(x)
    idxs = ctx.save_idxs
    push!(ctx.ts, R(t))
    ctx.user_ts === nothing || push!(ctx.user_ts, _user_t(ctx.tdir, R(t)))
    if ctx.dense
        du === nothing && (du = _derivative(ctx, R(t), full))
        push!(ctx.dus, _select(du, idxs))
        ctx.user_dus === nothing || push!(ctx.user_dus, -ctx.dus[end])
    end
    push!(ctx.us, _select(full, idxs))
    return du
end

function _record_end!(ctx, t, x)
    du = _record!(ctx, t, x, ctx.fend)
    ctx.hermite && (ctx.fend = du)
    return nothing
end

# PETSc's `dt_min` clamps and takes the step whatever its error, so check the floor here.
function _post_step!(ts_ptr::LibPETSc.CTS)::LibPETSc.PetscErrorCode
    ctx = POST_STEP_CTX[ts_ptr]::TSContext
    ctx.comm === nothing || return _post_step_collective!(ctx, ts_ptr)
    ctx.err === nothing || return LibPETSc.PetscErrorCode(0)
    try
        pl = ctx.petsclib
        ts = LibPETSc.TS(ts_ptr, pl)
        Int(LibPETSc.TSGetConvergedReason(pl, ts)) < 0 && return LibPETSc.PetscErrorCode(0)
        s = LibPETSc.TSGetTime(pl, ts)
        smax = LibPETSc.TSGetMaxTime(pl, ts)
        s >= smax - _near(smax) && return LibPETSc.PetscErrorCode(0)
        hnext = LibPETSc.TSGetTimeStep(pl, ts)
        stop = false
        if ctx.dtmin > 0 && hnext < ctx.dtmin && s + hnext < smax - _near(smax)
            ctx.dt_too_small = stop = true
        end
        if !stop && (ctx.unstable !== nothing || ctx.halt_nonfinite)
            x = Ref{LibPETSc.CVec}(C_NULL)
            ccall(
                _symbol(pl, :TSGetSolution), LibPETSc.PetscErrorCode,
                (LibPETSc.CTS, Ptr{LibPETSc.CVec}), ts_ptr, x,
            )
            u = _readvec!(ctx.u, pl, PETSc.VecPtr(pl, x[], false))
            ctx.unstable !== nothing &&
                ctx.unstable(ctx.tdir * hnext, u, ctx.p, _user_t(ctx.tdir, s)) &&
                (ctx.unstable_hit = stop = true)
            ctx.halt_nonfinite && !all(isfinite, u) && (stop = true)
        end
        stop && LibPETSc.TSSetConvergedReason(pl, ts, LibPETSc.TS_CONVERGED_USER)
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

# Reduces on every rank after every step, so no early return may skip it.
function _post_step_collective!(ctx, ts_ptr)
    pl = ctx.petsclib
    try
        ts = LibPETSc.TS(ts_ptr, pl)
        small = unstable = false
        s = LibPETSc.TSGetTime(pl, ts)
        smax = LibPETSc.TSGetMaxTime(pl, ts)
        if Int(LibPETSc.TSGetConvergedReason(pl, ts)) >= 0 && s < smax - _near(smax)
            hnext = LibPETSc.TSGetTimeStep(pl, ts)
            small = ctx.dtmin > 0 && hnext < ctx.dtmin && s + hnext < smax - _near(smax)
            if !small && ctx.unstable !== nothing
                x = Ref{LibPETSc.CVec}(C_NULL)
                ccall(
                    _symbol(pl, :TSGetSolution), LibPETSc.PetscErrorCode,
                    (LibPETSc.CTS, Ptr{LibPETSc.CVec}), ts_ptr, x,
                )
                u = _readvec!(ctx.u, pl, PETSc.VecPtr(pl, x[], false))
                unstable = _asked(ctx) do
                    ctx.unstable(ctx.tdir * hnext, u, ctx.p, _user_t(ctx.tdir, s))
                end
            end
        end
        threw, unstable = MPI.Allreduce([ctx.err !== nothing, unstable], |, ctx.comm)
        threw && ctx.err === nothing && (ctx.err = _remote_error())
        ctx.dt_too_small |= small
        ctx.unstable_hit |= unstable
        (threw || small || unstable) &&
            LibPETSc.TSSetConvergedReason(pl, ts, LibPETSc.TS_CONVERGED_USER)
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

const POST_STEP_PTR = Ref{Ptr{Cvoid}}(C_NULL)

# The post-step callback gets no context, and asking the TS may use another build's symbol.
const POST_STEP_CTX = Dict{LibPETSc.CTS, Any}()

function _set_post_step!(pl, ts, ctx)
    POST_STEP_CTX[ts.ptr] = ctx
    ccall(
        _symbol(pl, :TSSetPostStep), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Ptr{Cvoid}), ts, POST_STEP_PTR[],
    )
    return nothing
end

const PETSC_SYMBOLS = Dict{Tuple{String, Symbol}, Ptr{Cvoid}}()

_symbol(petsclib, name::Symbol) = get!(PETSC_SYMBOLS, (petsclib.petsc_library, name)) do
    Libdl.dlsym(Libdl.dlopen(petsclib.petsc_library), name)
end

# OrdinaryDiffEq applies dtmin only when adaptive. `force_dtmin` hands it to PETSc.
_floor(R, dtmin, force_dtmin, adaptive) =
    force_dtmin || !adaptive || dtmin === nothing ? zero(R) : abs(R(dtmin))

_near(t::Float64) = 100 * eps(max(1.0, abs(t)))
_near(t::Float32) = 4 * eps(abs(t))

# PETSc divides by zero on a step of a few ulps. Float64 already has a one-ulp floor.
_min_step(t::Float64) = zero(t)
_min_step(t::Float32) = 4 * eps(abs(t))

# Exact equality: accepted steps near a singularity can be closer than any tolerance.
_last_recorded(ctx, t) = !isempty(ctx.ts) && ctx.ts[end] == t

_select(u, ::Nothing) = u
_select(u, idxs::Vector{Int}) = u[idxs]

function _derivative(ctx::TSContext{R, S}, t, u) where {R, S}
    u = convert(Vector{S}, u)
    du = similar(u)
    _call_f!(ctx, du, u, t)
    if ctx.f2! !== nothing
        _call!(ctx.f2!, ctx, ctx.du, u, ctx.p, t)
        du .+= ctx.du
    end
    ctx.nf += 1
    ctx.f2! === nothing || (ctx.nf2 += 1)
    return du
end

_saved(ctx::TSContext{R, S, U}, u) where {R, S, U} =
    ctx.save_idxs === nothing ? Vector{U}(u) : U[u[i] for i in ctx.save_idxs]

_interp(ctx, ts, dus) = ctx.dense ? SciMLBase.HermiteInterpolation(ts, ctx.us, dus) :
    SciMLBase.LinearInterpolation(ts, ctx.us)

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

const PETSC_ERR_SUP = 56
const PETSC_ERR_MAT_LU_ZRPVT = 71
const PETSC_ERR_FP = 72
const CALLBACK_THREW = 1

# Dense LU raises on a zero pivot whatever -ts_error_if_step_fails says.
_failed_step(e, h) =
    e isa LibPETSc.PetscError && !h.pivot_raises &&
    (e.code == PETSC_ERR_MAT_LU_ZRPVT || e.code == PETSC_ERR_FP)

_verbose(kwargs) = get(kwargs, :verbose, true) !== false

function _warn_failed_step(alg, code, kwargs)
    _verbose(kwargs) || return nothing
    why = code == PETSC_ERR_FP ?
        "PETSc hit a floating point exception, an overflow or a NaN in the step or in " *
        "its error estimate" :
        "the LU factorization of its Newton matrix hit a zero pivot"
    @warn "`$(_warn_name(alg))` ends here because $why"
    return nothing
end

function _ts_dm(pl, ts)
    dm = Ref{Ptr{Cvoid}}(C_NULL)
    code = ccall(
        _symbol(pl, :TSGetDM), LibPETSc.PetscErrorCode, (LibPETSc.CTS, Ptr{Ptr{Cvoid}}), ts, dm,
    )
    code == 0 || throw(LibPETSc.PetscError(code))
    return dm[]
end

# TSAdapt raises PETSC_ERR_FP with its DM work vector still out, which pins the DM and fails
# its next DMClearGlobalVectors. Holding the vector the DM hands out first lets it go back.
function _hold_work_vec!(ctx, ts)
    pl = ctx.petsclib
    dm, v = _ts_dm(pl, ts), Ref{Ptr{Cvoid}}(C_NULL)
    code = ccall(
        _symbol(pl, :DMGetGlobalVector), LibPETSc.PetscErrorCode,
        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), dm, v,
    )
    code == 0 || throw(LibPETSc.PetscError(code))
    ccall(_symbol(pl, :PetscObjectReference), LibPETSc.PetscErrorCode, (Ptr{Cvoid},), v[])
    ctx.workvec = v[]
    ccall(
        _symbol(pl, :DMRestoreGlobalVector), LibPETSc.PetscErrorCode,
        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), dm, v,
    )
    return nothing
end

function _return_work_vec!(ctx, ts)
    ctx.workvec == C_NULL && return false
    pl = ctx.petsclib
    dm, owner = _ts_dm(pl, ts), Ref{Ptr{Cvoid}}(C_NULL)
    ccall(
        _symbol(pl, :VecGetDM), LibPETSc.PetscErrorCode, (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
        ctx.workvec, owner,
    )
    owner[] == dm || return false
    return ccall(
        _symbol(pl, :DMRestoreGlobalVector), LibPETSc.PetscErrorCode,
        (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), dm, Ref(ctx.workvec),
    ) == 0
end

function _release_work_vec!(ctx)
    ctx.workvec == C_NULL && return nothing
    ccall(
        _symbol(ctx.petsclib, :VecDestroy), LibPETSc.PetscErrorCode, (Ptr{Ptr{Cvoid}},),
        Ref(ctx.workvec),
    )
    ctx.workvec = C_NULL
    return nothing
end

function _retry_step(h, s, taken, floor, forced, failed)
    ctx = h.ctx
    ctx.nreject += 1
    smaller = taken / 5
    if forced && smaller < floor
        failed && taken <= floor || return min(floor, taken), true
        ctx.unstable_hit = true
        return nothing, false
    end
    below_floor = ctx.dtmin > 0 && smaller < ctx.dtmin
    if below_floor || smaller < 100 * eps(max(abs(s), abs(h.t0), abs(h.tf)))
        below_floor ? (ctx.dt_too_small = true) : (ctx.unstable_hit = true)
        return nothing, false
    end
    return smaller, false
end

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

function _set_pc_type!(pl, ts, type)
    lib = Libdl.dlopen(pl.petsc_library)
    snes, ksp = Ref{LibPETSc.CSNES}(C_NULL), Ref{LibPETSc.CKSP}(C_NULL)
    pc = Ref{Ptr{Cvoid}}(C_NULL)
    ccall(
        Libdl.dlsym(lib, :TSGetSNES), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Ptr{LibPETSc.CSNES}), ts, snes,
    )
    ccall(
        Libdl.dlsym(lib, :SNESGetKSP), LibPETSc.PetscErrorCode,
        (LibPETSc.CSNES, Ptr{LibPETSc.CKSP}), snes[], ksp,
    )
    ccall(
        Libdl.dlsym(lib, :KSPGetPC), LibPETSc.PetscErrorCode,
        (LibPETSc.CKSP, Ptr{Ptr{Cvoid}}), ksp[], pc,
    )
    code = ccall(
        Libdl.dlsym(lib, :PCSetType), LibPETSc.PetscErrorCode, (Ptr{Cvoid}, Cstring), pc[], type,
    )
    code == 0 || throw(LibPETSc.PetscError(code))
    return nothing
end

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

_fd_pattern(jac_prototype, M, n) = _jacobian_pattern(SparseMatrixCSC(jac_prototype), n, M)

# PETSc 3.22 has no getter for this, so the options are parsed as PETSc would.
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

const QUIET_FAILED_STEPS = Ref(true)

# Printing nothing also stops PETSc treating the next traceback as part of this one.
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

function _quiet_errors(f, h)
    QUIET_FAILED_STEPS[] = !h.pivot_raises
    pl = h.petsclib
    ccall(
        _symbol(pl, :PetscPushErrorHandler), LibPETSc.PetscErrorCode,
        (Ptr{Cvoid}, Ptr{Cvoid}), ZERO_PIVOT_HANDLER_PTR[],
        _symbol(pl, :PetscTraceBackErrorHandler),
    )
    try
        return f()
    finally
        ccall(_symbol(pl, :PetscPopErrorHandler), LibPETSc.PetscErrorCode, ())
    end
end

function _petsc_interpolate!(ctx, ts, s)
    pl = ctx.petsclib
    ctx.interpolates === false && return nothing
    ctx.interpolates === true && (LibPETSc.TSInterpolate(pl, ts, s, ctx.work); return ctx.work)
    # PETSc prints a traceback before refusing, so ask with printing off. Some types
    # register an interpolant that writes nothing, so prefill NaN to catch that.
    PETScCompat.with_local_array!(w -> fill!(w, NaN), ctx.work; read = false, write = true)
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
    written = _anywhere(
        ctx.comm,
        PETScCompat.with_local_array!(w -> any(!isnan, w), ctx.work; read = true, write = false),
    )
    ctx.interpolates = written
    return written ? ctx.work : nothing
end

_mass(ctx::TSContext{R, S}, i, j) where {R, S} =
    ctx.M === nothing ? (i == j ? one(S) : zero(S)) : ctx.M[i, j]

# A DAE jac is already PETSc's whole `shift * dG/du_dot + dG/du`.
# An ODE's J is turned into `shift * M - J` by the callers.
function _call_jac!(ctx, xdot_ptr, shift, t)
    if ctx.dae
        _readvec!(ctx.mudot, ctx.petsclib, PETSc.VecPtr(ctx.petsclib, xdot_ptr, false))
        ctx.jac!(ctx.J, ctx.mudot, ctx.u, ctx.p, shift, t)
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
            jv = src[k] == 0 ? zero(eltype(ctx.J)) : ctx.J.nzval[src[k]]
            buf[k] = ctx.dae ? jv : shift * _mass(ctx, i, j) - jv
        end
    end
    return nothing
end

# Relies on SeqAIJ storing rows by ascending column, the order `_row_structure` builds.
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

# The shift lands on the diagonal and M's nonzeros, so those always get a slot (src 0).
function _row_structure(J::SparseMatrixCSC, n, M = nothing, rstart = 0)
    cols = [Int[] for _ in 1:n]
    src = [Int[] for _ in 1:n]
    for j in axes(J, 2), k in J.colptr[j]:(J.colptr[j + 1] - 1)
        i = J.rowval[k]
        push!(cols[i], j)
        push!(src[i], k)
    end
    shifted = [CartesianIndex(i, rstart + i) for i in 1:n]
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
    buf = [zeros(eltype(J), length(cols[i])) for i in 1:n]
    return cols0, src, buf
end

function _coo_structure(J::SparseMatrixCSC{S}, rstart, M) where {S}
    n = size(J, 1)
    cols0, src, _ = _row_structure(J, n, nothing, rstart)
    rows = LibPETSc.PetscInt[rstart + i - 1 for i in 1:n for _ in cols0[i]]
    cols = LibPETSc.PetscInt[c for i in 1:n for c in cols0[i]]
    mass = S[
        c == rstart + i - 1 ? (M === nothing ? one(S) : M.diag[i]) : zero(S)
            for i in 1:n for c in cols0[i]
    ]
    coo = COOJacobian(Int[k for i in 1:n for k in src[i]], mass, zeros(S, length(rows)))
    return rows, cols, coo
end

function _coo_matrix!(A, petsclib, n, N, rows, cols)
    LibPETSc.MatSetSizes(
        petsclib, A, LibPETSc.PetscInt(n), LibPETSc.PetscInt(n), LibPETSc.PetscInt(N),
        LibPETSc.PetscInt(N),
    )
    LibPETSc.MatSetType(petsclib, A, "aij")
    LibPETSc.MatSetPreallocationCOO(
        petsclib, A, LibPETSc.PetscCount(length(rows)), rows, cols,
    )
    return A
end

function _coo_values!(ctx, xdot_ptr, shift, t)
    coo = ctx.coo
    try
        _call_jac!(ctx, xdot_ptr, shift, t)
        J = ctx.J.nzval
        @inbounds for k in eachindex(coo.vals)
            jv = coo.src[k] == 0 ? zero(eltype(J)) : J[coo.src[k]]
            coo.vals[k] = ctx.dae ? jv : shift * coo.mass[k] - jv
        end
    catch e
        ctx.err === nothing && (ctx.err = e)
        fill!(coo.vals, NaN)
    end
    return coo.vals
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

# PETSc only steps forward, so a reversed span runs in s = -t: dv/ds = -f(v, p, -s)
# and G(t, u, u') becomes G(-s, v, -dv/ds). `dv` is negated in place and restored.
_reverse_rhs(f) = (du, u, p, s) -> (f(du, u, p, _user_t(-one(s), s)); du .*= -1; nothing)
_reverse_jac(j) =
    (J, u, p, s) -> (j(J, u, p, _user_t(-one(s), s)); LinearAlgebra.rmul!(J, -1); nothing)
_reverse_residual(g) =
    (r, dv, u, p, s) -> (dv .*= -1; g(r, dv, u, p, _user_t(-one(s), s)); dv .*= -1; nothing)
_reverse_dae_jac(j) =
    (J, dv, u, p, gamma, s) -> (dv .*= -1; j(J, dv, u, p, -gamma, _user_t(-one(s), s)); dv .*= -1; nothing)

# `+ zero(s)` turns -0.0 into 0.0, which `isless` would otherwise put before 0.0.
_user_t(tdir, s) = tdir * s + zero(s)

_copy_jac!(J::AbstractMatrix, A) = (copyto!(J, A); nothing)

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

# PETSc.jl's array accessor allocates on every call.
function _readvec!(dest, pl, v)
    a = LibPETSc.VecGetArrayRead(pl, v)
    try
        copyto!(dest, a)
    finally
        LibPETSc.VecRestoreArrayRead(pl, v, a)
    end
    return dest
end

function _writevec!(pl, v, src)
    a = LibPETSc.VecGetArrayWrite(pl, v)
    try
        copyto!(a, src)
    finally
        LibPETSc.VecRestoreArrayWrite(pl, v, a)
    end
    return nothing
end

function _rhs!(
        ::LibPETSc.CTS,
        t,
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
        _call_f!(ctx, ctx.du, ctx.u, t)
        _writevec!(pl, PETSc.VecPtr(pl, f_ptr, false), ctx.du)
        ctx.nf += 1
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

function _split_rhs!(
        ::LibPETSc.CTS,
        t,
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
        _call!(ctx.f2!, ctx, ctx.du, ctx.u, ctx.p, t)
        _writevec!(pl, PETSc.VecPtr(pl, f_ptr, false), ctx.du)
        ctx.nf2 += 1
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

function _ifunction!(
        ::LibPETSc.CTS,
        t,
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
            _call!(ctx.f!, ctx, ctx.resid, udot, ctx.u, ctx.p, t)
        else
            _call_f!(ctx, ctx.du, ctx.u, t)
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

function _mprk_part!(ctx, t, x_ptr, f_ptr, idxs)
    pl = ctx.petsclib
    try
        _readvec!(ctx.u, pl, PETSc.VecPtr(pl, x_ptr, false))
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
        t,
        x_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _mprk_part!(ctx, t, x_ptr, f_ptr, ctx.slow_idxs)
end

function _mprk_medium!(
        ::LibPETSc.CTS,
        t,
        x_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _mprk_part!(ctx, t, x_ptr, f_ptr, ctx.medium_idxs)
end

function _mprk_fast!(
        ::LibPETSc.CTS,
        t,
        x_ptr::LibPETSc.CVec,
        f_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _mprk_part!(ctx, t, x_ptr, f_ptr, ctx.fast_idxs)
end

function _ijacobian!(
        ::LibPETSc.CTS,
        t,
        x_ptr::LibPETSc.CVec,
        xdot_ptr::LibPETSc.CVec,
        shift,
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
        # MatSetValues reads the block row-major, hence W[j, i].
        @inbounds for j in 1:n, i in 1:n
            ctx.W[j, i] = ctx.dae ? ctx.J[i, j] : shift * _mass(ctx, i, j) - ctx.J[i, j]
        end
        _setblock!(ctx, B, n)
        PETSc.assemble!(B)
        # Under -snes_mf_operator A is matrix-free and assembled only to pick up the state.
        B.ptr == A.ptr || PETSc.assemble!(A)
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

function _sparse_ijacobian!(
        ::LibPETSc.CTS,
        t,
        x_ptr::LibPETSc.CVec,
        xdot_ptr::LibPETSc.CVec,
        shift,
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
        if ctx.coo === nothing
            _call_jac!(ctx, xdot_ptr, shift, t)
            ctx.njacs += 1
            n = length(ctx.u)
            _fill_rows!(ctx, shift, n)
            _setrows!(ctx, B, n)
        else
            vals = _coo_values!(ctx, xdot_ptr, shift, t)
            ctx.njacs += 1
            LibPETSc.MatSetValuesCOO(ctx.petsclib, B, vals, LibPETSc.INSERT_VALUES)
        end
        PETSc.assemble!(B)
        B.ptr == A.ptr || PETSc.assemble!(A)
    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

function _monitor!(
        ts_ptr::LibPETSc.CTS,
        step::LibPETSc.PetscInt,
        t,
        x_ptr::LibPETSc.CVec,
        ctx_ptr::Ptr{Cvoid},
    )::LibPETSc.PetscErrorCode
    ctx = unsafe_pointer_to_objref(ctx_ptr)::TSContext
    return _monitor_body!(ctx, ts_ptr, step, t, x_ptr)
end

function _monitor_body!(ctx, ts_ptr, step, t, x_ptr)
    ctx.err === nothing || return LibPETSc.PetscErrorCode(0)
    x = PETSc.VecPtr(ctx.petsclib, x_ptr, false)
    try
        ctx.retry_fp && ctx.workvec == C_NULL && _hold_work_vec!(ctx, ts_ptr)
        ts = LibPETSc.TS(ts_ptr, ctx.petsclib)
        tol = _near(t)
        landed = false
        while ctx.saveat_idx <= length(ctx.saveat) && ctx.saveat[ctx.saveat_idx] <= t + tol
            want = ctx.saveat[ctx.saveat_idx]
            if step == 0 || abs(want - t) <= tol
                _record_end!(ctx, want, _readvec!(ctx.u, ctx.petsclib, x))
                landed = true
            elseif ctx.hermite
                # -ts_exact_final_time interpolate steps past tf and reports tf later.
                tmax = LibPETSc.TSGetMaxTime(ctx.petsclib, ts)
                want >= tmax - tol && t > tmax + tol && break
                u1 = _readvec!(ctx.u, ctx.petsclib, x)
                _record!(
                    ctx, want, _hermite!(similar(u1), ctx, want, ctx.step_t, ctx.step_u, t, u1),
                )
            elseif _petsc_interpolate!(ctx, ts, want) === nothing
                ctx.err = _no_interpolant(ctx)
                # A failing monitor makes PETSc print a traceback, so stop a step later.
                LibPETSc.TSSetMaxSteps(ctx.petsclib, ts, step + 1)
                return LibPETSc.PetscErrorCode(0)
            else
                _record!(ctx, want, _readvec!(ctx.u, ctx.petsclib, ctx.work))
            end
            ctx.saveat_idx += 1
        end
        # A failed step calls the monitor again at the last reported time.
        if (step == 0 ? ctx.save_start : ctx.save_everystep) && !landed &&
                !_last_recorded(ctx, t)
            _record!(ctx, t, _readvec!(ctx.u, ctx.petsclib, x))
        end
        if ctx.hermite && ctx.saveat_idx <= length(ctx.saveat)
            ctx.step_t = t
            _readvec!(ctx.step_u, ctx.petsclib, x)
            ctx.fstart = ctx.pdirty ? nothing : ctx.fend
            ctx.fend = nothing
            ctx.pdirty = false
        end
        # Steps below the spacing of t still change the state, so keep the first one at t.
        if step == 0 || t != ctx.end_s
            ctx.end_s = t
            _readvec!(ctx.end_u, ctx.petsclib, x)
        end

    catch e
        ctx.err = e
        return LibPETSc.PetscErrorCode(CALLBACK_THREW)
    end
    return LibPETSc.PetscErrorCode(0)
end

struct Callbacks
    rhs::Ptr{Cvoid}
    split_rhs::Ptr{Cvoid}
    monitor::Ptr{Cvoid}
    ifunction::Ptr{Cvoid}
    ijacobian::Ptr{Cvoid}
    sparse_ijacobian::Ptr{Cvoid}
    mprk_slow::Ptr{Cvoid}
    mprk_medium::Ptr{Cvoid}
    mprk_fast::Ptr{Cvoid}
end

const CALLBACKS = Dict{DataType, Callbacks}()

function _callbacks(petsclib)
    R = petsclib.PetscReal
    return get!(() -> _make_callbacks(R), CALLBACKS, R)
end

for R in (Float32, Float64)
    @eval _make_callbacks(::Type{$R}) = Callbacks(
        @cfunction(
            _rhs!,
            LibPETSc.PetscErrorCode,
            (LibPETSc.CTS, $R, LibPETSc.CVec, LibPETSc.CVec, Ptr{Cvoid})
        ),
        @cfunction(
            _split_rhs!,
            LibPETSc.PetscErrorCode,
            (LibPETSc.CTS, $R, LibPETSc.CVec, LibPETSc.CVec, Ptr{Cvoid})
        ),
        @cfunction(
            _monitor!,
            LibPETSc.PetscErrorCode,
            (LibPETSc.CTS, LibPETSc.PetscInt, $R, LibPETSc.CVec, Ptr{Cvoid})
        ),
        @cfunction(
            _ifunction!,
            LibPETSc.PetscErrorCode,
            (LibPETSc.CTS, $R, LibPETSc.CVec, LibPETSc.CVec, LibPETSc.CVec, Ptr{Cvoid})
        ),
        @cfunction(
            _ijacobian!,
            LibPETSc.PetscErrorCode,
            (
                LibPETSc.CTS, $R, LibPETSc.CVec, LibPETSc.CVec, $R, LibPETSc.CMat,
                LibPETSc.CMat, Ptr{Cvoid},
            )
        ),
        @cfunction(
            _sparse_ijacobian!,
            LibPETSc.PetscErrorCode,
            (
                LibPETSc.CTS, $R, LibPETSc.CVec, LibPETSc.CVec, $R, LibPETSc.CMat,
                LibPETSc.CMat, Ptr{Cvoid},
            )
        ),
        @cfunction(
            _mprk_slow!,
            LibPETSc.PetscErrorCode,
            (LibPETSc.CTS, $R, LibPETSc.CVec, LibPETSc.CVec, Ptr{Cvoid})
        ),
        @cfunction(
            _mprk_medium!,
            LibPETSc.PetscErrorCode,
            (LibPETSc.CTS, $R, LibPETSc.CVec, LibPETSc.CVec, Ptr{Cvoid})
        ),
        @cfunction(
            _mprk_fast!,
            LibPETSc.PetscErrorCode,
            (LibPETSc.CTS, $R, LibPETSc.CVec, LibPETSc.CVec, Ptr{Cvoid})
        ),
    )
end

# `@cfunction` pointers do not survive precompilation.
function __init__()
    POST_STEP_PTR[] = @cfunction(_post_step!, LibPETSc.PetscErrorCode, (LibPETSc.CTS,))
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

# PETSc.jl's `TSRHSSplitSetRHSFunction` wrapper takes no context, so ccall it.
function _set_split!(petsclib, ts, name, idxs, fptr, ctxptr)
    n = LibPETSc.PetscInt(length(idxs))
    is = LibPETSc.ISCreateGeneral(
        petsclib, MPI.COMM_SELF, n,
        LibPETSc.PetscInt[i - 1 for i in idxs], LibPETSc.PETSC_COPY_VALUES,
    )
    LibPETSc.TSRHSSplitSetIS(petsclib, ts, name, is)
    code = ccall(
        _symbol(petsclib, :TSRHSSplitSetRHSFunction), LibPETSc.PetscErrorCode,
        (LibPETSc.CTS, Cstring, LibPETSc.CVec, Ptr{Cvoid}, Ptr{Cvoid}),
        ts, name, C_NULL, fptr, ctxptr,
    )
    iszero(code) ||
        throw(ErrorException("TSRHSSplitSetRHSFunction(\"$name\") failed with $code"))
    return nothing
end

# PETSc.jl fixes `LibPETSc.PetscInt` at Int64 whatever the loaded library uses.
function _check_inttype(petsclib)
    PETSc.inttype(petsclib) === LibPETSc.PetscInt || error(
        "PETScDiffEq needs a PETSc built with $(LibPETSc.PetscInt) indices, but the " *
            "library it loaded uses $(PETSc.inttype(petsclib))",
    )
    return nothing
end

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

function _running_name(petsclib, ts)
    type = LibPETSc.TSGetType(petsclib, ts)
    type == "rk" && return "rk $(LibPETSc.TSRKGetType(petsclib, ts))"
    type == "rosw" && return "rosw $(LibPETSc.TSRosWGetType(petsclib, ts))"
    type == "arkimex" && return "arkimex $(LibPETSc.TSARKIMEXGetType(petsclib, ts))"
    return type
end

function _refuse_method(name, has_mass, has_jac, is_split, is_dae)
    if name == "irk" && has_mass
        throw(
            ArgumentError(
                "PETScDiffEq does not support a mass matrix with TSIRK; PETSc's " *
                    "coupled-stage matrix assumes dF/du_dot = I, and the answer drifts " *
                    "further from the true one as dt shrinks rather than failing",
            ),
        )
    end
    if name == "irk" && is_dae
        throw(
            ArgumentError(
                "PETScDiffEq does not support a DAEProblem with TSIRK, whose " *
                    "stage matrix in PETSc assumes dG/du' = I",
            ),
        )
    end
    if name == "irk" && !has_jac
        throw(
            ArgumentError(
                "TSIRK needs a Jacobian; give the ODEFunction a `jac` or leave `autodiff` " *
                    "at a backend other than `AutoFiniteDiff()`, since PETSc builds its " *
                    "coupled-stage matrix from one and has no finite-difference fallback for it",
            ),
        )
    end
    sub = last(split(name))
    if startswith(name, "rosw ") && sub in _ROSW_NO_STEP
        throw(
            ArgumentError(
                "TSRosW(\"$sub\") cannot be used: without a `jac` PETSc stops " *
                    "and asks for one, and with one it does not restore its Jacobian lag " *
                    "after the explicit last stage, so an adaptive solve fails within its " *
                    "first two steps and a fixed-step solve diverges; use another TSRosW type",
            ),
        )
    end
    if name == "rosw assp3p3s1c" && has_mass
        throw(
            ArgumentError(
                "TSRosW(\"assp3p3s1c\") cannot take a mass matrix; PETSc leaves the mass " *
                    "matrix out of its explicit first stage, so the solve reports success " *
                    "with an error that does not shrink with dt",
            ),
        )
    end
    if name == "rosw assp3p3s1c" && !has_jac
        throw(
            ArgumentError(
                "TSRosW(\"assp3p3s1c\") needs a Jacobian; give the ODEFunction a `jac` " *
                    "or leave `autodiff` at a backend other than `AutoFiniteDiff()`, since " *
                    "PETSc asks for one at the start of every step and has no " *
                    "finite-difference fallback there",
            ),
        )
    end
    if name == "arkimex ars122" && !is_split
        throw(
            ArgumentError(
                "TSARKIMEX(\"ars122\") needs a SplitODEProblem; it has an explicit first " *
                    "stage and is not stiffly accurate, so PETSc cannot evaluate its " *
                    "first-stage slope when the whole problem is implicit",
            ),
        )
    end
    if name == "arkimex bpr3" && is_split
        throw(
            ArgumentError(
                "TSARKIMEX(\"bpr3\") converges at first order on a SplitODEProblem; solve " *
                    "a plain ODEProblem with it, or use another TSARKIMEX type",
            ),
        )
    end
    return nothing
end

_set_subtype!(petsclib, ts, alg::TSRK) =
    PETScCompat.TSRKSetType(petsclib, ts, alg.subtype)
_set_subtype!(petsclib, ts, alg::TSRosW) =
    PETScCompat.TSRosWSetType(petsclib, ts, alg.subtype)
function _set_subtype!(petsclib, ts, alg::TSImplicit)
    if alg.subtype == "theta" && alg.theta !== nothing
        LibPETSc.TSThetaSetTheta(petsclib, ts, petsclib.PetscReal(alg.theta))
    end
    if alg.order !== nothing
        LibPETSc.TSBDFSetOrder(petsclib, ts, LibPETSc.PetscInt(alg.order))
    end
    return nothing
end
function _set_subtype!(petsclib, ts, alg::TSIRK)
    LibPETSc.TSIRKSetNumStages(petsclib, ts, LibPETSc.PetscInt(alg.nstages))
    # Set the type after the stage count: PETSc builds the tableau from it.
    PETScCompat.TSIRKSetType(petsclib, ts, "gauss")
    return nothing
end
_set_subtype!(petsclib, ts, alg::TSARKIMEX) =
    PETScCompat.TSARKIMEXSetType(petsclib, ts, alg.subtype)
function _set_subtype!(petsclib, ts, alg::TSDAE)
    if alg.order !== nothing
        LibPETSc.TSBDFSetOrder(petsclib, ts, LibPETSc.PetscInt(alg.order))
    end
    return nothing
end
_set_subtype!(petsclib, ts, ::TSMPRK) = nothing
_set_subtype!(petsclib, ts, ::TSGeneric) = nothing

_default_options(::AnyPETScTS) = String[]
# Without -ts_use_splitrhsfunction PETSc never calls the per-part functions.
_default_options(alg::TSMPRK) =
    ["-ts_mprk_type", alg.subtype, "-ts_use_splitrhsfunction", "true"]

const UNSUPPORTED_KWARGS = (
    :internalnorm, :calck, :alias_u0, :sensealg,
    :controller, :qmax, :qmin, :gamma, :beta1, :beta2, :failfactor,
    :step_limiter, :stage_limiter,
)

mutable struct TSHandles{CTX, L, R, S}
    ctx::CTX
    petsclib::L
    ts::Any
    u::Any
    jac_mat::Any
    fd_mat::Any
    ad_calls::Union{Nothing, Base.RefValue{Int}}
    opts::Any
    t0::R
    tf::R
    tdir::R
    u0::Vector{S}
    maxiters::Int
    save_start::Bool
    save_end::Bool
    pivot_raises::Bool
    stopped::Int
    matches::Bool
    fixed::Bool
    tolvecs::Vector{Any}
    tolbufs::Vector{Vector{S}}
    destroyed::Bool
end

# Finalizers run after atexit hooks, when freeing aborts, so exit frees live handles.
const LIVE_HANDLES = WeakKeyDict{Any, Int}()
const HANDLES_MADE = Ref(0)
# Freeing on a parallel comm is collective, so those handles are never left to the GC.
const PARALLEL_HANDLES = IdDict{Any, Nothing}()
const EXIT_CLEANUP_ARMED = Set{String}()

# atexit runs newest first, so arming after init puts this ahead of the build's teardown.
function _arm_exit_cleanup!(petsclib)
    lib = petsclib.petsc_library
    lib in EXIT_CLEANUP_ARMED && return nothing
    push!(EXIT_CLEANUP_ARMED, lib)
    atexit(() -> _destroy_live_handles!(lib))
    return nothing
end

# Creation order, so every rank frees its parallel handles in the same order.
function _destroy_live_handles!(lib)
    for (h, _) in sort!(collect(LIVE_HANDLES); by = last)
        h.petsclib.petsc_library == lib && _destroy!(h)
    end
    return nothing
end

const PETSC_LOCK = ReentrantLock()
_locked(f) = lock(f, PETSC_LOCK)

# A finalizer must not block on a lock, so it re-registers and retries later.
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
    delete!(PARALLEL_HANDLES, h)
    h.ts === nothing || delete!(POST_STEP_CTX, h.ts.ptr)
    (PETScCompat.isfinalized(h.petsclib) || MPI.Finalized()) && return nothing
    h.opts === nothing || PETScCompat.destroy!(h.opts)
    h.jac_mat === nothing || PETScCompat.destroy!(h.jac_mat)
    h.fd_mat === nothing || PETScCompat.destroy!(h.fd_mat)
    for v in h.tolvecs
        v.ptr == C_NULL || PETScCompat.destroy!(v)
    end
    h.ctx.work.ptr == C_NULL || PETScCompat.destroy!(h.ctx.work)
    h.u === nothing || PETScCompat.destroy!(h.u)
    h.ts === nothing || _return_work_vec!(h.ctx, h.ts)
    _release_work_vec!(h.ctx)
    h.ts === nothing || LibPETSc.TSDestroy(h.petsclib, h.ts)
    return nothing
end

_tolscalar(R, tol, default) =
    R(tol === nothing || tol isa AbstractVector ? default : tol)

function _check_real(x, name; accept = v -> v isa Real)
    x === nothing || all(accept, x) || throw(
        ArgumentError("`$name` must be real, even for a complex state; got $(repr(x))"),
    )
    return nothing
end

_check_real_tol(tol, name) = _check_real(tol, name; accept = isreal)

function _check_tol(tol, n, name)
    _check_real_tol(tol, name)
    tol isa AbstractVector || return nothing
    length(tol) == n ||
        throw(ArgumentError("`$name` has length $(length(tol)), but the state has $n"))
    all(t -> real(t) >= 0, tol) || throw(ArgumentError("`$name` has a negative entry"))
    return nothing
end

function _set_tolerances!(h::TSHandles{<:Any, <:Any, R}, abstol, reltol) where {R}
    pl, n = h.petsclib, length(h.u0)
    novec = LibPETSc.PetscVec{typeof(pl)}()
    avec = _tolvec(h, pl, abstol, n, "abstol")
    rvec = _tolvec(h, pl, reltol, n, "reltol")
    LibPETSc.TSSetTolerances(
        pl, h.ts, _tolscalar(R, abstol, 1.0e-6), avec === nothing ? novec : avec,
        _tolscalar(R, reltol, 1.0e-3), rvec === nothing ? novec : rvec,
    )
    return nothing
end

function _tolvec(h::TSHandles{<:Any, <:Any, R, S}, petsclib, tol, n, name) where {R, S}
    tol isa AbstractVector || return nothing
    # PETSc borrows `buf` and reads it every step, so the handle keeps it alive.
    buf = Vector{S}(collect(tol))
    push!(h.tolbufs, buf)
    comm = h.ctx.comm
    v = comm === nothing ? PETScCompat.PetscVec(petsclib, buf) :
        LibPETSc.VecCreateMPIWithArray(
            petsclib, comm, LibPETSc.PetscInt(1), LibPETSc.PetscInt(n),
            LibPETSc.PetscInt(LibPETSc.PETSC_DECIDE), buf,
        )
    push!(h.tolvecs, v)
    return v
end

_state_vec(petsclib, ::Nothing, n) = PETScCompat.PetscVec(petsclib, n)
_state_vec(petsclib, comm::MPI.Comm, n) = LibPETSc.VecCreateMPI(
    petsclib, comm, LibPETSc.PetscInt(n), LibPETSc.PetscInt(LibPETSc.PETSC_DECIDE),
)
_work_vec(petsclib, ::Nothing, u, n) = PETScCompat.PetscVec(petsclib, n)
_work_vec(petsclib, ::MPI.Comm, u, n) = LibPETSc.VecDuplicate(petsclib, u)

function _global_norm(comm, u, t)
    x = zero(eltype(u))
    for ui in u
        x += abs2(ui)
    end
    sums = MPI.Allreduce([Float64(real(x)), Float64(length(u))], +, comm)
    return real(eltype(u))(sqrt(sums[1] / max(sums[2], 1)))
end

# Takes the user's `f` and time, so it runs before `f` is reversed.
function _initial_dt(
        f1, f2, u0, p, t0::R, tdir, order, abstol, reltol, dtmin, dtmax, comm = nothing,
    ) where {R}
    dtmin_floor = max(nextfloat(max(dtmin, eps(t0))), _min_step(t0))
    smalldt = max(dtmin_floor, R(1.0e-6))
    _everywhere(comm, isempty(u0)) && return smalldt
    norm = comm === nothing ? DiffEqBase.ODE_DEFAULT_NORM : (u, t) -> _global_norm(comm, u, t)
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
    sk = R.(abstol) .+ abs.(u0) .* R.(reltol)
    f0 = rhs(u0, t0)
    _everywhere(comm, all(isfinite, f0)) || return smalldt
    d0 = norm(u0 ./ sk, t0)
    d1 = norm(f0 ./ sk, t0)
    isnan(d1) && return smalldt
    dt0 = min(d0 < 1.0e-5 || d1 < 1.0e-5 ? smalldt : (d0 / d1) / 100, dtmax)
    dt0 < 10 * eps(R) && return smalldt
    f_next = rhs(u0 .+ (tdir * dt0) .* f0, t0 + tdir * dt0)
    _everywhere(comm, f0 == f_next) && return min(max(dtmin_floor, 100 * dt0), dtmax)
    d2 = norm((f_next .- f0) ./ sk, t0) / dt0
    m = max(d1, d2)
    dt1 = R(m <= 1.0e-15 ? max(1.0e-6, dt0 * 1.0e-3) : 10.0^(-(2 + log10(m)) / order))
    dt = max(dtmin_floor, min(100 * dt0, dt1, dtmax))
    return isfinite(dt) && dt > 0 ? dt : smalldt
end

const SupportedProblem = Union{SciMLBase.AbstractODEProblem, SciMLBase.AbstractDAEProblem}

const _NOT_SELF = "on a communicator other than MPI.COMM_SELF"

function _check_irk_layout(n, N, comm)
    nranks = MPI.Comm_size(comm)
    share = N ÷ nranks + (MPI.Comm_rank(comm) < N % nranks)
    _everywhere(comm, n == share) && return nothing
    throw(
        ArgumentError(
            "TSIRK $_NOT_SELF needs each rank to hold PETSc's own share of the state, " *
                "since PETSc lays out its stage vector that way: $(N ÷ nranks) rows" *
                (N % nranks == 0 ? "" : ", and one more on the first $(N % nranks) ranks"),
        ),
    )
end

const _DISTRIBUTED_IMPLICIT = ("beuler", "cn", "theta", "bdf", "rosw", "arkimex", "irk")

function _refuse_distributed(prob, alg, is_dae, N)
    alg isa Union{TSRK, TSRosW, TSImplicit, TSIRK, TSDAE, TSARKIMEX} ||
        alg isa TSGeneric && alg.explicit || throw(
        ArgumentError(
            "PETScDiffEq cannot run " *
                "$(alg isa TSGeneric ? "an implicit TSGeneric" : nameof(typeof(alg))) " *
                "$_NOT_SELF; TSRK, TSRosW, TSImplicit, TSIRK, TSDAE, TSARKIMEX and " *
                "TSGeneric(...; explicit = true) can",
        ),
    )
    has_jac = prob.f.jac !== nothing
    if !_uses_ifunction(alg)
        has_jac && throw(
            ArgumentError(
                "PETScDiffEq does not take a `jac` for an explicit method $_NOT_SELF; " *
                    "it never uses one, so leave it out",
            ),
        )
        return nothing
    end
    n = length(prob.u0)
    mass = is_dae ? nothing : prob.f.mass_matrix
    if !(mass === nothing || mass == LinearAlgebra.I)
        mass isa LinearAlgebra.Diagonal || throw(
            ArgumentError(
                "PETScDiffEq takes only a `Diagonal` mass matrix $_NOT_SELF, not a " *
                    "$(nameof(typeof(mass)))",
            ),
        )
        size(mass) == (n, n) || throw(
            ArgumentError(
                "the mass matrix is $(join(size(mass), " x ")), but this rank's block of " *
                    "the state has $n rows",
            ),
        )
    end
    proto = prob.f.jac_prototype
    if has_jac
        proto isa SparseMatrixCSC || throw(
            ArgumentError(
                "a `jac` $_NOT_SELF needs a sparse `jac_prototype` holding this rank's rows " *
                    "of the Jacobian, with global column indices",
            ),
        )
    else
        _petsc_differences(alg) || throw(
            ArgumentError(
                "PETScDiffEq cannot use `$(_autodiff(alg))` $_NOT_SELF, since it would call " *
                    "`f` a different number of times on each rank; give the problem a `jac`, " *
                    "or leave `autodiff` at its default, `AutoFiniteDiff()` there, for " *
                    "PETSc's colouring",
            ),
        )
        alg isa TSIRK && throw(
            ArgumentError(
                "TSIRK needs a `jac` $_NOT_SELF, since PETSc builds its coupled-stage " *
                    "matrix from one and has no finite-difference fallback for it",
            ),
        )
        proto isa SparseArrays.AbstractSparseMatrix || throw(
            ArgumentError(
                "without a `jac`, PETSc's colouring $_NOT_SELF needs a sparse " *
                    "`jac_prototype` holding this rank's rows of the Jacobian, with global " *
                    "column indices",
            ),
        )
    end
    size(proto) == (n, N) || throw(
        ArgumentError(
            "the `jac_prototype` is $(join(size(proto), " x ")), but $_NOT_SELF it holds " *
                "this rank's rows, so it must be $n x $N",
        ),
    )
    return nothing
end

# On i686 PETSc_jll's single builds fail BDF and ARKIMEX at stops.
const _SINGLE_BUILDS = Sys.ARCH !== :i686

_loaded_builds() = Type[
    PETSc.scalartype(pl) for pl in PETSc.petsclibs if
        PETSc.inttype(pl) === LibPETSc.PetscInt &&
        (_SINGLE_BUILDS || real(PETSc.scalartype(pl)) !== Float32)
]

function _eltypes(prob, builds = _loaded_builds())
    E, tE = eltype(prob.u0), eltype(prob.tspan)
    single = E === Float32 || E === ComplexF32
    R = single && tE === Float32 && (E <: Complex ? ComplexF32 : Float32) in builds ?
        Float32 : Float64
    S = E <: Complex ? Complex{R} : R
    U = single ? E : E <: Complex ? ComplexF64 : Float64
    return R, S, U
end

_build_name(S) = "$(real(S)) $(S <: Complex ? "complex" : "real")"

function _petsclib(S, builds = _loaded_builds())
    S in builds && return PETSc.getlib(; PetscScalar = S)
    throw(
        ArgumentError(
            "this problem needs PETSc's $(_build_name(S)) build, which PETSc.jl has not " *
                "loaded; with a library set by `PETSc.set_library!` it loads only that one, " *
                "and here it has loaded the $(join(map(_build_name, builds), ", ")) build",
        ),
    )
end

# DiffEqBase passes the prototype through `similar`, so its values are garbage.
function _structure(S, P::SparseMatrixCSC)
    k = SparseArrays.nnz(P)
    return SparseMatrixCSC{S, Int}(
        size(P)..., Vector{Int}(P.colptr), Vector{Int}(P.rowval[1:k]), zeros(S, k),
    )
end

# PETSc cannot parse Julia's Float32 printing (`1.0f-5`).
_option(x::Real) = string(Float64(x))

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
        eltypes = _eltypes(prob),
        kwargs...,
    )
    for key in UNSUPPORTED_KWARGS
        if haskey(kwargs, key)
            @warn "PETScDiffEq does not support `$key` and is ignoring it"
        end
    end
    get(kwargs, :progress, false) === true &&
        @warn "PETScDiffEq does not support `progress` and is ignoring it"
    prob.u0 isa AbstractVector{<:Union{Real, Complex}} || throw(
        ArgumentError("PETScDiffEq requires an AbstractVector u0 of real or complex numbers"),
    )
    for (name, value) in (
            (:dt, dt), (:dtmin, dtmin), (:dtmax, dtmax), (:saveat, saveat), (:tstops, tstops),
        )
        _check_real(value, name)
    end
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
    if has_mass && is_split
        throw(ArgumentError("PETScDiffEq does not support a mass matrix on a SplitODEProblem"))
    end
    comm = _distributed(alg) ? alg.comm : nothing
    N = comm === nothing ? length(prob.u0) : MPI.Allreduce(length(prob.u0), +, comm)
    comm === nothing || _checked_everywhere(comm) do
        _refuse_distributed(prob, alg, is_dae, N)
    end

    R, S, U = eltypes
    t0, tf = R(prob.tspan[1]), R(prob.tspan[2])
    t0 == tf && throw(ArgumentError("PETScDiffEq requires tspan[1] != tspan[2]"))
    # From here on, times are PETSc's forward-running s = tdir * t.
    tdir = t0 < tf ? one(R) : -one(R)
    t0, tf = tdir * t0, tdir * tf

    u0 = Vector{S}(vec(prob.u0))
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

    petsclib = _petsclib(S)
    _check_inttype(petsclib)
    PETScCompat.isinitialized(petsclib) || PETSc.initialize(petsclib)
    _arm_exit_cleanup!(petsclib)

    iip = SciMLBase.isinplace(prob)
    # SciMLBase's wrapper is typed for the problem's types, so unwrap where PETSc's differ.
    unwrap = eltype(prob.u0) === S && eltype(prob.tspan) === R ? identity :
        SciMLBase.unwrapped_f
    f1 = unwrap(is_split ? prob.f.f1.f : prob.f.f)
    is_dae && !iip &&
        throw(ArgumentError("PETScDiffEq requires an in-place DAEProblem residual"))
    f2 = is_split ? unwrap(prob.f.f2.f) : nothing
    for g in (f1, f2)
        g isa SciMLOperators.AbstractSciMLOperator && throw(
            ArgumentError(
                "PETScDiffEq does not support an operator-valued right-hand side; " *
                    "supply a function f!(du, u, p, t)",
            ),
        )
    end
    f1 = is_dae ? f1 : _as_inplace(f1, iip)
    f2 = f2 === nothing ? nothing : _as_inplace(f2, iip)
    builds_jac = _uses_ifunction(alg) && prob.f.jac === nothing && !_petsc_differences(alg)
    has_jac = _uses_ifunction(alg) && (prob.f.jac !== nothing || builds_jac)
    ad_calls = builds_jac ? Ref(0) : nothing
    jac_fn = if !has_jac
        nothing
    elseif builds_jac
        # Dual numbers need the unwrapped function.
        f_ad = SciMLBase.unwrapped_f(is_split ? prob.f.f1.f : prob.f.f)
        user_t0 = R(prob.tspan[1])
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
    _checked_everywhere(comm) do
        _check_tol(abstol, n, "abstol")
        _check_tol(reltol, n, "reltol")
    end
    if !dt_given
        user_t0 = R(prob.tspan[1])
        est_dtmin = dtmin === nothing ? zero(R) : abs(R(dtmin))
        dt = if is_dae
            max(R(1.0e-6) * abs(tf - t0), _min_step(user_t0))
        elseif has_mass
            max(nextfloat(max(est_dtmin, eps(user_t0))), R(1.0e-6), _min_step(user_t0))
        else
            est_abstol = something(abstol, 1.0e-6)
            est_reltol = something(reltol, 1.0e-3)
            user_dtmax = dtmax === nothing || isinf(dtmax) ? R(Inf) : abs(R(dtmax))
            first_stop = minimum(
                (abs(R(s) - user_t0) for s in tstops if tdir * (R(s) - user_t0) > 0);
                init = R(Inf),
            )
            threw = Ref{Any}(nothing)
            estimate = _initial_dt(
                _guard_f(f1, comm, threw),
                f2 === nothing ? nothing : _guard_f(f2, comm, threw), u0, prob.p, user_t0, tdir,
                SciMLBase.alg_order(alg), est_abstol, est_reltol, est_dtmin,
                min(user_dtmax, first_stop, abs(tf - t0)), comm,
            )
            _throw_anywhere(comm, threw[])
            estimate
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
        zeros(S, 0, 0)
    elseif uses_sparse_jac
        _structure(S, jac_prototype)
    else
        zeros(S, n, n)
    end
    saveat_times = saveat isa Number ?
        collect(R, t0:abs(R(saveat)):tf) :
        sort!(tdir .* Vector{R}(collect(saveat)))
    filter!(t -> t0 - eps(tf) <= t <= tf + eps(tf), saveat_times)
    at_start(t) = abs(t - t0) <= _near(t0)
    at_end(t) = abs(t - tf) <= _near(tf)
    no_saveat = !(saveat isa Number) && isempty(saveat)
    save_everystep = save_on && something(save_everystep, no_saveat)
    save_start = something(
        save_start, save_everystep || no_saveat || saveat isa Number ||
            any(at_start, saveat_times),
    )
    save_end = something(
        save_end, save_everystep || no_saveat || saveat isa Number || any(at_end, saveat_times),
    )
    save_on || empty!(saveat_times)
    save_start || filter!(!at_start, saveat_times)
    save_end || filter!(!at_end, saveat_times)
    M = if !has_mass
        nothing
    elseif comm === nothing
        Matrix{S}(mass_matrix)
    else
        LinearAlgebra.Diagonal(Vector{S}(mass_matrix.diag))
    end
    missing_diag = uses_sparse_jac ?
        [i for i in 1:n if !_stored(J0, i, i)] : Int[]
    W0 = has_jac && !uses_sparse_jac ? zeros(S, n, n) : zeros(S, 0, 0)
    idx0 = has_jac && !uses_sparse_jac ?
        LibPETSc.PetscInt[i - 1 for i in 1:n] : LibPETSc.PetscInt[]
    row_cols0, row_src, row_buf = uses_sparse_jac && comm === nothing ?
        _row_structure(J0, n, M) : (Vector{LibPETSc.PetscInt}[], Vector{Int}[], Vector{S}[])
    kept = if save_idxs === nothing
        nothing
    else
        _checked_everywhere(comm) do
            v = save_idxs isa Integer ? [Int(save_idxs)] : Vector{Int}(collect(save_idxs))
            isempty(v) && throw(ArgumentError("`save_idxs` must name at least one component"))
            all(i -> 1 <= i <= n, v) || throw(
                ArgumentError("`save_idxs` has an index outside 1:$n"),
            )
            v
        end
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
    uvec = _state_vec(petsclib, comm, n)
    ctx = TSContext(
        petsclib, f1, f2, jac_fn, prob.p,
        similar(u0), similar(u0), similar(u0), similar(u0), M, is_dae, missing_diag, W0,
        idx0,
        row_cols0, row_src, row_buf, J0,
        R[], Vector{U}[], Vector{U}[], nothing, nothing,
        saveat_times, 1, save_everystep, save_start, dense_out, kept,
        _work_vec(petsclib, comm, uvec, n),
        !has_mass && !is_dae && !_petsc_interpolant(alg), _interpolates(alg), _warn_name(alg),
        R(NaN), similar(u0), t0, copy(u0), nothing, nothing, false,
        slow_idxs, medium_idxs, fast_idxs,
        R(NaN), similar(u0), false,
        _floor(R, dtmin, force_dtmin, adaptive && _adapts(alg) !== false), false,
        unstable_check, false, tdir, isoutofdomain,
        0, 0, 0, nothing, comm,
        comm === nothing && adaptive && _adapts(alg) !== false && !_uses_ifunction(alg),
        C_NULL, 0, false, nothing,
    )
    h = TSHandles(
        ctx, petsclib, nothing, uvec, nothing, nothing, ad_calls, nothing,
        t0, tf, tdir, u0, Int(maxiters), save_start, save_end, false, 0, false, false,
        Any[], Vector{S}[], false,
    )
    if comm !== nothing && MPI.Comm_size(comm) > 1
        PARALLEL_HANDLES[h] = nothing
    else
        finalizer(_finalize!, h)
    end
    LIVE_HANDLES[h] = (HANDLES_MADE[] += 1)

    try
        h.ts = LibPETSc.TSCreate(petsclib, something(comm, MPI.COMM_SELF))
        ts = h.ts
        LibPETSc.TSSetProblemType(petsclib, ts, LibPETSc.TS_NONLINEAR)
        LibPETSc.TSSetType(petsclib, ts, _ts_type(alg))
        _set_subtype!(petsclib, ts, alg)

        u = h.u
        PETScCompat.with_local_array!(u; read = false, write = true) do ua
            copyto!(ua, u0)
        end
        LibPETSc.TSSetSolution(petsclib, ts, u)

        ctxptr = pointer_from_objref(ctx)
        ptrs = _callbacks(petsclib)
        GC.@preserve ctx begin
            if _uses_ifunction(alg)
                LibPETSc.TSSetIFunction(petsclib, ts, nothing, ptrs.ifunction, ctxptr)
            else
                LibPETSc.TSSetRHSFunction(petsclib, ts, nothing, ptrs.rhs, ctxptr)
            end
            if alg isa TSMPRK
                _set_split!(petsclib, ts, "slow", slow_idxs, ptrs.mprk_slow, ctxptr)
                isempty(medium_idxs) || _set_split!(
                    petsclib, ts, "medium", medium_idxs, ptrs.mprk_medium, ctxptr,
                )
                _set_split!(petsclib, ts, "fast", fast_idxs, ptrs.mprk_fast, ctxptr)
            end
            if is_split
                LibPETSc.TSSetRHSFunction(petsclib, ts, nothing, ptrs.split_rhs, ctxptr)
            end
            if comm !== nothing && _uses_ifunction(alg)
                rstart = first(LibPETSc.VecGetOwnershipRange(petsclib, u))
                P = has_jac ? J0 : _structure(S, SparseMatrixCSC(prob.f.jac_prototype))
                rows, cols, coo = _coo_structure(P, rstart, M)
                mat = LibPETSc.MatCreate(petsclib, comm)
                has_jac ? (h.jac_mat = mat) : (h.fd_mat = mat)
                _coo_matrix!(mat, petsclib, n, N, rows, cols)
                if has_jac
                    ctx.coo = coo
                    LibPETSc.TSSetIJacobian(
                        petsclib, ts, mat, mat, ptrs.sparse_ijacobian, ctxptr,
                    )
                else
                    LibPETSc.MatSetValuesCOO(petsclib, mat, coo.vals, LibPETSc.INSERT_VALUES)
                    PETSc.assemble!(mat)
                    _colour_jacobian!(petsclib, ts, mat)
                end
            elseif has_jac && uses_sparse_jac
                pattern = _jacobian_pattern(J0, n, M)
                h.jac_mat = PETScCompat.PetscMat(
                    petsclib, MPI.COMM_SELF, pattern; with_arrays = true,
                )
                LibPETSc.TSSetIJacobian(
                    petsclib, ts, h.jac_mat, h.jac_mat, ptrs.sparse_ijacobian, ctxptr,
                )
            elseif has_jac
                h.jac_mat = PETScCompat.PetscMat(petsclib, zeros(S, n, n))
                LibPETSc.TSSetIJacobian(
                    petsclib, ts, h.jac_mat, h.jac_mat, ptrs.ijacobian, ctxptr,
                )
            elseif _uses_ifunction(alg) && prob.f.jac_prototype isa SparseArrays.AbstractSparseMatrix
                h.fd_mat = PETScCompat.PetscMat(
                    petsclib, MPI.COMM_SELF, _fd_pattern(prob.f.jac_prototype, M, n);
                    with_arrays = true,
                )
                _colour_jacobian!(petsclib, ts, h.fd_mat)
            end
            LibPETSc.TSMonitorSet(petsclib, ts, ptrs.monitor, ctxptr)
            if ctx.dtmin > 0 || ctx.unstable !== nothing || comm !== nothing
                _set_post_step!(petsclib, ts, ctx)
            end
            LibPETSc.TSSetTime(petsclib, ts, t0)
            # PETSc's floor clamps only the steps its adaptor chooses, not the one given.
            LibPETSc.TSSetTimeStep(
                petsclib, ts,
                force_dtmin && dtmin !== nothing ? max(abs(R(dt)), abs(R(dtmin))) : abs(R(dt)),
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
            _set_tolerances!(h, something(abstol, 1.0e-6), something(reltol, 1.0e-3))
            effective_options = ["-ts_error_if_step_fails", "false"]
            append!(effective_options, _default_options(alg))
            # PETSc's sparse LU does not pivot, and an algebraic row has a zero diagonal.
            if h.jac_mat !== nothing && uses_sparse_jac || h.fd_mat !== nothing
                append!(effective_options, ["-pc_factor_nonzeros_along_diagonal"])
                comm === nothing ||
                    append!(effective_options, ["-sub_pc_factor_nonzeros_along_diagonal"])
            end
            adaptive || append!(effective_options, ["-ts_adapt_type", "none"])
            forced = force_dtmin && dtmin !== nothing && dtmin != 0
            forced &&
                append!(effective_options, ["-ts_adapt_dt_min", _option(abs(R(dtmin)))])
            (dtmax === nothing || isinf(dtmax)) || append!(
                effective_options,
                [
                    "-ts_adapt_dt_max",
                    _option(_above(abs(R(dtmax)), forced ? abs(R(dtmin)) : zero(R))),
                ],
            )
            append!(effective_options, alg.petsc_options)
            append!(effective_options, extra_options)
            if !isempty(effective_options)
                parsed = PETSc.parse_options(effective_options)
                h.opts = PETScCompat.PetscOptions(petsclib; parsed...)
                push!(h.opts)
                try
                    LibPETSc.TSSetFromOptions(petsclib, ts)
                finally
                    pop!(h.opts)
                end
            else
                LibPETSc.TSSetFromOptions(petsclib, ts)
            end
            # An option can change the type or subtype, so refuse on what PETSc runs.
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
            !_uses_ifunction(alg) && chosen == "irk" && throw(
                ArgumentError(
                    "`irk` is an implicit PETSc type, so it needs `TSIRK` or " *
                        "`TSGeneric(\"irk\")` rather than an option on an explicit algorithm",
                ),
            )
            distributable = _uses_ifunction(alg) ? _DISTRIBUTED_IMPLICIT : _EXPLICIT_ONLY
            comm === nothing || chosen in distributable || throw(
                ArgumentError(
                    "`$chosen` cannot run $_NOT_SELF when an option picks it for " *
                        "$(nameof(typeof(alg))); only $(join(distributable, ", ")) can",
                ),
            )
            comm === nothing || chosen != "irk" || _check_irk_layout(n, N, comm)
            running = _running_name(petsclib, ts)
            _refuse_method(running, has_mass, has_jac, is_split, is_dae)
            # PETSc's IRK needs an AIJ Jacobian, even when picked by an option.
            if chosen == "irk" && has_jac && !uses_sparse_jac
                PETScCompat.destroy!(h.jac_mat)
                h.jac_mat = PETScCompat.PetscMat(petsclib, n, n, n)
                LibPETSc.TSSetIJacobian(
                    petsclib, ts, h.jac_mat, h.jac_mat, ptrs.ijacobian, ctxptr,
                )
            end
            # PETSc's default LU cannot factor IRK's Kronecker-product stage matrix.
            chosen == "irk" && !any(o -> _names_option(o, "pc_type"), effective_options) &&
                _set_pc_type!(petsclib, ts, "pbjacobi")
            # Only valid once TSSetFromOptions has reached the linear solve.
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
        nreject = Int(LibPETSc.TSGetStepRejections(pl, ts)) + h.ctx.nreject,
        nnonliniter = Int(LibPETSc.TSGetSNESIterations(pl, ts)),
        nnonlinfail = Int(LibPETSc.TSGetSNESFailures(pl, ts)),
    )
end

const _NORM_FLOOR = sqrt(floatmin(Float32))

_maxabs(u, ::Nothing) = maximum(abs, u; init = 0.0f0)
_maxabs(u, comm::MPI.Comm) = MPI.Allreduce(maximum(abs, u; init = 0.0f0), max, comm)

_tiny(u, comm) = 0 < _maxabs(u, comm) < _NORM_FLOOR

_underflows(::TSHandles{<:Any, <:Any, Float64}, alg, uend, retcode) = false
_underflows(h::TSHandles{<:Any, <:Any, Float32}, alg, uend, retcode) =
    _uses_ifunction(alg) && _tiny(uend, h.ctx.comm) &&
    (_tiny(h.u0, h.ctx.comm) || retcode != SciMLBase.ReturnCode.Success)

function _assemble(prob, alg, h::TSHandles, tend, uend, st, kwargs)
    ctx = h.ctx
    tf, t0, tol = h.tf, h.t0, _near(h.tf)
    # -ts_exact_final_time interpolate reports a point past tf mid-sequence.
    keep = findall(t -> t <= tf + tol, ctx.ts)
    if length(keep) != length(ctx.ts)
        ctx.ts = ctx.ts[keep]
        ctx.us = ctx.us[keep]
        ctx.dense && (ctx.dus = ctx.dus[keep])
    end
    if h.save_end && (
            isempty(ctx.ts) || ctx.ts[end] < tend - tol ||
                _anywhere(ctx.comm, ctx.ts[end] <= tend + tol && ctx.us[end] != _saved(ctx, uend))
        )
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
    if !h.save_start && length(ctx.ts) > 1 && abs(ctx.ts[1] - t0) <= _near(t0)
        popfirst!(ctx.ts)
        popfirst!(ctx.us)
        ctx.dense && popfirst!(ctx.dus)
    end

    finite = _everywhere(ctx.comm, all(isfinite, uend))
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
    if h.jac_mat !== nothing && h.ad_calls === nothing && st.nsteps > 0 && ctx.njacs == 0
        @warn "`$(_ts_type(alg))` took $(st.nsteps) steps without ever calling the " *
            "Jacobian this package gave PETSc, so it is not solving implicitly and the " *
            "result should not be trusted; an explicit PETSc type needs `explicit = true`"
    end
    _underflows(h, alg, uend, retcode) && @warn "`$(_warn_name(alg))` ended in PETSc's " *
        "single-precision build on a state whose entries are all below " *
        "$(Float32(_NORM_FLOOR)) in size, where that build's vector norms underflow to " *
        "zero, so its Newton iteration can stop without moving the state, whether it " *
        "reports success or fails. Rescale the problem, or give it a Float64 span to solve " *
        "it in double precision"
    stats = SciMLBase.DEStats(
        _nf(h), ctx.nf2, -1, -1, ctx.njacs, st.nnonliniter, st.nnonlinfail, -1, -1, -1,
        st.nsteps, st.nreject, 0.0,
    )
    ts, dus = _user_time(h)
    return SciMLBase.build_solution(
        prob, alg, ts, ctx.us; retcode = retcode, stats = stats,
        dense = ctx.dense, interp = _interp(ctx, ts, dus),
        timeseries_errors = get(kwargs, :timeseries_errors, true),
        dense_errors = get(kwargs, :dense_errors, false),
    )
end

# Block Jacobi reads its sub-solvers' options when it first sets them up, inside the solve.
function _with_options(f, h::TSHandles)
    h.ctx.comm === nothing && return f()
    push!(h.opts)
    try
        return f()
    finally
        pop!(h.opts)
    end
end

# Under a distributed comm the error is returned, to raise once the ranks agree on who threw.
function _run!(f, h::TSHandles)
    ctx = h.ctx
    GC.@preserve ctx begin
        try
            _quiet_errors(() -> _with_options(f, h), h)
        catch e
            if ctx.comm === nothing
                ctx.err === nothing && !_failed_step(e, h) && rethrow()
            elseif !_failed_step(e, h)
                return e
            end
            h.stopped = e.code
        end
    end
    return nothing
end

function _solve_unlocked(
        prob::SupportedProblem, alg::AnyPETScTS;
        callback = nothing, tstops = (), d_discontinuities = (), kwargs...,
    )
    # MPRK misses tf, Float32 landing refuses short steps, and isoutofdomain retries steps.
    if !_no_callback(callback) || !isempty(tstops) || !isempty(d_discontinuities) ||
            alg isa TSMPRK || get(kwargs, :isoutofdomain, nothing) !== nothing ||
            first(_eltypes(prob)) === Float32
        integ = _init_unlocked(
            prob, alg; callback = callback, tstops = tstops,
            d_discontinuities = d_discontinuities, kwargs...,
        )
        try
            return _solve_integrator_unlocked(integ)
        catch
            _destroy!(integ.h)
            rethrow()
        end
    end
    h = _setup(prob, alg; kwargs...)
    ctx, pl = h.ctx, h.petsclib
    floor = abs(oftype(h.t0, something(get(kwargs, :dtmin, nothing), 0.0)))
    forced = get(kwargs, :force_dtmin, false) === true
    tend, uend, st = h.t0, copy(h.u0), nothing
    try
        if ctx.comm === nothing &&
                LibPETSc.TSAdaptGetType(pl, LibPETSc.TSGetAdapt(pl, h.ts)) == "none"
            ctx.halt_nonfinite = true
            _set_post_step!(pl, h.ts, ctx)
        end
        raised, failure = false, nothing
        while true
            _release_work_vec!(ctx)
            failure = _run!(h) do
                LibPETSc.TSSolve(pl, h.ts, h.u)
            end
            raised = h.stopped != 0
            raised && ctx.err === nothing && failure === nothing &&
                _retry_solve!(h, alg, floor, forced, kwargs) || break
        end
        _throw_if_threw!(ctx)
        failure === nothing || throw(failure)
        h.stopped == 0 || _warn_failed_step(alg, h.stopped, kwargs)
        # PETSc sets the solve time only when TSSolve returns normally, and a step that
        # raises leaves its rejected trial in the solution vector.
        tend, uend = raised ? (ctx.end_s, copy(ctx.end_u)) :
            (LibPETSc.TSGetSolveTime(pl, h.ts), _readvec!(similar(h.u0), pl, h.u))
        st = _read_stats(h)
    finally
        _destroy!(h)
    end
    sol = _assemble(prob, alg, h, tend, uend, st, kwargs)
    ctx.comm === nothing || _throw_if_threw!(ctx)
    return sol
end

function _retry_solve!(h, alg, floor, forced, kwargs)
    ctx, pl = h.ctx, h.petsclib
    h.stopped == PETSC_ERR_FP && _return_work_vec!(ctx, h.ts) || return false
    h.stopped = 0
    dt, _ = _retry_step(h, ctx.end_s, LibPETSc.TSGetTimeStep(pl, h.ts), floor, forced, true)
    dt === nothing && (_warn_failed_step(alg, PETSC_ERR_FP, kwargs); return false)
    # TSSolve zeroes its counters when it starts on step 0.
    LibPETSc.TSGetStepNumber(pl, h.ts) == 0 &&
        (ctx.nreject += Int(LibPETSc.TSGetStepRejections(pl, h.ts)))
    PETScCompat.with_local_array!(ua -> copyto!(ua, ctx.end_u), h.u; read = false, write = true)
    ctx.hermite && (ctx.fend = ctx.fstart)
    LibPETSc.TSSetTimeStep(pl, h.ts, dt)
    LibPETSc.TSRestartStep(pl, h.ts)
    return true
end

SciMLBase.__solve(prob::SupportedProblem, alg::AnyPETScTS; kwargs...) =
    _locked(() -> _solve_unlocked(prob, alg; kwargs...))

mutable struct PETScIntegratorOpts{H, R}
    h::H
    adaptive::Bool
    abstol::Any
    reltol::Any
    dtmin::R
    dtmax::R
    verbose::Bool
    force_dtmin::Bool
end

function _setopt_unlocked(o::PETScIntegratorOpts{H, R}, name::Symbol, v) where {H, R}
    h = getfield(o, :h)
    name in (:abstol, :reltol) &&
        _checked_everywhere(() -> _check_tol(v, length(h.u0), name), h.ctx.comm)
    setfield!(o, name, name in (:dtmin, :dtmax) ? R(v) : v)
    (h === nothing || h.destroyed) && return v
    pl = h.petsclib
    if name === :abstol || name === :reltol
        _set_tolerances!(h, getfield(o, :abstol), getfield(o, :reltol))
    elseif name === :dtmin && getfield(o, :force_dtmin)
        adapt = LibPETSc.TSGetAdapt(pl, h.ts)
        _, hi = LibPETSc.TSAdaptGetStepLimits(pl, adapt)
        lo = abs(getfield(o, :dtmin))
        LibPETSc.TSAdaptSetStepLimits(pl, adapt, lo, _above(hi, lo))
    elseif name === :dtmin
        h.ctx.dtmin = _floor(R, getfield(o, :dtmin), false, getfield(o, :adaptive))
    elseif name === :dtmax
        adapt = LibPETSc.TSGetAdapt(pl, h.ts)
        lo, _ = LibPETSc.TSAdaptGetStepLimits(pl, adapt)
        hi = abs(getfield(o, :dtmax))
        LibPETSc.TSAdaptSetStepLimits(pl, adapt, lo, isfinite(hi) ? _above(hi, lo) : floatmax(R))
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

These are in the types PETSc steps in. The clock, and so `t`, `dt` and the saved times, is
`Float32` for a `Float32` or `ComplexF32` state with a `Float32` span and `Float64`
otherwise, whatever the span's type. The state is the problem's own, except that a
whole-number one is stepped in `Float64` and a single-precision one with a `Float64` span in
double precision, while the solution it saves stays in single.
"""
mutable struct PETScIntegrator{Alg, S, R, P, H, Pr, CB, CC} <:
    SciMLBase.AbstractODEIntegrator{Alg, true, Vector{S}, R}
    alg::Alg
    u::Vector{S}
    uprev::Vector{S}
    t::R
    tprev::R
    dt::R
    tdir::R
    p::P
    h::H
    prob::Pr
    callbacks::CB
    continuous::CC
    f::Any
    opts::Any
    ucache::Vector{S}
    tmp1::Vector{S}
    tmp2::Vector{S}
    event_t::Vector{Vector{R}}
    event_residual::Vector{Vector{Float64}}
    kwargs::Any
    tstops::Vector{R}
    tstops_cache::Vector{R}
    d_discontinuities::Vector{R}
    d_discontinuities_cache::Vector{R}
    dtcache::R
    sol::Any
    finished::Bool
    derivative_discontinuity::Bool
end

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

function _set_p_unlocked(integ::PETScIntegrator, v)
    setfield!(integ, :p, convert(fieldtype(typeof(integ), :p), v))
    h = integ.h
    h.ctx.p = integ.p
    h.ctx.pdirty = true
    # An FSAL method reuses its last stage's slope, taken with the old p, unless restarted.
    # BDF keeps only past states, which stay valid, and a restart drops it to first order.
    h.destroyed || h.ctx.alg_name == "bdf" || LibPETSc.TSRestartStep(h.petsclib, h.ts)
    return v
end

Base.setproperty!(integ::PETScIntegrator, name::Symbol, v) = name === :p ?
    _locked(() -> _set_p_unlocked(integ, v)) :
    setfield!(integ, name, convert(fieldtype(typeof(integ), name), v))

SciMLBase.get_dt(integ::PETScIntegrator) = integ.dt
function _proposed_dt_unlocked(integ::PETScIntegrator)
    integ.finished && return abs(integ.dt)
    return LibPETSc.TSGetTimeStep(integ.h.petsclib, integ.h.ts)
end

SciMLBase.get_proposed_dt(integ::PETScIntegrator) = _locked(() -> _proposed_dt_unlocked(integ))
function _set_proposed_dt_unlocked(integ::PETScIntegrator, dt)
    integ.finished || LibPETSc.TSSetTimeStep(
        integ.h.petsclib, integ.h.ts, _smallest(integ.h.ctx.comm, abs(oftype(integ.t, dt))),
    )
    return nothing
end

SciMLBase.set_proposed_dt!(integ::PETScIntegrator, dt) =
    _locked(() -> _set_proposed_dt_unlocked(integ, dt))
_make_opts(h::TSHandles{<:Any, <:Any, R}, kwargs) where {R} = PETScIntegratorOpts(
    h, get(kwargs, :adaptive, true) === true,
    get(kwargs, :abstol, 1.0e-6), get(kwargs, :reltol, 1.0e-3),
    R(something(get(kwargs, :dtmin, nothing), 0.0)),
    R(something(get(kwargs, :dtmax, nothing), Inf)),
    get(kwargs, :verbose, true) === true, get(kwargs, :force_dtmin, false) === true,
)

SciMLBase.isadaptive(integ::PETScIntegrator) =
    getfield(integ.opts, :adaptive) && _adapts(integ.alg) !== false

(integ::PETScIntegrator)(t::Number) = copy(_checked_state(integ, t))
(integ::PETScIntegrator)(t::Number, ::Type{Val{0}}) = copy(_checked_state(integ, t))
(integ::PETScIntegrator)(out::AbstractArray, t) = copyto!(out, _checked_state(integ, t))
(integ::PETScIntegrator)(out::AbstractArray, t, ::Type{Val{0}}) =
    copyto!(out, _checked_state(integ, t))

function _raise_threw!(integ::PETScIntegrator)
    ctx = integ.h.ctx
    (ctx.comm === nothing || !_threw!(ctx)) && return nothing
    # The cached slopes may hold the NaN that stood in for `f`.
    err, ctx.err, ctx.fstart, ctx.fend = ctx.err, nothing, nothing, nothing
    throw(err)
end

function _checked_du(integ::PETScIntegrator)
    du = integ.tdir .* _derivative(integ.h.ctx, integ.tdir * integ.t, integ.u)
    _raise_threw!(integ)
    return du
end

SciMLBase.get_du(integ::PETScIntegrator) = _checked_du(integ)
function SciMLBase.get_du!(out, integ::PETScIntegrator)
    copyto!(out, _checked_du(integ))
    return out
end
SciMLBase.get_tmp_cache(integ::PETScIntegrator) = (integ.tmp1, integ.tmp2)
DiffEqBase.get_tstops(integ::PETScIntegrator) = integ.tstops
DiffEqBase.get_tstops_array(integ::PETScIntegrator) = integ.tstops
DiffEqBase.get_tstops_max(integ::PETScIntegrator) = last(integ.tstops)

function _set_u_unlocked(integ::PETScIntegrator, u)
    copyto!(integ.u, u)
    integ.finished && return nothing
    PETScCompat.with_local_array!(
        ua -> copyto!(ua, integ.u), integ.h.u; read = false, write = true,
    )
    LibPETSc.TSRestartStep(integ.h.petsclib, integ.h.ts)
    return nothing
end

SciMLBase.set_u!(integ::PETScIntegrator, u) = _locked(() -> _set_u_unlocked(integ, u))

function _set_t_unlocked(integ::PETScIntegrator, t)
    integ.t = oftype(integ.t, t)
    _end_step_here!(integ)
    integ.finished || LibPETSc.TSSetTime(integ.h.petsclib, integ.h.ts, integ.tdir * integ.t)
    return nothing
end

SciMLBase.set_t!(integ::PETScIntegrator, t) = _locked(() -> _set_t_unlocked(integ, t))

function SciMLBase.add_saveat!(integ::PETScIntegrator, t)
    t = oftype(integ.t, t)
    ctx = integ.h.ctx
    s = integ.tdir * t
    s < integ.tdir * integ.t &&
        throw(ArgumentError("cannot add a saveat at $t, behind the current time $(integ.t)"))
    i = searchsortedfirst(ctx.saveat, s)
    (i <= length(ctx.saveat) && ctx.saveat[i] == s) || insert!(ctx.saveat, i, s)
    i < ctx.saveat_idx && (ctx.saveat_idx += 1)
    return nothing
end

function _change_t_unlocked(
        integ::PETScIntegrator, t, modify_save_endpoint::Type{Val{T}} = Val{false},
    ) where {T}
    integ.finished && return nothing
    t = oftype(integ.t, t)
    copyto!(integ.u, _state_at(integ, t))
    integ.t = t
    integ.dt = integ.t - integ.tprev
    _end_step_here!(integ)
    PETScCompat.with_local_array!(
        ua -> copyto!(ua, integ.u), integ.h.u; read = false, write = true,
    )
    LibPETSc.TSSetTime(integ.h.petsclib, integ.h.ts, integ.tdir * t)
    LibPETSc.TSRestartStep(integ.h.petsclib, integ.h.ts)
    _raise_threw!(integ)
    return nothing
end

SciMLBase.change_t_via_interpolation!(
    integ::PETScIntegrator, t, modify_save_endpoint::Type{Val{T}} = Val{false},
) where {T} = _locked(() -> _change_t_unlocked(integ, t, modify_save_endpoint))

function _save_here!(integ::PETScIntegrator)
    get(integ.kwargs, :save_on, true) && _record!(integ.h.ctx, integ.tdir * integ.t, integ.u)
    return nothing
end

function _savevalues_unlocked(integ::PETScIntegrator, force_save = false)
    (integ.finished || !get(integ.kwargs, :save_on, true)) && return (false, false)
    ctx = integ.h.ctx
    n = length(ctx.ts)
    _save_step!(integ, integ.t, false)
    s = integ.tdir * integ.t
    if force_save || (ctx.save_everystep && !_last_recorded(ctx, s))
        _record!(ctx, s, integ.u)
    end
    saved = length(ctx.ts) > n
    _raise_threw!(integ)
    return (saved, saved && ctx.ts[end] == s)
end

SciMLBase.savevalues!(integ::PETScIntegrator, force_save = false) =
    _locked(() -> _savevalues_unlocked(integ, force_save))

function SciMLBase.step!(integ::PETScIntegrator, dt, stop_at_tdt = false)
    integ.tdir * dt < 0 && throw(ArgumentError("cannot step backward in time"))
    next_t = integ.t + oftype(integ.t, dt)
    tf = _user_t(integ.tdir, integ.h.tf)
    stop_at_tdt && integ.tdir * next_t < integ.tdir * tf && SciMLBase.add_tstop!(integ, next_t)
    while !integ.finished && integ.tdir * integ.t < integ.tdir * next_t
        SciMLBase.step!(integ)
    end
    return nothing
end

function _state_at_unlocked(integ::PETScIntegrator, t)
    # Snap only from outside the step: Float32 root-finder probes can round onto an end.
    s, s0, s1 = integ.tdir * t, integ.tdir * integ.tprev, integ.tdir * integ.t
    (s == s1 || s1 < s <= s1 + _near(integ.t)) && return integ.u
    (s == s0 || s0 - _near(integ.tprev) <= s < s0) && return integ.uprev
    s0 <= s <= s1 || throw(
        ArgumentError(
            "PETScDiffEq can only interpolate inside the step just taken, " *
                "$(integ.tprev) to $(integ.t), but $t was asked for",
        ),
    )
    return _interpolate!(integ, s)
end

_state_at(integ::PETScIntegrator, t) =
    _locked(() -> _state_at_unlocked(integ, oftype(integ.t, t)))

function _checked_state(integ::PETScIntegrator, t)
    u = _state_at(integ, t)
    _raise_threw!(integ)
    return u
end

# `s` is PETSc's time. Returns `integ.ucache`, which the next call overwrites.
function _interpolate!(integ::PETScIntegrator, s)
    s = oftype(integ.t, s)
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

function _interpolate_finished!(integ::PETScIntegrator, s)
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

function _pin_step!(integ::PETScIntegrator)
    ctx = integ.h.ctx
    ctx.hermite || return nothing
    ctx.fstart === nothing &&
        (ctx.fstart = _derivative(ctx, integ.tdir * integ.tprev, integ.uprev))
    ctx.fend === nothing && (ctx.fend = _derivative(ctx, ctx.end_s, ctx.end_u))
    return nothing
end

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

function _is_event(prev, next, cb::SciMLBase.ContinuousCallback)
    return (
        (prev < 0 && cb.affect! !== nothing) || (prev > 0 && cb.affect_neg! !== nothing)
    ) && prev * next <= 0
end
_is_event(prev, next, ::SciMLBase.VectorContinuousCallback) = prev != 0 && prev * next <= 0

function _event_root(integ::PETScIntegrator, cb, lo, hi, i::Int, buf)
    condition(t, _) = _fill_conditions!(buf, integ, cb, t)[i]
    return DiffEqBase.find_root(condition, (lo, hi), cb.rootfind)
end

# SciMLBase's vector mask: +1 is a crossing from negative to positive.
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
    ev, residual = integ.event_t[k], integ.event_residual[k]
    nudged = [ev[i] == t0 && abs(s0[i] - residual[i]) <= cb.abstol for i in 1:m]
    start = fill(t0, m)
    if any(nudged)
        tn = t0 + (t1 - t0) * oftype(t0, cb.repeat_nudge)
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
        tk = k == n ? t1 : t0 + (t1 - t0) * oftype(t0, k / n)
        _fill_conditions!(sk, integ, cb, tk)
        hit = [i for i in 1:m if past(i, tk) && _is_event(s0[i], sk[i], cb)]
        if !isempty(hit)
            roots = [
                sk[i] == 0 ? tk :
                    _event_root(integ, cb, past(i, lo) ? lo : start[i], tk, i, buf)
                    for i in hit
            ]
            first_root = roots[argmin(integ.tdir .* roots)]
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

function _agreed_event(integ::PETScIntegrator, cb, k::Int)
    comm = integ.h.ctx.comm
    comm === nothing && return _find_event(integ, cb, k)
    m = _ncond(cb)
    found = err = nothing
    try
        found = _find_event(integ, cb, k)
    catch e
        err = e
    end
    mine = zeros(m + 3)
    mine[1] = err !== nothing
    found === nothing || (mine[2] = 1; mine[3] = found[1]; mine[4:end] .= found[2])
    seen = reshape(MPI.Allgather(mine, comm), m + 3, :)
    any(!iszero, view(seen, 1, :)) && throw(something(err, _remote_error()))
    hits = findall(!iszero, view(seen, 2, :))
    isempty(hits) && return nothing
    j = hits[argmin([integ.tdir * seen[3, i] for i in hits])]
    crossing = cb isa SciMLBase.VectorContinuousCallback ? Int8.(seen[4:end, j]) : seen[4, j]
    return (oftype(integ.t, seen[3, j]), crossing)
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

function _rollback!(integ::PETScIntegrator, t, dt, interpolate::Bool)
    t, dt = oftype(integ.t, t), oftype(integ.t, dt)
    h = integ.h
    pl = h.petsclib
    if interpolate && t != integ.t
        copyto!(integ.u, _interpolate!(integ, integ.tdir * t))
        integ.t = t
        _end_step_here!(integ)
    end
    integ.t = t
    h.ctx.pdirty = true
    PETScCompat.with_local_array!(
        ua -> copyto!(ua, integ.u), h.u; read = false, write = true,
    )
    LibPETSc.TSSetTime(pl, h.ts, integ.tdir * t)
    LibPETSc.TSSetTimeStep(pl, h.ts, integ.tdir * dt)
    LibPETSc.TSRestartStep(pl, h.ts)
    integ.dt = integ.t - integ.tprev
    return nothing
end

function _apply_continuous_callbacks!(integ::PETScIntegrator, dt)
    isempty(integ.continuous) && return false
    ctx = integ.h.ctx
    # A rank's own root search must not call `f`, which is collective.
    ctx.comm === nothing || _pin_step!(integ)
    best, best_cb, best_crossing, best_k = nothing, nothing, nothing, 0
    for (k, cb) in enumerate(integ.continuous)
        found = _agreed_event(integ, cb, k)
        found === nothing && continue
        if best === nothing || integ.tdir * found[1] < integ.tdir * best
            best, best_cb, best_crossing, best_k = found[1], cb, found[2], k
        end
    end
    best === nothing && return false
    saved = _save_step!(integ, best, false; slack = zero(best))
    _rollback!(integ, best, dt, true)
    residual = integ.event_residual[best_k]
    best_cb.rootfind === SciMLBase.NoRootFind ? fill!(residual, 0.0) :
        _fill_conditions!(residual, integ, best_cb, integ.t)
    best_cb.save_positions[1] && !saved && _save_here!(integ)
    integ.derivative_discontinuity = true
    _pin_step!(integ)
    _checked_everywhere(() -> _fire!(integ, best_cb, best_crossing), ctx.comm)
    integ.finished && return true
    _rollback!(integ, integ.t, dt, false)
    _mark_fired!(integ.event_t[best_k], best_cb, best_crossing, integ.t)
    best_cb.save_positions[2] && _save_here!(integ)
    return true
end

function _apply_callbacks!(integ::PETScIntegrator, saved::Bool)
    h = integ.h
    ctx = h.ctx
    for cb in integ.callbacks
        integ.finished && return nothing
        _checked_anywhere(() -> cb.condition(integ.u, integ.t, integ), ctx.comm) || continue
        cb.save_positions[1] && !saved && _save_here!(integ)
        saved = false
        integ.derivative_discontinuity = true
        _pin_step!(integ)
        _checked_everywhere(() -> cb.affect!(integ), ctx.comm)
        integ.finished && return nothing
        if _anywhere(ctx.comm, integ.derivative_discontinuity)
            PETScCompat.with_local_array!(
                ua -> copyto!(ua, integ.u), h.u; read = false, write = true,
            )
            LibPETSc.TSRestartStep(h.petsclib, h.ts)
            ctx.pdirty = true
        end
        cb.save_positions[2] && _save_here!(integ)
    end
    return nothing
end

function _initialize_callbacks!(integ::PETScIntegrator, initialize_save::Bool)
    h = integ.h
    cbs = (integ.callbacks..., integ.continuous...)
    before = copy(integ.u)
    _checked_everywhere(h.ctx.comm) do
        for cb in cbs
            cb.initialize(cb, integ.u, integ.t, integ)
        end
    end
    integ.derivative_discontinuity = false
    _everywhere(h.ctx.comm, integ.u == before) && return nothing
    copyto!(integ.uprev, integ.u)
    PETScCompat.with_local_array!(
        ua -> copyto!(ua, integ.u), h.u; read = false, write = true,
    )
    LibPETSc.TSRestartStep(h.petsclib, h.ts)
    initialize_save && any(cb -> cb.save_positions[2], cbs) && _save_here!(integ)
    return nothing
end

function _init_unlocked(
        prob::SupportedProblem, alg::AnyPETScTS;
        callback = nothing, tstops = (), d_discontinuities = (), kwargs...,
    )
    _check_real(tstops, :tstops)
    _check_real(d_discontinuities, :d_discontinuities)
    R = first(_eltypes(prob))
    tstops, d_discontinuities = _times(R, tstops), _times(R, d_discontinuities)
    stops_given = vcat(tstops, d_discontinuities)
    callbacks, continuous = _split_callbacks(callback)
    h = _setup(prob, alg; tstops = stops_given, kwargs...)
    LibPETSc.TSSetUp(h.petsclib, h.ts)
    _match_steps_here!(h)
    _initial_save!(h)
    stops = _tstops(stops_given, h)
    dt0 = h.tdir * LibPETSc.TSGetTimeStep(h.petsclib, h.ts)
    integ = PETScIntegrator(
        alg, copy(h.u0), copy(h.u0), _user_t(h.tdir, h.t0), _user_t(h.tdir, h.t0),
        dt0, h.tdir,
        prob.p, h, prob, callbacks, continuous, prob.f, _make_opts(h, kwargs),
        copy(h.u0), similar(h.u0), similar(h.u0),
        Vector{R}[fill(R(NaN), _ncond(cb)) for cb in continuous],
        Vector{Float64}[fill(NaN, _ncond(cb)) for cb in continuous], NamedTuple(kwargs),
        stops, tstops, d_discontinuities, d_discontinuities, dt0,
        _initial_solution(prob, alg, h), false, false,
    )
    try
        _initialize_callbacks!(integ, true)
    catch
        _destroy!(h)
        rethrow()
    end
    _past_discontinuity!(integ)
    return integ
end

SciMLBase.__init(prob::SupportedProblem, alg::AnyPETScTS; kwargs...) =
    _locked(() -> _init_unlocked(prob, alg; kwargs...))

function _reject_step!(integ::PETScIntegrator, before, taken)
    h = integ.h
    ctx, pl = h.ctx, h.petsclib
    failed, h.stopped = h.stopped != 0, 0
    nstep, integ.dt, integ.dtcache, ctx.pdirty, outer = before
    integ.t = integ.tprev
    copyto!(integ.u, integ.uprev)
    PETScCompat.with_local_array!(
        ua -> copyto!(ua, integ.u), h.u; read = false, write = true,
    )
    LibPETSc.TSSetTime(pl, h.ts, integ.tdir * integ.t)
    LibPETSc.TSSetStepNumber(pl, h.ts, LibPETSc.PetscInt(nstep))
    floor = abs(oftype(integ.t, something(get(integ.kwargs, :dtmin, nothing), 0.0)))
    forced = get(integ.kwargs, :force_dtmin, false) === true
    dt, at_floor = _retry_step(h, integ.tdir * integ.t, taken, floor, forced, failed)
    if dt === nothing
        failed && _warn_failed_step(integ.alg, PETSC_ERR_FP, integ.kwargs)
        if outer !== nothing
            integ.tprev = outer[1]
            copyto!(integ.uprev, outer[2])
        end
        ctx.fstart = nothing
        _finish!(integ)
        return nothing
    end
    LibPETSc.TSSetTimeStep(pl, h.ts, dt)
    LibPETSc.TSRestartStep(pl, h.ts)
    at_floor || return _step_unlocked(integ, outer)
    domain, ctx.domain = ctx.domain, nothing
    try
        return _step_unlocked(integ, outer)
    finally
        ctx.domain = domain
    end
end

# PETSc needs dt_max strictly above dt_min.
_above(hi, lo) = hi > lo ? hi : nextfloat(lo)

function _take_written_state!(integ::PETScIntegrator)
    h = integ.h
    _everywhere(h.ctx.comm, _readvec!(integ.ucache, h.petsclib, h.u) == integ.u) &&
        return nothing
    PETScCompat.with_local_array!(
        ua -> copyto!(ua, integ.u), h.u; read = false, write = true,
    )
    LibPETSc.TSRestartStep(h.petsclib, h.ts)
    h.ctx.pdirty = true
    return nothing
end

_times(R, ts) = ts isa Number ? [R(ts)] : collect(R, ts)

# d_discontinuities are right-continuous, so the next step starts an ulp past one.
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

function _tstops(tstops, h::TSHandles{<:Any, <:Any, R}) where {R}
    stops = sort!(unique!(h.tdir .* _times(R, tstops)))
    filter!(s -> h.t0 < s < h.tf, stops)
    return push!(stops, h.tf)
end

function _add_tstop_unlocked(integ::PETScIntegrator, t)
    t = oftype(integ.t, t)
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
# These return the queue key `tdir * t`, not the caller's time.
SciMLBase.first_tstop(integ::PETScIntegrator) = integ.tstops[1]
SciMLBase.pop_tstop!(integ::PETScIntegrator) = popfirst!(integ.tstops)

function _initial_save!(h::TSHandles)
    ctx = h.ctx
    tol = _near(h.t0)
    landed = false
    while ctx.saveat_idx <= length(ctx.saveat) && ctx.saveat[ctx.saveat_idx] <= h.t0 + tol
        _record_end!(ctx, ctx.saveat[ctx.saveat_idx], h.u0)
        ctx.saveat_idx += 1
        landed = true
    end
    h.save_start && !landed && _record_end!(ctx, h.t0, h.u0)
    return nothing
end

_user_time(h::TSHandles) = h.tdir > 0 ? (h.ctx.ts, h.ctx.dus) :
    (_user_t.(h.tdir, h.ctx.ts), [-d for d in h.ctx.dus])

_nf(h::TSHandles) = h.ctx.nf + (h.ad_calls === nothing ? 0 : h.ad_calls[])

function _initial_solution(prob, alg, h::TSHandles)
    ts, dus = _user_time(h)
    # The running `sol` shares these arrays, and `_record!` extends a reversed span's copies.
    h.tdir > 0 || ((h.ctx.user_ts, h.ctx.user_dus) = (ts, dus))
    return SciMLBase.build_solution(
        prob, alg, ts, h.ctx.us; retcode = SciMLBase.ReturnCode.Default,
        dense = h.ctx.dense, interp = _interp(h.ctx, ts, dus), stats = SciMLBase.DEStats(0),
    )
end

function _live_stats!(integ::PETScIntegrator)
    stats = integ.sol.stats
    stats isa SciMLBase.DEStats || return nothing
    ctx, st = integ.h.ctx, _read_stats(integ.h)
    stats.nf, stats.nf2, stats.njacs = _nf(integ.h), ctx.nf2, ctx.njacs
    stats.nnonliniter, stats.nnonlinconvfail = st.nnonliniter, st.nnonlinfail
    stats.naccept, stats.nreject = st.nsteps, st.nreject
    return nothing
end

function _reinit_unlocked(
        integ::PETScIntegrator, u0 = integ.prob.u0;
        t0 = integ.prob.tspan[1], tf = integ.prob.tspan[2],
        erase_sol = true, saveat = nothing, tstops = integ.tstops_cache,
        d_discontinuities = integ.d_discontinuities_cache,
        reinit_callbacks = true, initialize_save = true,
    )
    _check_real(tstops, :tstops)
    _check_real(d_discontinuities, :d_discontinuities)
    R = typeof(integ.t)
    tstops, d_discontinuities = _times(R, tstops), _times(R, d_discontinuities)
    old = integ.h
    prob = SciMLBase.remake(
        integ.prob; u0 = _retype(integ.prob.u0, u0), tspan = _retype(integ.prob.tspan, (t0, tf)),
        p = integ.p,
    )
    setup_kwargs = saveat === nothing ? integ.kwargs : merge(integ.kwargs, (saveat = saveat,))
    h = _setup(prob, integ.alg; tstops = vcat(tstops, d_discontinuities), setup_kwargs...)
    LibPETSc.TSSetUp(h.petsclib, h.ts)
    _match_steps_here!(h)
    if !erase_sol
        append!(h.ctx.ts, old.ctx.ts)
        append!(h.ctx.us, old.ctx.us)
        if h.ctx.dense
            if old.ctx.dense
                append!(h.ctx.dus, old.ctx.dus)
            else
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
    integ.dt = h.tdir * LibPETSc.TSGetTimeStep(h.petsclib, h.ts)
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

_retype(old, new) = eltype(old) <: Union{AbstractFloat, Complex} ? eltype(old).(new) : new

SciMLBase.reinit!(integ::PETScIntegrator, u0 = integ.prob.u0; kwargs...) =
    _locked(() -> _reinit_unlocked(integ, u0; kwargs...))

@static if isdefined(SciMLBase, :has_reinit)
    SciMLBase.has_reinit(::PETScIntegrator) = true
end

function _finish!(integ::PETScIntegrator, retcode = nothing)
    integ.finished && return nothing
    h = integ.h
    _checked_everywhere(h.ctx.comm) do
        for cb in (integ.callbacks..., integ.continuous...)
            cb.finalize(cb, integ.u, integ.t, integ)
        end
    end
    st = _read_stats(h)
    sol = _assemble(
        integ.prob, integ.alg, h, integ.tdir * integ.t, copy(integ.u), st, integ.kwargs,
    )
    integ.sol = retcode === nothing ? sol : SciMLBase.solution_new_retcode(sol, retcode)
    integ.finished = true
    _destroy!(h)
    h.ctx.comm === nothing || _throw_if_threw!(h.ctx)
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

function _save_step!(integ::PETScIntegrator, upto, endpoint::Bool; slack = _near(upto))
    h = integ.h
    ctx = h.ctx
    tol = _near(integ.t)
    n = length(ctx.ts)
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
    return length(ctx.ts) > n && _last_recorded(ctx, integ.tdir * upto)
end

# MATCHSTEP refuses a step leaving under 10 eps (1.2e-6 in Float32) before a stop.
function _match_steps_here!(h::TSHandles{<:Any, <:Any, R}) where {R}
    pl, ts = h.petsclib, h.ts
    R === Float32 && _exact_final_time(pl, ts) == LibPETSc.TS_EXACTFINALTIME_MATCHSTEP ||
        return nothing
    LibPETSc.TSSetExactFinalTime(pl, ts, LibPETSc.TS_EXACTFINALTIME_STEPOVER)
    h.matches = true
    h.fixed = LibPETSc.TSAdaptGetType(pl, LibPETSc.TSGetAdapt(pl, ts)) == "none"
    return nothing
end

# PETSc's MATCHSTEP rule, so no sliver is left before the stop.
function _match_step!(integ::PETScIntegrator, remaining)
    h = integ.h
    pl, s = h.petsclib, integ.tdir * integ.t
    p = LibPETSc.TSGetTimeStep(pl, h.ts)
    h.fixed || (integ.dtcache = integ.tdir * p)
    q = p * oftype(p, 1.01) > remaining ? remaining : 2p > remaining ? remaining / 2 : p
    h.fixed || (q = max(q, min(remaining, _min_step(s))))
    q == p || LibPETSc.TSSetTimeStep(pl, h.ts, q)
    return nothing
end

_resumed_step(integ::PETScIntegrator, stop) = integ.h.matches && !integ.h.fixed ?
    max(integ.tdir * integ.dtcache, _min_step(stop)) : integ.tdir * integ.dtcache

function _step_unlocked(integ::PETScIntegrator, outer = nothing)
    integ.finished && throw(
        ArgumentError(
            "this integrator has finished at t = $(integ.t) and cannot step further; " *
                "call reinit! to restart it",
        ),
    )
    h = integ.h
    ctx, pl = h.ctx, h.petsclib
    if Int(LibPETSc.TSGetStepNumber(pl, h.ts)) >= h.maxiters
        _finish!(integ)
        return nothing
    end
    outer === nothing && _take_written_state!(integ)
    dtprev = integ.tdir * LibPETSc.TSGetTimeStep(pl, h.ts)
    before = (
        Int(LibPETSc.TSGetStepNumber(pl, h.ts)), integ.dt, integ.dtcache, ctx.pdirty,
        outer === nothing && ctx.domain !== nothing ? (integ.tprev, copy(integ.uprev)) : outer,
    )
    copyto!(integ.uprev, integ.u)
    integ.tprev = integ.t
    reuse = ctx.hermite && integ.tdir * integ.t == ctx.end_s &&
        _everywhere(ctx.comm, integ.u == ctx.end_u && !ctx.pdirty)
    ctx.fstart = reuse ? ctx.fend : nothing
    ctx.pdirty = false
    tol = _near(h.tf)
    while !isempty(integ.tstops) && integ.tstops[1] <= integ.tdir * integ.t + _near(integ.t)
        popfirst!(integ.tstops)
    end
    # PETSc keeps the step shortened onto its max time, so `dtcache` holds the uncut one.
    stop = !isempty(integ.tstops) && integ.tstops[1] < h.tf - tol ? integ.tstops[1] : nothing
    target = stop === nothing ? h.tf : stop
    LibPETSc.TSSetMaxTime(pl, h.ts, target)
    if h.matches
        _match_step!(integ, target - integ.tdir * integ.t)
    elseif LibPETSc.TSGetTimeStep(pl, h.ts) > target - integ.tdir * integ.t
        # TSAdaptChoose rejects a step past the max time, so it is shortened here.
        LibPETSc.TSSetTimeStep(pl, h.ts, target - integ.tdir * integ.t)
    end
    h.stopped = 0
    ctx.retry_fp && ctx.workvec == C_NULL && _hold_work_vec!(ctx, h.ts)
    failure = _run!(h) do
        LibPETSc.TSStep(pl, h.ts)
    end
    if _threw!(ctx)
        err = ctx.err
        _finish!(integ)
        throw(err)
    end
    failure === nothing || throw(failure)
    if h.stopped == PETSC_ERR_FP && SciMLBase.isadaptive(integ) && _return_work_vec!(ctx, h.ts)
        return _reject_step!(integ, before, LibPETSc.TSGetTimeStep(pl, h.ts))
    end
    h.stopped == 0 || _warn_failed_step(integ.alg, h.stopped, integ.kwargs)
    integ.t = _user_t(integ.tdir, LibPETSc.TSGetTime(pl, h.ts))
    if integ.tdir * integ.t <= integ.tdir * integ.tprev
        if h.stopped == 0 && SciMLBase.isadaptive(integ) &&
                Int(LibPETSc.TSGetConvergedReason(pl, h.ts)) == 0
            ctx.unstable_hit = true
            _verbose(integ.kwargs) && @warn "`$(_warn_name(integ.alg))` ends here because " *
                "its step fell below the floating point spacing at t = $(integ.t)"
        end
        _finish!(integ)
        return nothing
    end
    if stop === nothing
        h.matches || (integ.dtcache = integ.tdir * LibPETSc.TSGetTimeStep(pl, h.ts))
        if integ.tdir * integ.t != h.tf && integ.tdir * integ.t >= h.tf - tol
            integ.t = _user_t(integ.tdir, h.tf)
            LibPETSc.TSSetTime(pl, h.ts, h.tf)
        end
    elseif integ.tdir * integ.t >= stop - _near(stop)
        integ.t = _user_t(integ.tdir, stop)
        LibPETSc.TSSetTime(pl, h.ts, stop)
        LibPETSc.TSSetTimeStep(pl, h.ts, _resumed_step(integ, stop))
    end
    integ.dt = integ.t - integ.tprev
    _readvec!(integ.u, pl, h.u)
    if ctx.domain !== nothing && SciMLBase.isadaptive(integ) &&
            _predicate(() -> ctx.domain(integ.u, ctx.p, integ.t), ctx)
        return _reject_step!(integ, before, abs(integ.t - integ.tprev))
    end
    _end_step_here!(integ)
    _live_stats!(integ)
    fired = _apply_continuous_callbacks!(integ, dtprev)
    integ.finished && return nothing
    saved = !fired && _save_step!(integ, integ.t, true)
    # Discrete callbacks run with the stop they landed on still at the head of the queue.
    _apply_callbacks!(integ, saved)
    while !isempty(integ.tstops) && integ.tstops[1] <= integ.tdir * integ.t + _near(integ.t)
        popfirst!(integ.tstops)
    end
    integ.finished && return nothing
    if integ.tdir * integ.t < h.tf - tol
        hnext = LibPETSc.TSGetTimeStep(pl, h.ts)
        limit = isempty(integ.tstops) ? h.tf : min(integ.tstops[1], h.tf)
        if ctx.dtmin > 0 && hnext < ctx.dtmin && integ.tdir * integ.t + hnext < limit - tol
            ctx.dt_too_small = true
        elseif ctx.unstable !== nothing &&
                _predicate(() -> ctx.unstable(integ.tdir * hnext, integ.u, ctx.p, integ.t), ctx)
            ctx.unstable_hit = true
        end
    end
    if !_everywhere(ctx.comm, all(isfinite, integ.u)) || integ.tdir * integ.t >= h.tf - tol ||
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
