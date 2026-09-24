# The PETSc.jl calls that differ between 0.4 and 0.5, under their 0.5 names.
module PETScCompat

using PETSc: PETSc
using PETSc.LibPETSc: LibPETSc

const V05 = pkgversion(PETSc) >= v"0.5.0-"

@static if V05
    const with_local_array! = PETSc.with_local_array!
    const PetscVec = PETSc.PetscVec
    const PetscMat = PETSc.PetscMat
    const PetscOptions = PETSc.PetscOptions
    const TSRKSetType = LibPETSc.TSRKSetType
    const TSRosWSetType = LibPETSc.TSRosWSetType
    const TSIRKSetType = LibPETSc.TSIRKSetType
    const TSARKIMEXSetType = LibPETSc.TSARKIMEXSetType
else
    # A development checkout can carry a version number older than its API.
    let pl = first(PETSc.petsclibs)
        hasmethod(
            LibPETSc.TSRKSetType, Tuple{typeof(pl), LibPETSc.TS{typeof(pl)}, Ptr{Cchar}},
        ) || error("PETSc.jl $(pkgversion(PETSc)) lacks the 0.4 API this branch uses")
    end

    const with_local_array! = PETSc.withlocalarray!
    PetscVec(pl, x) = PETSc.VecSeq(pl, x)
    PetscMat(pl, A::Matrix) = PETSc.MatSeqDense(pl, A)
    PetscMat(pl, m::Integer, n::Integer, nnz::Integer) = PETSc.MatSeqAIJ(pl, m, n, nnz)
    function PetscMat(pl, comm, S; with_arrays::Bool = false)
        with_arrays ||
            throw(ArgumentError("with PETSc.jl 0.4 only `with_arrays = true` is supported"))
        return PETSc.MatSeqAIJWithArrays(pl, comm, S)
    end
    PetscOptions(pl; kw...) = PETSc.Options(pl; kw...)

    function _cstr(f::F, s::AbstractString) where {F}
        str = String(s)
        return GC.@preserve str f(Base.unsafe_convert(Ptr{Cchar}, str))
    end
    TSRKSetType(pl, ts, s) = _cstr(p -> LibPETSc.TSRKSetType(pl, ts, p), s)
    TSRosWSetType(pl, ts, s) = _cstr(p -> LibPETSc.TSRosWSetType(pl, ts, p), s)
    TSIRKSetType(pl, ts, s) = _cstr(p -> LibPETSc.TSIRKSetType(pl, ts, p), s)
    TSARKIMEXSetType(pl, ts, s) = _cstr(p -> LibPETSc.TSARKIMEXSetType(pl, ts, p), s)
end

end
