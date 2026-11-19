module PETScDiffEqStaticArraysExt

import PETScDiffEq
using StaticArrays: SArray, similar_type

# An SArray can't be filled in place, so a state in its type is rebuilt from PETSc's flat vector.
function PETScDiffEq._partition(like::SArray{S, T0, N, L}, x, T = eltype(x)) where {S, T0, N, L}
    return similar_type(typeof(like), T)(ntuple(i -> x[i], Val(L)))
end
PETScDiffEq._buffer(like::SArray) = like
PETScDiffEq._load!(buf::SArray, x) = PETScDiffEq._partition(buf, x)
PETScDiffEq._input(buf::SArray, x, same) = PETScDiffEq._partition(buf, x)
PETScDiffEq._state_type(like::SArray, U) = similar_type(typeof(like), U)
PETScDiffEq._static(::SArray) = true
PETScDiffEq._user_state(like::SArray, u) = PETScDiffEq._partition(like, u)

end
