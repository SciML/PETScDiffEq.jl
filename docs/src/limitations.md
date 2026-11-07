# Limitations

A `DynamicalODEProblem` or `SecondOrderODEProblem` runs distributed over a `comm`, in a solve
and under `PETScAdjoint`, but not with a `dm`. PETSc TS is built for large
distributed problems, and reaching it from the SciML interface is what this package is for;
use OrdinaryDiffEq.jl for serial problems where it applies.

An array-shaped `u0` is taken for an `ODEProblem` and a `SplitODEProblem` on
`MPI.COMM_SELF`. A `DAEProblem`, a solve on another `comm` or with a `dm`, and `PETScAdjoint`
need the state as a vector and raise an `ArgumentError` that says so: pass `vec(u0)` and
reshape it inside the problem's functions. A `u0` of another vector type, such as a
`ComponentArray`, keeps its type wherever an array-shaped one is taken, `PETScAdjoint`
included, and is stepped and given back as a `Vector` in a `DAEProblem` or a solve on
another `comm` or with a `dm`. A `DynamicalODEProblem` or `SecondOrderODEProblem` takes
vectors for its velocity and position.

On 32-bit Julia, use Julia 1.10, or add `PETSc_jll = "~3.22"` to your own compat: PETSc_jll
3.25 has no 32-bit builds, and newer Julia versions would otherwise resolve it.

On Windows, use Julia 1.12 or newer. Julia 1.10 and 1.11 resolve PETSc_jll 3.22, whose
Windows build has no MPI, and PETSc.jl fails to create any PETSc object with it, so there
every solve raises an `ArgumentError` that says so.

On 32-bit x86, PETSc 3.22 fails a consistency check of its own when it shortens the last
steps to land on a stop or on the final time, which ended long spans in `PetscError(77)`.
There this package has PETSc step over the final time and applies PETSc's rule for those
steps itself, so `solve` and the integrator take the steps they would otherwise. The
forward solve inside `PETScAdjoint` is still landed by PETSc, so a gradient over a long
span can end in that error on 32-bit x86.

Solves from several threads, such as an `EnsembleThreads` ensemble, are safe but run one
at a time: PETSc's options and MPI are shared by the whole process. Distributed ones are not,
as the [MPI](@ref) section says.

Finish or terminate every integrator you start. A serial one dropped part way is released
by a finalizer, and if that finalizer runs at process exit, after MPI has shut down, PETSc's
own object finalizers print an MPI warning and the process exits non-zero. One on a
communicator of more than one rank has no finalizer, since freeing it is collective and the
ranks' garbage collectors would not free it together. Dropped, it keeps its TS, vectors and
matrices until the process exits, when every rank frees them in the order they were made, so
run a distributed integrator to its end or call `terminate!` on it on every rank: a loop
that drops them holds every one until exit.
