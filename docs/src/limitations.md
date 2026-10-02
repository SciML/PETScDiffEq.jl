# Limitations

A `DynamicalODEProblem` or `SecondOrderODEProblem` does not run distributed yet, whatever the
algorithm, so `TSBasicSymplectic` and `TSAlpha2` run on `MPI.COMM_SELF` only. PETSc TS is
built for large distributed problems, and reaching it from the SciML interface is what this
package is for; use OrdinaryDiffEq.jl for serial problems where it applies.

On 32-bit Julia, use Julia 1.10, or add `PETSc_jll = "~3.22"` to your own compat: PETSc_jll
3.25 has no 32-bit builds, and newer Julia versions would otherwise resolve it.

Solves from several threads, such as an `EnsembleThreads` ensemble, are safe but run one
at a time: PETSc's options and MPI are shared by the whole process. Distributed ones are not,
as the [MPI](@ref) section says.

Finish or terminate every integrator you start. One dropped part way is released by a
finalizer, and if that finalizer runs at process exit, after MPI has shut down, PETSc's
own object finalizers print an MPI warning and the process exits non-zero.
