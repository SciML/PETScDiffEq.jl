using SciMLTesting, PETScDiffEq, Test

# PETSc.jl declares no names public, so every call into it is a non-public access.
run_qa(PETScDiffEq; ei_broken = (:all_qualified_accesses_are_public,))
