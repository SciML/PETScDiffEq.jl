using SciMLTesting, PETScDiffEq, Test

# PETSc_jll is a dep only to cap versions PETSc.jl allows but cannot load.
run_qa(
    PETScDiffEq; ei_broken = (:all_qualified_accesses_are_public,),
    aqua_kwargs = (stale_deps = (ignore = [:PETSc_jll],),),
)
