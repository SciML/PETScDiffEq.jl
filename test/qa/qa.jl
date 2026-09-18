using SciMLTesting, PETScDiffEq, Test

# PETSc.jl declares no names public, so every call into it is a non-public access.
# PETSc_jll is a dependency only to bound its version, since PETSc.jl allows releases it
# cannot load.
run_qa(
    PETScDiffEq; ei_broken = (:all_qualified_accesses_are_public,),
    aqua_kwargs = (stale_deps = (ignore = [:PETSc_jll],),),
)
