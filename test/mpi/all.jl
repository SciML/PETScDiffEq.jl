using MPI, Test
# Error messages print a type unqualified only when Main sees it, as it does running one script.
using PETScDiffEq: AutoForwardDiff

MPI.Init()
failed = String[]
for script in ARGS
    try
        Base.include(Module(Symbol(script)), joinpath(@__DIR__, script))
    catch err
        err isa LoadError && err.error isa Test.TestSetException || rethrow()
        push!(failed, script)
    end
end
isempty(failed) || error("tests failed in ", join(failed, ", "))
