using PETScDiffEq
using Test

@testset "PETScDiffEq.jl" begin
    @test isdefined(PETScDiffEq, :PETScDiffEq)
end
