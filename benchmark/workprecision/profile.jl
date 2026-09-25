ENV["PETSC_OPTIONS"] = "-log_view"

include(joinpath(@__DIR__, "workprecision.jl"))
using Libdl

const LIB = Libdl.dlopen(PETScDiffEq._petsclib(Float64).petsc_library)

function stage(name)
    s = Ref{Cint}(0)
    ccall(Libdl.dlsym(LIB, :PetscLogStageRegister), Cint, (Cstring, Ptr{Cint}), name, s)
    return s[]
end

function profile(label, prob, alg, ref; runs = 5)
    kw = (; abstol = 1.0e-8, reltol = 1.0e-5, save_everystep = false)
    sol = solve(prob, alg; kw...)
    s = stage(label)
    ccall(Libdl.dlsym(LIB, :PetscLogStagePush), Cint, (Cint,), s)
    t = minimum(@elapsed(solve(prob, alg; kw...)) for _ in 1:runs)
    ccall(Libdl.dlsym(LIB, :PetscLogStagePop), Cint, ())
    @printf(
        "%-24s fastest of %d runs %.4e s, error %.2e, %d steps, %d Jacobians, %d Newton iterations\n",
        label, runs, t, sum(abs, sol.u[end] - ref.u[end]) / length(ref.u[end]),
        sol.stats.naccept, sol.stats.njacs, sol.stats.nnonliniter,
    )
    return nothing
end

let hires = hires_problem(), bruss = brusselator_problem(500)
    hires_ref = reference(hires, Rodas5P(); tol = 1.0e-14)
    bruss_ref = reference(bruss, Rodas5P(); tol = 1.0e-12)
    lu = ["-ksp_type", "preonly", "-pc_type", "lu"]
    lag = [lu; "-snes_lag_jacobian"; "10"; "-snes_lag_jacobian_persists"; "true"]
    profile("hires-rosw-ra34pw2", hires, TSRosW("ra34pw2"), hires_ref)
    profile("hires-bdf5", hires, TSImplicit("bdf"; order = 5), hires_ref)
    profile("bruss-rosw-ra34pw2", bruss, TSRosW("ra34pw2"), bruss_ref)
    profile("bruss-rosw-ra34pw2-lu", bruss, TSRosW("ra34pw2", lu), bruss_ref)
    profile("bruss-bdf5", bruss, TSImplicit("bdf"; order = 5), bruss_ref)
    profile("bruss-bdf5-lu", bruss, TSImplicit("bdf", lu; order = 5), bruss_ref)
    profile("bruss-bdf5-lu-lag", bruss, TSImplicit("bdf", lag; order = 5), bruss_ref)
    profile("bruss-arkimex4-lu", bruss, TSARKIMEX("4", lu), bruss_ref)
    profile("bruss-arkimex4-lu-lag", bruss, TSARKIMEX("4", lag), bruss_ref)
end
