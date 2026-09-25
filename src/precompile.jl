using PrecompileTools: @compile_workload, @setup_workload

function _wl_decay!(du, u, p, t)
    for i in eachindex(u)
        du[i] = -u[i] + (i > 1 ? u[i - 1] : zero(eltype(u)))
    end
    return nothing
end

function _wl_stiff!(du, u, p, t)
    du[1] = -p[1] * u[1] + p[2] * u[2]
    du[2] = p[1] * u[1] - p[2] * u[2] - u[2]^2
    return nothing
end

_wl_cost!(out, u, p, t, i) = (out .= u; nothing)

# Under mpiexec or srun, MPI_Init in the precompiling process aborts it.
_under_mpi_launcher() = any(
    k -> startswith(k, "PMI_") || startswith(k, "PMIX_") || startswith(k, "OMPI_COMM_WORLD_"),
    keys(ENV),
)

function _run_workload()
    _under_mpi_launcher() && return nothing
    builds = _loaded_builds()
    if Float64 in builds
        u0, tspan = [1.0, 0.5], (0.0, 1.0)
        prob = SciMLBase.ODEProblem(_wl_decay!, u0, tspan)
        pprob = SciMLBase.ODEProblem(_wl_stiff!, u0, tspan, [2.0, 1.0])
        SciMLBase.solve(prob, TSRK())
        SciMLBase.solve(prob, TSRK(); abstol = 1.0e-8, reltol = 1.0e-8)
        SciMLBase.solve(prob, TSRK("4"); dt = 0.1, saveat = 0.5)
        SciMLBase.solve(pprob, TSRK())
        SciMLBase.solve(pprob, TSRosW())
        SciMLBase.solve(pprob, TSImplicit("bdf"))
        sparse_f = SciMLBase.ODEFunction(
            _wl_decay!; jac_prototype = SparseArrays.sparse([1, 2, 2], [1, 1, 2], ones(3)),
        )
        SciMLBase.solve(SciMLBase.ODEProblem(sparse_f, u0, tspan), TSImplicit("bdf"))
        integ = SciMLBase.init(prob, TSRK())
        SciMLBase.step!(integ)
        integ((integ.tprev + integ.t) / 2)
        SciMLBase.solve!(integ)
        _discrete_adjoint(
            pprob, TSRK("4"), PETScAdjoint(); t = [0.0, 1.0], dgdu_discrete = _wl_cost!,
            dt = 0.1, adaptive = false,
        )
    end
    Float32 in builds && SciMLBase.solve(
        SciMLBase.ODEProblem(_wl_decay!, Float32[1, 0.5], (0.0f0, 1.0f0)), TSRK();
        dt = 0.1f0,
    )
    ComplexF64 in builds &&
        SciMLBase.solve(SciMLBase.ODEProblem(_wl_decay!, ComplexF64[1, 0.5im], (0.0, 1.0)), TSRK())
    return nothing
end

@setup_workload begin
    @compile_workload begin
        __init__()
        try
            _run_workload()
        catch
            # A PETSc that cannot run here must leave the package loadable.
        finally
            for pl in PETSc.petsclibs
                PETScCompat.isinitialized(pl) && !PETScCompat.isfinalized(pl) &&
                    PETSc.finalize(pl)
            end
        end
    end
    # The workload's pointers and handles belong to the precompiling process.
    empty!(CALLBACKS)
    empty!(PETSC_SYMBOLS)
    empty!(EXIT_CLEANUP_ARMED)
end
