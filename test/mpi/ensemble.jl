using MPI, PETScDiffEq, SciMLBase, Test
using PETScDiffEq: PETSc
using SciMLBase: ODEProblem, EnsembleProblem, EnsembleSerial, EnsembleThreads, remake, solve

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
PETSc.initialize(PETSc.getlib(; PetscScalar = Float64))

const N = 23
const counts = let c = floor.(Int, N .* (1:nranks) ./ sum(1:nranks))
    c[end] += N - sum(c)
    c
end
const rows = (lo = sum(counts[1:rank]) + 1; lo:(lo + counts[rank + 1] - 1))

function gathered(u)
    out = rank == 0 ? zeros(N) : nothing
    MPI.Gatherv!(u, rank == 0 ? MPI.VBuffer(out, counts) : nothing, comm)
    return out
end

function halo(u)
    left = rank == 0 ? MPI.PROC_NULL : rank - 1
    right = rank == nranks - 1 ? MPI.PROC_NULL : rank + 1
    gl, gr = zeros(1), zeros(1)
    MPI.Sendrecv!(u[1:1], gr, comm; dest = left, source = right)
    MPI.Sendrecv!(u[end:end], gl, comm; dest = right, source = left)
    return gl[1], gr[1]
end

const dx = 1 / (N + 1)
function laplacian!(du, u, left, right)
    n = length(u)
    for i in 1:n
        l = i == 1 ? left : u[i - 1]
        r = i == n ? right : u[i + 1]
        du[i] = (l - 2u[i] + r) / dx^2
    end
    return nothing
end
heat!(du, u, p, t) = laplacian!(du, u, halo(u)...)
heat_serial!(du, u, p, t) = laplacian!(du, u, 0.0, 0.0)
heat0(idx) = sinpi.(idx .* dx)
spread(prob, i) = remake(prob; u0 = i .* prob.u0, tspan = (0.0, 0.01i))
spread(prob, ctx::SciMLBase.EnsembleContext) = spread(prob, ctx.sim_id)
ensemble(f, idx) = EnsembleProblem(ODEProblem(f, heat0(idx), (0.0, 0.01)); prob_func = spread)
const FIXED = (; dt = 1.0e-4, adaptive = false, trajectories = 4)

function caught(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

@testset "MPI ensembles, $nranks ranks, $(Threads.nthreads()) threads" begin
    sim = solve(ensemble(heat!, rows), TSRK("5dp"; comm), EnsembleSerial(); FIXED...)
    ends = [gathered(s.u[end]) for s in sim.u]
    if rank == 0
        ref = solve(ensemble(heat_serial!, 1:N), TSRK("5dp"), EnsembleThreads(); FIXED...)
        @test [s.t for s in sim.u] == [s.t for s in ref.u]
        @test maximum(maximum(abs, a - s.u[end]) for (a, s) in zip(ends, ref.u)) <= 1.0e-14
    end
    e = caught(() -> solve(ensemble(heat!, rows), TSRK("5dp"; comm), EnsembleThreads(); FIXED...))
    @test e !== nothing && occursin("EnsembleSerial", sprint(showerror, e))
    heat_solve() =
        solve(ODEProblem(heat!, heat0(rows), (0.0, 0.01)), TSRK("5dp"; comm); dt = 1.0e-3)
    looped() = Threads.@threads for i in 1:2
        heat_solve()
    end
    e = caught(looped)
    @test e !== nothing && occursin("Threads.@threads", sprint(showerror, e))
    function one_rank()
        rank == 0 || return heat_solve()
        Threads.@threads for i in 1:1
            heat_solve()
        end
    end
    e = caught(one_rank)
    @test e !== nothing && occursin("Threads.@threads", sprint(showerror, e))

    started, release = Threads.Atomic{Int}(0), Base.Event()
    elsewhere = Threads.@spawn Threads.@threads for i in 1:2
        Threads.atomic_add!(started, 1)
        wait(release)
    end
    while started[] == 0
        yield()
    end
    e = caught(heat_solve)
    notify(release)
    wait(elsewhere)
    @test e === nothing
    @test isempty(PETScDiffEq.PARALLEL_HANDLES)
end
