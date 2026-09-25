using MPI, PETScDiffEq, SciMLBase, SparseArrays, Test
using PETScDiffEq: PETSc
using SciMLBase: ODEProblem, ODEFunction, DiscreteCallback, ContinuousCallback, CallbackSet,
    ReturnCode, init, solve, step!

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nranks = MPI.Comm_size(comm)
# PetscInitialize is collective, and the serial reference solves run on single ranks.
for S in (Float64, Float32, ComplexF64, ComplexF32)
    PETSc.initialize(PETSc.getlib(; PetscScalar = S))
end

const N = 23
const thrower = nranks - 1

const counts = let c = floor.(Int, N .* (1:nranks) ./ sum(1:nranks))
    c[end] += N - sum(c)
    c
end
const rows = (lo = sum(counts[1:rank]) + 1; lo:(lo + counts[rank + 1] - 1))

function gathered(u::AbstractVector{<:Number})
    out = rank == 0 ? zeros(eltype(u), N) : nothing
    MPI.Gatherv!(u, rank == 0 ? MPI.VBuffer(out, counts) : nothing, comm)
    return out
end
gathered(sol::SciMLBase.AbstractODESolution) = [gathered(u) for u in sol.u]

same_everywhere(x) = MPI.bcast(x, 0, comm) == x
anywhere(b) = MPI.Allreduce(b, |, comm)
maxdiff(a, b) = maximum(maximum(abs, x - y) for (x, y) in zip(a, b))

function caught(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

remote(e) = e isa ErrorException && occursin("another rank", e.msg)
raised(e, what) = rank == thrower ? e isa ErrorException && occursin(what, e.msg) : remote(e)

const dx = 1 / (N + 1)
scale(S) = S <: Complex ? S(1 + 0.5im) : one(S)
heat0(S, idx) = S.(scale(S) .* (sinpi.(idx .* dx) .+ 0.5 .* sinpi.(3 .* idx .* dx)))
eigval(k) = -4 / dx^2 * sinpi(k * dx / 2)^2
heat_exact(S, t) = scale(S) .* (
    exp(eigval(1) * t) .* sinpi.((1:N) .* dx) .+
        0.5 * exp(eigval(3) * t) .* sinpi.(3 .* (1:N) .* dx)
)

function laplacian!(du, u, left, right)
    h2 = real(eltype(u))(dx)^2
    n = length(u)
    for i in 1:n
        l = i == 1 ? left : u[i - 1]
        r = i == n ? right : u[i + 1]
        du[i] = (l - 2u[i] + r) / h2
    end
    return nothing
end

function halo(u)
    left = rank == 0 ? MPI.PROC_NULL : rank - 1
    right = rank == nranks - 1 ? MPI.PROC_NULL : rank + 1
    gl, gr = zeros(eltype(u), 1), zeros(eltype(u), 1)
    MPI.Sendrecv!(u[1:1], gr, comm; dest = left, source = right)
    MPI.Sendrecv!(u[end:end], gl, comm; dest = right, source = left)
    return gl[1], gr[1]
end

heat!(du, u, p, t) = laplacian!(du, u, halo(u)...)
heat_serial!(du, u, p, t) = laplacian!(du, u, zero(eltype(u)), zero(eltype(u)))

neighbours(i) = max(1, i - 1):min(N, i + 1)

function heat_proto(S, idx)
    I = [k for (k, i) in enumerate(idx) for _ in neighbours(i)]
    J = [j for i in idx for j in neighbours(i)]
    return sparse(I, J, ones(S, length(I)), length(idx), N)
end

heat_jac(idx) = function (J, u, p, t)
    h2 = real(eltype(J))(dx)^2
    for (k, i) in enumerate(idx), j in neighbours(i)
        J[k, j] = (i == j ? -2 : 1) / h2
    end
    return nothing
end

parallel(alg) = alg.comm != MPI.COMM_SELF

function heat(S, idx, alg; jac = nothing, span = (0, 0.1))
    f = parallel(alg) ? heat! : heat_serial!
    R = real(S)
    fn = jac === nothing ? f : jac ?
        ODEFunction(f; jac = heat_jac(idx), jac_prototype = heat_proto(S, idx)) :
        ODEFunction(f; jac_prototype = heat_proto(S, idx))
    return ODEProblem(fn, heat0(S, idx), R.(span))
end

function against_serial(run, alg, serial_alg)
    got = run(rows, alg)
    return got, rank == 0 ? run(1:N, serial_alg) : got
end

const TYPES = (Float32, ComplexF64, ComplexF32)
const BOUNDS = Dict(
    Float32 => (serial = 5.0e-5, exact = 1.0e-4),
    ComplexF64 => (serial = 5.0e-8, exact = 1.0e-4),
    ComplexF32 => (serial = 2.0e-5, exact = 1.0e-4),
)

@testset "MPI number types, $nranks ranks" begin
    @testset "$S" for S in TYPES
        R = real(S)
        tol = (abstol = R(1.0e-5), reltol = R(1.0e-5))
        bound = BOUNDS[S]
        fd = PETScDiffEq.AutoFiniteDiff()
        methods = (
            (TSRK("5dp"; comm), TSRK("5dp"), nothing),
            (TSImplicit("bdf"; comm), TSImplicit("bdf"; autodiff = fd), true),
            (TSImplicit("bdf"; comm), TSImplicit("bdf"; autodiff = fd), false),
            (TSRosW(; comm), TSRosW(; autodiff = fd), true),
        )

        @testset "adaptive solves" begin
            for (alg, serial_alg, jac) in methods
                sol, ref = against_serial(alg, serial_alg) do idx, a
                    solve(heat(S, idx, a; jac), a; tol...)
                end
                @test sol.retcode == ReturnCode.Success
                @test same_everywhere(sol.t)
                @test eltype(sol.t) === R && eltype(sol.u[end]) === S
                u = gathered(sol.u[end])
                if rank == 0
                    @test maximum(abs, u - ref.u[end]) <= bound.serial
                    @test maximum(abs, u - heat_exact(S, 0.1)) <= bound.exact
                end
            end
        end

        @testset "saveat, vector tolerances and a fixed step" begin
            vector_tol(n) = (; abstol = fill(R(1.0e-5), n), reltol = fill(R(1.0e-5), n))
            cases = (
                ("saveat", n -> (; saveat = R(0.01), tol...)),
                ("vector tolerances", vector_tol),
                ("fixed step", n -> (; dt = R(1.0e-3), adaptive = false)),
            )
            for (what, kw) in cases
                sol, ref = against_serial(TSRK("5dp"; comm), TSRK("5dp")) do idx, a
                    solve(heat(S, idx, a), a; kw(length(idx))...)
                end
                @test sol.retcode == ReturnCode.Success
                @test same_everywhere(sol.t)
                us = gathered(sol)
                rank == 0 || continue
                what == "vector tolerances" || @test sol.t == ref.t
                @test maximum(abs, us[end] - ref.u[end]) <= bound.serial
                what == "vector tolerances" || @test maxdiff(us, ref.u) <= bound.serial
            end
        end

        @testset "callbacks and the integrator" begin
            last_row(idx) = findfirst(==(N), idx)
            crossing(idx) =
                (u, t, i) -> (j = last_row(idx); j === nothing ? 1.0 : real(u[j]) - 0.15)
            got, ref = against_serial(TSRK("5dp"; comm), TSRK("5dp")) do idx, a
                n = Ref(0)
                halve = DiscreteCallback((u, t, i) -> t == R(0.05), i -> (i.u .*= R(0.5)))
                double = ContinuousCallback(crossing(idx), i -> (n[] += 1; i.u .*= 2))
                integ = init(
                    heat(S, idx, a), a; callback = CallbackSet(halve, double), tstops = [R(0.05)],
                    dt = R(1.0e-3), adaptive = false,
                )
                mids = Vector{S}[]
                while !SciMLBase.done(integ)
                    step!(integ)
                    push!(mids, integ((integ.tprev + integ.t) / 2))
                end
                (; sol = integ.sol, mids, fired = n[])
            end
            @test got.sol.retcode == ReturnCode.Success
            @test same_everywhere(got.sol.t)
            @test same_everywhere(got.fired)
            us, mids = gathered(got.sol), [gathered(m) for m in got.mids]
            if rank == 0
                @test got.fired == ref.fired > 0
                @test length(got.sol.t) == length(ref.sol.t)
                @test maximum(abs, got.sol.t - ref.sol.t) <= 4 * eps(R)
                @test maxdiff(us, ref.sol.u) <= bound.serial
                @test maxdiff(mids, ref.mids) <= bound.serial
            end
        end

        @testset "a NaN on one rank's rows is retried as in a serial solve" begin
            breaks!(du, u, p, t) = (du .= ifelse.((p .== N) .& (t > 0.5), NaN, -u); nothing)
            exchanging!(du, u, p, t) = (halo(u); breaks!(du, u, p, t))
            sol, ref = against_serial(TSRK("5dp"; comm), TSRK("5dp")) do idx, a
                f = parallel(a) ? exchanging! : breaks!
                logs, s = Test.collect_test_logs() do
                    solve(ODEProblem(f, ones(S, length(idx)), (zero(R), one(R)), idx), a)
                end
                s
            end
            @test sol.retcode == ReturnCode.Unstable
            @test same_everywhere(sol.t)
            if rank == 0
                @test length(sol.t) == length(ref.t)
                @test maximum(abs, sol.t - ref.t) <= 4 * eps(R)
                @test sol.stats.nreject == ref.stats.nreject
            end
        end

        @testset "f throwing on one rank raises on every rank" begin
            function f(du, u, p, t)
                heat!(du, u, p, t)
                rank == thrower && t > 0.02 && error("f threw")
                return nothing
            end
            for alg in (TSRK("5dp"; comm), TSImplicit("bdf"; comm))
                fn = alg isa TSRK ? f : ODEFunction(f; jac_prototype = heat_proto(S, rows))
                prob = ODEProblem(fn, heat0(S, rows), (zero(R), R(0.1)))
                @test raised(caught(() -> solve(prob, alg)), "f threw")
            end
        end
    end

    @testset "every handle is freed" begin
        @test isempty(PETScDiffEq.PARALLEL_HANDLES)
        @test all(h -> h.destroyed, keys(PETScDiffEq.LIVE_HANDLES))
    end
end
