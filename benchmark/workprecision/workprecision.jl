using DiffEqDevTools, LinearAlgebra, Plots, Printf, SciMLBase, SparseArrays, Sundials
using OrdinaryDiffEqBDF, OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqSDIRK, OrdinaryDiffEqTsit5, OrdinaryDiffEqVerner
using PETScDiffEq

const OUT = abspath(
    get(
        ENV, "WORKPRECISION_OUT",
        joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "workprecision"),
    ),
)
const NUMRUNS = parse(Int, get(ENV, "WORKPRECISION_RUNS", "20"))
const COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

struct Setup
    label::String
    alg::Any
    color::Int
    lu::Bool
    extra::Dict{Symbol, Any}
end
Setup(label, alg, color; lu = false, extra...) =
    Setup(label, alg, color, lu, Dict{Symbol, Any}(extra))

is_petsc(alg) = parentmodule(typeof(alg)) === PETScDiffEq
package(alg) = is_petsc(alg) ? "PETScDiffEq" :
    parentmodule(typeof(alg)) === Sundials ? "Sundials" : "OrdinaryDiffEq"

function lotka_volterra!(du, u, p, t)
    du[1] = 1.5 * u[1] - u[1] * u[2]
    du[2] = -3.0 * u[2] + u[1] * u[2]
    return nothing
end

function hires!(du, u, p, t)
    y1, y2, y3, y4, y5, y6, y7, y8 = u
    du[1] = -1.71 * y1 + 0.43 * y2 + 8.32 * y3 + 0.0007
    du[2] = 1.71 * y1 - 8.75 * y2
    du[3] = -10.03 * y3 + 0.43 * y4 + 0.035 * y5
    du[4] = 8.32 * y2 + 1.71 * y3 - 1.12 * y4
    du[5] = -1.745 * y5 + 0.43 * y6 + 0.43 * y7
    du[6] = -280.0 * y6 * y8 + 0.69 * y4 + 1.71 * y5 - 0.43 * y6 + 0.69 * y7
    du[7] = 280.0 * y6 * y8 - 1.81 * y7
    du[8] = -280.0 * y6 * y8 + 1.81 * y7
    return nothing
end

function robertson_residual!(r, du, u, p, t)
    r[1] = -0.04 * u[1] + 1.0e4 * u[2] * u[3] - du[1]
    r[2] = 0.04 * u[1] - 3.0e7 * u[2]^2 - 1.0e4 * u[2] * u[3] - du[2]
    r[3] = u[1] + u[2] + u[3] - 1.0
    return nothing
end

function robertson_mass!(du, u, p, t)
    du[1] = -0.04 * u[1] + 1.0e4 * u[2] * u[3]
    du[2] = 0.04 * u[1] - 3.0e7 * u[2]^2 - 1.0e4 * u[2] * u[3]
    du[3] = u[1] + u[2] + u[3] - 1.0
    return nothing
end

function brusselator!(du, x, p, t)
    n, a = p
    for i in 1:n
        u, v = x[2i - 1], x[2i]
        ul, vl = i == 1 ? (1.0, 3.0) : (x[2i - 3], x[2i - 2])
        ur, vr = i == n ? (1.0, 3.0) : (x[2i + 1], x[2i + 2])
        du[2i - 1] = 1.0 + u^2 * v - 4.0 * u + a * (ul - 2u + ur)
        du[2i] = 3.0 * u - u^2 * v + a * (vl - 2v + vr)
    end
    return nothing
end

function brusselator_pattern(n)
    rows, cols = Int[], Int[]
    for i in 1:n, (r, c) in ((2i - 1, 2i - 1), (2i - 1, 2i), (2i, 2i - 1), (2i, 2i))
        push!(rows, r)
        push!(cols, c)
    end
    for k in 1:(2n - 2)
        append!(rows, (k, k + 2))
        append!(cols, (k + 2, k))
    end
    return sparse(rows, cols, ones(length(rows)), 2n, 2n)
end

function brusselator_problem(n)
    x = range(0, 1; length = n + 2)[2:(end - 1)]
    u0 = vec(permutedims([1.0 .+ sin.(2pi .* x) fill(3.0, n)]))
    f = ODEFunction(brusselator!; jac_prototype = brusselator_pattern(n))
    return ODEProblem(f, u0, (0.0, 10.0), (n, (1 / 50) * (n + 1)^2))
end

function reference(prob, alg; tol)
    return solve(prob, alg; abstol = tol, reltol = tol, save_everystep = false, maxiters = 10^8)
end

function run_set(b)
    ref, check = b.references()
    wp = WorkPrecisionSet(
        b.prob, b.abstols, b.reltols,
        [Dict{Symbol, Any}(:alg => s.alg, s.extra...) for s in b.setups];
        names = [s.label for s in b.setups], save_everystep = false, maxiters = 10^7,
        appxsol = b.prob isa AbstractVector ? fill(ref, length(b.prob)) : ref,
        numruns = NUMRUNS,
    )
    write_csv(b.name, b.setups, wp)
    plot_set(b)
    return maximum(abs, ref.u[end] - check.u[end])
end

quoted(x) = "\"" * replace(x, "\"" => "\"\"") * "\""

function write_csv(name, setups, wp)
    open(joinpath(OUT, name * ".csv"), "w") do io
        println(
            io, "solver,package,abstol,reltol,error,time_s,steps,rejects,nf,njacs,newton_iters",
        )
        for (s, w) in zip(setups, wp.wps), i in 1:(w.N)
            st = w.stats[i]
            @printf(
                io, "%s,%s,%.0e,%.0e,%.3e,%.4e,%d,%d,%d,%d,%d\n", quoted(s.label),
                package(s.alg), w.abstols[i], w.reltols[i], w.errors.final[i], w.times[i],
                st.naccept, st.nreject, st.nf, st.njacs, st.nnonliniter,
            )
        end
    end
    return nothing
end

function read_csv(name)
    rows = Dict{String, Vector{Tuple{Float64, Float64}}}()
    for line in Iterators.drop(eachline(joinpath(OUT, name * ".csv")), 1)
        m = match(r"^\"((?:[^\"]|\"\")*)\",(.*)$", line)
        fields = split(m[2], ",")
        err, t = parse(Float64, fields[4]), parse(Float64, fields[5])
        isfinite(err) && isfinite(t) || continue
        push!(get!(rows, replace(m[1], "\"\"" => "\""), Tuple{Float64, Float64}[]), (err, t))
    end
    return rows
end

function time_at(points, target)
    isempty(points) && return NaN, true
    front = Tuple{Float64, Float64}[]
    for p in sort(points; by = last)
        (isempty(front) || p[1] < front[end][1]) && push!(front, p)
    end
    front[1][1] <= target && return front[1][2], false
    for ((e1, t1), (e2, t2)) in zip(front, front[2:end])
        e1 >= target >= e2 || continue
        w = log(e1 / target) / log(e1 / e2)
        return exp((1 - w) * log(t1) + w * log(t2)), true
    end
    return NaN, true
end

function code_label(label)
    depth = 0
    for (i, c) in pairs(label)
        depth += (c == '(') - (c == ')')
        depth == 0 && c == ',' && return "`" * label[1:prevind(label, i)] * "`" * label[i:end]
    end
    return "`" * label * "`"
end

function summarize(b)
    rows = read_csv(b.name)
    times = [time_at(get(rows, s.label, Tuple{Float64, Float64}[]), b.target) for s in b.setups]
    best = minimum(t for (t, _) in times if isfinite(t))
    @printf("\n%s, time to an error of %.0e\n\n", b.title, b.target)
    println("| Solver | Time (ms) | Relative to the fastest |\n|:--- | ---:| ---:|")
    for (s, (t, exact)) in zip(b.setups, times)
        cell = !isfinite(t) ? "not reached" :
            exact ? @sprintf("%.3g", 1.0e3 * t) : @sprintf("at most %.3g", 1.0e3 * t)
        ratio = isfinite(t) ? @sprintf("%.1f", t / best) : ""
        println("| ", code_label(s.label), " | ", cell, " | ", ratio, " |")
    end
    return nothing
end

function plot_set(b)
    rows = read_csv(b.name)
    plt = plot(;
        xscale = :log10, yscale = :log10, xticks = exp10.(-16:0), yticks = exp10.(-6:1),
        xlabel = "error at the final time", ylabel = "wall time (s)", b.title,
        legend = :outerright, size = (900, 480), framestyle = :box, gridalpha = 0.15,
        titlefontsize = 11, guidefontsize = 10, legendfontsize = 8,
        left_margin = 4Plots.mm, bottom_margin = 4Plots.mm, background_color = :white,
    )
    for s in b.setups
        points = get(rows, s.label, Tuple{Float64, Float64}[])
        plot!(
            plt, first.(points), last.(points); label = s.label, color = COLORS[s.color],
            linewidth = 2, markersize = 4, markerstrokewidth = 0,
            linestyle = s.lu ? :dot : is_petsc(s.alg) ? :solid : :dash,
            markershape = s.lu ? :utriangle : is_petsc(s.alg) ? :circle : :rect,
        )
    end
    savefig(plt, joinpath(OUT, b.name * ".png"))
    return nothing
end

function nonstiff()
    prob = ODEProblem(lotka_volterra!, [1.0, 1.0], (0.0, 10.0))
    references() = (
        reference(prob, Vern9(); tol = 1.0e-14), reference(prob, Vern8(); tol = 1.0e-14),
    )
    setups = [
        Setup("TSRK(\"3bs\")", TSRK("3bs"), 1), Setup("BS3", BS3(), 1),
        Setup("TSRK(\"5dp\")", TSRK("5dp"), 2), Setup("DP5", DP5(), 2),
        Setup("TSRK(\"7vr\")", TSRK("7vr"), 3), Setup("Vern7", Vern7(), 3),
        Setup("TSRK(\"8vr\")", TSRK("8vr"), 4), Setup("Vern8", Vern8(), 4),
        Setup("Tsit5", Tsit5(), 5), Setup("CVODE_Adams", CVODE_Adams(), 6),
    ]
    return (;
        name = "lotka_volterra", title = "Lotka-Volterra, non-stiff, 2 states", prob,
        setups, abstols = 1.0 ./ 10.0 .^ (6:13), reltols = 1.0 ./ 10.0 .^ (3:10),
        references, target = 1.0e-8,
    )
end

hires_problem() = ODEProblem(hires!, [1.0, 0, 0, 0, 0, 0, 0, 0.0057], (0.0, 321.8122))

function stiff()
    prob = hires_problem()
    references() = (
        reference(prob, Rodas5P(); tol = 1.0e-14), reference(prob, CVODE_BDF(); tol = 1.0e-13),
    )
    setups = [
        Setup("TSRosW(\"ra34pw2\")", TSRosW("ra34pw2"), 1), Setup("ROS34PW2", ROS34PW2(), 1),
        Setup("TSRosW(\"rodas3\")", TSRosW("rodas3"), 2), Setup("Rodas3", Rodas3(), 2),
        Setup("TSImplicit(\"bdf\"; order = 5)", TSImplicit("bdf"; order = 5), 3),
        Setup("FBDF", FBDF(), 3),
        Setup("TSARKIMEX(\"4\")", TSARKIMEX("4"), 4), Setup("KenCarp4", KenCarp4(), 4),
        Setup("Rodas5P", Rodas5P(), 5), Setup("CVODE_BDF", CVODE_BDF(), 6),
    ]
    return (;
        name = "hires", title = "HIRES, stiff, 8 states", prob, setups,
        abstols = 1.0 ./ 10.0 .^ (5:10), reltols = 1.0 ./ 10.0 .^ (2:7), references,
        target = 1.0e-7,
    )
end

function dae()
    u0, du0 = [1.0, 0.0, 0.0], [-0.04, 0.04, 0.0]
    tspan = (0.0, 1.0e5)
    residual = DAEProblem(
        robertson_residual!, du0, u0, tspan; differential_vars = [true, true, false],
    )
    mass = ODEProblem(
        ODEFunction(robertson_mass!; mass_matrix = Diagonal([1.0, 1.0, 0.0])), u0, tspan,
    )
    references() = (
        reference(mass, Rodas5P(); tol = 1.0e-14), reference(residual, IDA(); tol = 1.0e-12),
    )
    setups = [
        Setup("TSDAE(\"bdf\"; order = 5)", TSDAE("bdf"; order = 5), 1),
        Setup("DFBDF", DFBDF(), 1), Setup("IDA", IDA(), 6),
        Setup(
            "TSImplicit(\"bdf\"; order = 5), mass matrix",
            TSImplicit("bdf"; order = 5), 3; prob_choice = 2,
        ),
        Setup("FBDF, mass matrix", FBDF(), 3; prob_choice = 2),
        Setup(
            "TSRosW(\"ra34pw2\"), mass matrix", TSRosW("ra34pw2"), 2; prob_choice = 2,
        ),
        Setup("ROS34PW2, mass matrix", ROS34PW2(), 2; prob_choice = 2),
        Setup("Rodas5P, mass matrix", Rodas5P(), 5; prob_choice = 2),
    ]
    return (;
        name = "robertson_dae", title = "Robertson DAE, 3 states", prob = [residual, mass],
        setups, abstols = 1.0 ./ 10.0 .^ (6:10), reltols = 1.0 ./ 10.0 .^ (2:6), references,
        target = 1.0e-6,
    )
end

function pde()
    prob = brusselator_problem(500)
    band = CVODE_BDF(; linear_solver = :Band, jac_upper = 2, jac_lower = 2)
    references() = (
        reference(prob, Rodas5P(); tol = 1.0e-12), reference(prob, band; tol = 1.0e-12),
    )
    lu = ["-ksp_type", "preonly", "-pc_type", "lu"]
    setups = [
        Setup("TSRosW(\"ra34pw2\")", TSRosW("ra34pw2"), 1),
        Setup("TSRosW(\"ra34pw2\"), LU", TSRosW("ra34pw2", lu), 1; lu = true),
        Setup("ROS34PW2", ROS34PW2(), 1),
        Setup("TSImplicit(\"bdf\"; order = 5)", TSImplicit("bdf"; order = 5), 2),
        Setup(
            "TSImplicit(\"bdf\"; order = 5), LU", TSImplicit("bdf", lu; order = 5), 2;
            lu = true,
        ),
        Setup("FBDF", FBDF(), 2),
        Setup("TSARKIMEX(\"4\")", TSARKIMEX("4"), 3), Setup("KenCarp4", KenCarp4(), 3),
        Setup("Rodas5P", Rodas5P(), 5), Setup("CVODE_BDF, banded", band, 6),
    ]
    return (;
        name = "brusselator", title = "1-D Brusselator, 1000 states, sparse Jacobian",
        prob, setups, abstols = 1.0 ./ 10.0 .^ (5:9), reltols = 1.0 ./ 10.0 .^ (3:7),
        references, target = 1.0e-5,
    )
end

decay!(du, u, p, t) = (@. du = -u; nothing)

function fastest(prob, alg)
    run() = solve(prob, alg; dt = 1.0e-3, adaptive = false, save_everystep = false)
    run()
    return minimum(@elapsed(run()) for _ in 1:NUMRUNS)
end

function overhead()
    open(joinpath(OUT, "overhead.csv"), "w") do io
        println(io, "states,package,per_solve_us,per_step_us")
        for n in (1, 100, 10_000), alg in (TSRK("4"), RK4())
            one = fastest(ODEProblem(decay!, ones(n), (0.0, 1.0e-3)), alg)
            many = fastest(ODEProblem(decay!, ones(n), (0.0, 1.0)), alg)
            @printf(
                io, "%d,%s,%.2f,%.3f\n", n, package(alg), 1.0e6 * one,
                1.0e6 * (many - one) / 999,
            )
        end
    end
    return nothing
end

function main()
    BLAS.set_num_threads(1)
    mkpath(OUT)
    benchmarks = [nonstiff(), stiff(), dae(), pde()]
    gaps = map(run_set, benchmarks)
    overhead()
    open(joinpath(OUT, "reference.csv"), "w") do io
        println(io, "problem,reference_gap")
        for (b, gap) in zip(benchmarks, gaps)
            @printf(io, "%s,%.3e\n", b.name, gap)
        end
    end
    open(joinpath(OUT, "environment.csv"), "w") do io
        println(io, "name,value")
        println(io, "julia,", VERSION)
        println(io, "cpu,", quoted(Sys.cpu_info()[1].model))
        for m in (
                PETScDiffEq, PETScDiffEq.PETSc, DiffEqDevTools, OrdinaryDiffEqBDF,
                OrdinaryDiffEqRosenbrock, OrdinaryDiffEqSDIRK, Sundials,
            )
            println(io, nameof(m), ",", pkgversion(m))
        end
        jll = only(m for m in values(Base.loaded_modules) if nameof(m) === :PETSc_jll)
        println(io, "PETSc_jll,", pkgversion(jll))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "report" in ARGS
        for b in (nonstiff(), stiff(), dae(), pde())
            plot_set(b)
            summarize(b)
        end
    else
        main()
    end
end
