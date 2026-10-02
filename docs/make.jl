using Documenter, PETScDiffEq

if isfile(joinpath(@__DIR__, "Manifest.toml"))
    for file in ("Manifest.toml", "Project.toml")
        cp(joinpath(@__DIR__, file), joinpath(@__DIR__, "src", "assets", file); force = true)
    end
end

include("pages.jl")

makedocs(;
    sitename = "PETScDiffEq.jl",
    authors = "Harsh Singh",
    modules = [PETScDiffEq],
    clean = true,
    doctest = false,
    linkcheck = true,
    checkdocs = :exports,
    format = Documenter.HTML(;
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/PETScDiffEq/stable/",
    ),
    pages = pages,
)

deploydocs(; repo = "github.com/SciML/PETScDiffEq.jl.git", push_preview = true)
