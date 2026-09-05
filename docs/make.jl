using Documenter, PETScDiffEq

makedocs(;
    sitename = "PETScDiffEq.jl",
    authors = "Harsh Singh",
    modules = [PETScDiffEq],
    clean = true,
    doctest = false,
    format = Documenter.HTML(;
        canonical = "https://docs.sciml.ai/PETScDiffEq/stable/",
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/SciML/PETScDiffEq.jl.git", push_preview = true)
