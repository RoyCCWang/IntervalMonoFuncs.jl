using Documenter

# include("../src/IntervalMonoFuncs.jl")
# import .IntervalMonoFuncs

using IntervalMonoFuncs

makedocs(
    sitename="IntervalMonoFuncs.jl",
    modules=[IntervalMonoFuncs],
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing)=="true"),
    pages=["Home" => "index.md",
        "Public API" => "reference.md",
        #"General usage" => "interpolations.md",
        "License" => "citation.md",
    ],
    #strict=true,
)

# deploydocs(
#     #branch = "gh-pages",
#     #versions = nothing,
#     repo = "github.com/RoyCCWang/IntervalMonoFuncs.jl.git"
# )
