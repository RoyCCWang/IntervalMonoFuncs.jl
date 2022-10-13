using Documenter

include("../src/IntervalMonoFuncs.jl")
import .IntervalMonoFuncs

#import ParametricMonotoneFunctions

makedocs(
    sitename="ParametricMonotoneFunctions.jl",
    format = Documenter.HTML()
)

deploydocs(
    repo = ""
)
