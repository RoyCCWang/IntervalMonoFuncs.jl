# MonotoneMaps.jl

A collection of monotonic maps. Only 1D maps are considered for now.

## To Install:
From the Julia REPL, type `]` then `add https://github.com/RoyCCWang/MonotoneMaps.jl`

## Examples (in progress)
So far, there are:
- the piece-wise linear monotonic maps over a specified finite interval (see `warp.jl` and `piece-wise_linear.jl`)
- a sigmoid/probit-like monotonic map (see `logistic.jl`)
- the fitting of the sigmoid/probit-like monotomic map (see `function intervalsigmoid() in utils.jl` to a given piece-wise linear map (see `optim.jl`).

TODO: Use weave.jl to document the scripts in the example folder.
