# MonotoneMaps.jl
[![Build Status](https://github.com/RoyCCWang/MonotoneMaps.jl/workflows/CI/badge.svg)](https://github.com/RoyCCWang/MonotoneMaps.jl/actions)
[![Coverage](https://codecov.io/gh/RoyCCWang/MonotoneMaps.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoyCCWang/MonotoneMaps.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ai4dbiological-systems.github.io/MonotoneMaps.jl/)
A collection of monotonic maps. Only 1D endomorphisms on finite intervals are considered for now.

## To Install:
From the Julia REPL, type `]` then `add https://github.com/RoyCCWang/MonotoneMaps.jl`

## Examples
All example scripts are in the `/examples` folder. There are:
- the piece-wise linear monotonic maps over a specified finite interval (see `warp.jl` and `piece-wise_linear.jl`)
- a sigmoid/probit-like monotonic map (see `logistic-logit_fit.jl`)

See the [documentation page](https://royccwang.github.io/MonotoneMaps.jl/) for more information and walk-through of the two example scripts mentioned above.
