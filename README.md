# IntervalMonoFuncs.jl
[![Build Status](https://github.com/RoyCCWang/IntervalMonoFuncs.jl/workflows/CI/badge.svg)](https://github.com/RoyCCWang/IntervalMonoFuncs.jl/actions)
[![Coverage](https://codecov.io/gh/RoyCCWang/IntervalMonoFuncs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoyCCWang/IntervalMonoFuncs.jl)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ai4dbiological-systems.github.io/IntervalMonoFuncs.jl/)
A collection of monotonic maps. Only 1D endomorphisms on finite intervals are considered for now.


## To Install:
From the Julia REPL, type `]` then `add https://github.com/RoyCCWang/IntervalMonoFuncs.jl`

## Examples
All example scripts are in the `/examples` folder. There are:
- the piece-wise linear monotonic maps over a specified finite interval (see `warp.jl` and `piece-wise_linear.jl`)
- a sigmoid/probit-like monotonic map (see `logistic-logit_fit.jl`)

See the [documentation page](https://royccwang.github.io/IntervalMonoFuncs.jl/) for more information and walk-through of the two example scripts mentioned above.


createendopiewiselines1
getcompactsigmoidparameters

evalinversecompositelogisticprobit
evalcompositelogisticprobit

https://thewinnower.com/papers/9324-finalizing-your-julia-package-documentation-testing-coverage-and-publishing
