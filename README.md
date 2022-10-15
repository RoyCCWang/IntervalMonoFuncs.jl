# IntervalMonoFuncs.jl
[![CI](https://github.com/RoyCCWang/IntervalMonoFuncs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/RoyCCWang/IntervalMonoFuncs.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://royccwang.github.io/IntervalMonoFuncs.jl/)
A collection of monotonic maps. Only 1D endomorphisms on finite intervals are considered for now.

Consider 1D monotonic functions on an interval, i.e. 

$$ f: \left[ a,b \right] \rightarrow \left[ a,b \right] \subset\mathbb{R}, a < b.$$

This package create piecewise-linear functions and logistic-probit functions of this type; see nomenclature section of the documentation. They can be used to create transport maps that transform the uniform distribution on an interval to some piecewise-uniform distribution that has mass concentrated in the user-specified regions.

See the [documentation page](https://royccwang.github.io/IntervalMonoFuncs.jl/) for details.

## Install:
From the Julia REPL, type `]` then `add https://github.com/RoyCCWang/IntervalMonoFuncs.jl`.

## Examples
All example Julia scripts are in the `/examples` folder. There are also full example guides based on the example Julia scripts on the documentation website.

## Notes
This package was tested against the `Float64` real number concrete Julia type. `BigFloat` was not tested in the current release.

There are known numerical instabilities for certain parameter combinations with `evalcompositelogisticprobit()` and `evalinversecompositelogisticprobit()`. We advise the user to test for numerical stability when using a particular combination of parameters (`a` and `b`) for those functions in the application at hand. See `/examples/logistic-probit.jl` for an example of basic stability test.