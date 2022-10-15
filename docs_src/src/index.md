
This library provides methods to construct transport maps that transform the uniform distribution on an interval to some distribution on the same interval that has mass concentrated in user-specified regions.

## Install
Add (currently) unregistered public Julia package for dependency before installing NMRSpectraSimulator.jl
``` julia
import Pkg
Pkg.add(path="https://github.com/RoyCCWang/IntervalMonoFuncs.jl")
```

To update this package once it is installed, do
``` julia
Pkg.update("IntervalMonoFuncs")
```

# Overview
The following summarizes the usage of the public API:

* `getpiecewiselines()` constructs piecewise-linear functions from finite intervals on $\mathbb{R}$ to the same interval (an endomorphism in mathematics). The constructed function is a transport map that transforms the uniform distribution over the interval to a piecewise-uniform distribution over the interval. The construction method requires inputs that specify properties of the piecewise-uniform distribution and the endomorphism domain/range interval.

* methods to evaluate piecewise-linear and a the composition of a logistic and a probit function, and their inverse. A composite function of this type is referred as a *logistic-probit* function throughout this package. It has domain and range [0,1].

* `createendopiewiselines1()` is similar to `getpiecewiselines()`, but is designed specifically to generate a family of two-segment piecewise-linear functions. The generated functions are evenly "spaced/centered" over the user-specified subintervals in the domain and range. The domain and range is fixed to a subset of [-1,1] in the current version of this package. This creates a transport map that could drastically relocate the mass to a single interval.

* `getlogisticprobitparameters()` is a routine to fit the parameters of a logistic-probit function to each family of two-segment piecewise-linear functions generated by `createendopiewiselines1()`, if the domain and range is set to the interval [0,1]. This creates a smooth version for each of the piecewise-uniform transport maps (each a two-segment piecewise-linear function).

## Nomenclature
See our [nomenclature](./out/nomenclature.html) page.

## Usage examples

### Julia examples:

* piecewise-linear construction guide: [HTML](./out/piecewise-linear.html), [Jupyter notebook](./out/piecewise-linear.ipynb)

* logistic-probit fit and usage guide: [HTML](./out/logistic-logit_fit.html), [Jupyter notebook](./out/logistic-logit_fit.ipynb)






