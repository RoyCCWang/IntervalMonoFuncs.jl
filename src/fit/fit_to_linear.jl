# Copyright (c) 2022 Roy Wang

# Exhibit A - Source Code Form License Notice
# -------------------------------------------

#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.

# If it is not possible or desirable to put the notice in a particular
# file, then You may include the notice in a location (such as a LICENSE
# file in a relevant directory) where a recipient would be likely to look
# for such a notice.

# You may add additional accurate notices of copyright ownership.




### routines for fitting a composite sigmoid to piecewise-linear functions.

function evalcompositelogisticprobitcost(t_range, f::Function, p::Vector{T})::T where T

    s = zero(T)
    for (i,t) in enumerate(t_range)
        s += ( f(t) - evalcompositelogisticprobit(t, p[1], p[2]) )^2
    end

    return s
end


function getcompactsigmoidparameters(infos::Vector{Piecewise2DLineType{T}},
    runoptimfunc::Function;
    N_fit_positions::Int = 15,
    p0 = [0.5; 0.0],
    p_lb = [0.1; -5.0],
    p_ub = [0.6; 5.0],
    evalcostfunc = evalcompositelogisticprobitcost) where T

    

    fs = collect( xx->evalpiecewise2Dlinearfunc(xx, infos[i]) for i in eachindex(infos) )

    L = length(fs)
    gs = Vector{Function}(undef, L)
    p_star_set = Vector{Vector{T}}(undef, L)

    for l = 1:L

        t_range = LinRange(infos[l].xs[2], infos[l].xs[3], N_fit_positions)

        g = pp->evalcostfunc(t_range, fs[l], pp)
        #dg = xx->Zygote.gradient(f, xx)
        dg = xx->xx # we're not using gradient-based optimization.

        minx = runoptimfunc(p0, g, dg, p_lb, p_ub)
        
        # update.
        p_star_set[l] = minx
        gs[l] = g
    end

    return gs, p_star_set
end

"""
```
getlogisticprobitparameters(infos::Vector{Piecewise2DLineType{T}},
    runoptimfunc::Function;
    N_fit_positions::Int = 15,
    a_lb::T = 0.1,
    a_ub::T = 0.6,
    b_lb::T = -5.0,
    b_ub::T = 5.0,
    a_initial = (a_ub-a_lb)/2,
    b_initial = (b_ub-b_lb)/2) where T <: Real
```

Given a set of single-focus interval region piecewise-linear functions' parameters (collectively contained in `info`), fit the `a` and `b` parameters of the logistic-probit function (see `evalcompositelogisticprobit`) for each piecewise-linear function.

# Inputs
- `infos`: obtained from `createendopiewiselines1()`.

- `runoptimfunc`: This function runs an optimization routine that the user must supply.
`IntervalMonoFuncs.jl` does not currently ship with any optimization routines, nor does it use third party optimization routines as dependencies. This is done so that the user can have more flexibility to choose the optimization package and tuning parameters.

`runoptimfunc` should be in the following form:
```
runoptimfunc = (pp0, ff, dff, pp_lb, pp_ub)->runoptimroutine(pp0, ff, dff, pp_lb, pp_ub, other_args...)
```
where `runoptimroutine()` is the user-supplied routine for invoking their box-constrained numerical minimization code of choice. The other optimization package-specific tuning parameters can go where `other_args...` is.
`runoptimfunc()` should return a `Vector{T}` that contains the solution to the numerical minimization of `ff`.

The `pp0::Vector{T}` input slot is the optimization variable initial guess slot.
the `ff::Function` slot is for the cost function, It should be such that `ff(pp0)` is the cost associated with the initial guess.
the `dff::Function` slot is for the gradient of the cost function, It should be such that `dff(pp0` is the gradient of the cost function evaluated at the initial guess, but one can assign it any function (such as the identity `xx->xx`) if they are not using a gradient-based optimization algorithm in their `runoptimfunc()`.
the `pp_lb::Vector{T}` slot is for the lower bounds of the optimization variable,
the `pp_ub::Vector{T}` slot is for the upper bounds of the variable.

`runoptimfunc()` must return a 1D array of type `Vector{T}`, where `T = eltype(pp0)`.

There are examples on how to create a valid `runoptimfunc` on the repository documentation website and in `/examples/fit_logistic-logit.jl`.

# Optional inputs:
- N_fit_positions: The number of fit positions used in the optimization.
- `a_lb`: lower bound used for optimizing `a`
- `a_ub`: lower bound used for optimizing `a`
- `b_lb`: lower bound used for optimizing `b`
- `b_ub`: lower bound used for optimizing `b`
- `a_initial`: initial guess for `a`
- `b_initial`: initial guess for `b`

# Outputs (in order):
- `costfuncs::Vector{Function}`: the costfunction used to optimize each logistic-probit function against its corresponding piecewise-linear function.
- `minxs::Vector{Vector{T}}`: 1-D array of solution arrays.
`first(minxs[m])` is the optimized `a` variable for the m-th logistic-probit function.
`last(minxs[m])` is the optimized `b` variable for the m-th logistic-probit function.


Usage for `minxs`:
The following creates a vector of functions, each implements a fitted logistic-probit function.
```
qs = collect( tt->IntervalMonoFuncs.evalcompositelogisticprobit(tt, first(minxs[i]), last(minxs[i])) for i in eachindex(minxs) )
```
"""
function getlogisticprobitparameters(infos::Vector{Piecewise2DLineType{T}},
    runoptimfunc::Function;
    N_fit_positions::Int = 15,
    a_lb::T = 0.1,
    a_ub::T = 0.6,
    b_lb::T = -5.0,
    b_ub::T = 5.0,
    a_initial = (a_ub-a_lb)/2,
    b_initial = (b_ub-b_lb)/2) where T <: Real

    @assert a_lb <= a_initial <= a_ub
    @assert b_lb <= b_initial <= b_ub

    return getcompactsigmoidparameters(infos, runoptimfunc;
        N_fit_positions = N_fit_positions,
        p0 = [a_initial; b_initial],
        p_lb = [a_lb; b_lb],
        p_ub = [a_ub; b_ub],
        evalcostfunc = evalcompositelogisticprobitcost)
end