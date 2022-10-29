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

"""
    evalcompositelogisticprobit(x::T, a::T, b::T)::T where T <: Real

evaluates the map from `(0,1)` to `(0,1)`.
returns `1/(1 + exp(-a*(log(x/(1-x))-b)))`

Some numerical stability issues when the magnitudes of `a` or `b` is larger than 2 when `T = Float64`. See `/examples/logistic-probit.jl` for an example of stability test in the root package directory.
"""
function evalcompositelogisticprobit(x::T, a::T, b::T)::T where T <: Real

    return 1/(1 + exp(-a*(log(x/(1-x))-b)))
end

"""
    evalcompositelogisticprobit(x_inp::T, a::T, b::T, lb::T, ub::T)::T where T <: Real

Applies the following:
- Transforms `y` from `[lb,ub]` to `(0,1)`.
- evaluates the map, `1/(1 + exp(-a*(log(x/(1-x))-b)))`, which transforms a value from `(0,1)` to `(0,1)`.
- transforms the evaluated map value from `(0,1)` to `[lb,ub]`, and returns the result.

Some numerical stability issues when the magnitudes of `a` or `b` is larger than 2 when `T = Float64`. See `/examples/logistic-probit.jl` for an example of stability test in the root package directory.
"""
function evalcompositelogisticprobit(x_inp::T, a::T, b::T, lb::T, ub::T)::T where T <: Real

    x = convertcompactdomain(x_inp, lb, ub, zero(T), one(T))
    y = evalcompositelogisticprobit(x, a, b)
    y_out = convertcompactdomain(y, zero(T), one(T), lb, ub)

    return y_out
end

"""
    evalinversecompositelogisticprobit(y::T, a::T, b::T)::T where T <: Real

evaluates the map from `(0,1)` to `(0,1)`.
return `exp(b)/(exp(b) + (-1 + 1/y)^(1/a))`

Some numerical stability issues when the magnitudes of `a` or `b` is larger than 2 when `T = Float64`. See `/examples/logistic-probit.jl` for an example of stability test in the root package directory.
"""
function evalinversecompositelogisticprobit(y::T, a::T, b::T)::T where T <: Real
    return exp(b)/(exp(b) + (-1 + 1/y)^(1/a))
end

"""
    evalinversecompositelogisticprobit(y::T, a::T, b::T)::T where T <: Real

Applies the following:
- Transforms `y` from `[lb,ub]` to `(0,1)`.
- evaluates the map, `exp(b)/(exp(b) + (-1 + 1/y)^(1/a))`, which transforms a value from `(0,1)` to `(0,1)`.
- transforms the evaluated map value from `(0,1)` to `[lb,ub]`, and returns the result.

Some numerical stability issues when the magnitudes of `a` or `b` is larger than 2 when `T = Float64`. See `/examples/logistic-probit.jl` for an example of stability test in the root package directory.
"""
function evalinversecompositelogisticprobit(y_inp::T, a, b, lb, ub)::T where T <: Real

    y = convertcompactdomain(y_inp, lb, ub, zero(T), one(T))
    x = evalinversecompositelogisticprobit(y, a, b)
    x_out = convertcompactdomain(x, zero(T), one(T), lb, ub)

    return x_out
end