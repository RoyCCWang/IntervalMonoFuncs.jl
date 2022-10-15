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
    evalcompositelogisticprobit(x::T, a, b)::T where T <: Real

evaluates the map from `(0,1)` to `(0,1)`.
returns `1/(1 + exp(-a*(log(x/(1-x))-b)))`
"""
function evalcompositelogisticprobit(x::T, a, b)::T where T <: Real

    return 1/(1 + exp(-a*(log(x/(1-x))-b)))
end

function evalcompositelogisticprobit(x_inp::T, a, b, lb, ub)::T where T <: Real

    x = convertcompactdomain(x_inp, lb, ub, zero(T), one(T))
    y = evalcompositelogisticprobit(x, a, b)
    y_out = convertcompactdomain(y, zero(T), one(T), lb, ub)

    return y_out
end

"""
    evalinversecompositelogisticprobit(y::T, a, b)::T where T <: Real

evaluates the map from `(0,1)` to `(0,1)`.
return `exp(b)/(exp(b) + (-1 + 1/y)^(1/a))`
"""
function evalinversecompositelogisticprobit(y::T, a, b)::T where T <: Real
    return exp(b)/(exp(b) + (-1 + 1/y)^(1/a))
end

function evalinversecompositelogisticprobit(y_inp::T, a, b, lb, ub)::T where T <: Real

    y = convertcompactdomain(y_inp, lb, ub, zero(T), one(T))
    x = evalinversecompositelogisticprobit(y, a, b)
    x_out = convertcompactdomain(x, zero(T), one(T), lb, ub)

    return x_out
end

# function eval1Dnumericalinverse(f::Function,
#     y::T,
#     x0::T,
#     a::T,
#     b::T,
#     max_iters::Int) where T <: Real

#     @assert a < b

#     obj_func = xx->((f(xx[1])-y)^2)::T

#     op = Optim.Options( iterations = max_iters,
#                          store_trace = false,
#                          show_trace = false)

#     results = Optim.optimize(   obj_func,
#                                 [x0],
#                                 Optim.NewtonTrustRegion(),
#                                 op)

#     x_star = results.minimizer
#     x_out = clamp(x_star[1], a, b)

#     return x_out, results
# end
