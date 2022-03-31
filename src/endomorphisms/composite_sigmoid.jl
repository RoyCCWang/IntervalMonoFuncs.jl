
"""
    evalcompositelogisticprobit(x::T, a, b)::T where T <: Real

evaluates the map from (0,1) to (0,1).
returns 1/(1 + exp(-a*(log(x/(1-x))-b)))


    evalcompositelogisticprobit(x::T, a, b, lb, ub)::T where T <: Real
evaluates the map from (lb, ub) to (lb, ub), by mapping (lb,ub) to (0,1), run evalcompositelogisticprobit(), then map (0,1) to (lb,ub).    
"""
function evalcompositelogisticprobit(x::T, a, b)::T where T <: Real

    return 1/(1 + exp(-a*(log(x/(1-x))-b)))
end

function evalcompositelogisticprobit(x_inp::T, a, b, lb, ub)::T where T <: Real

    x = convertcompactdomain(x_inp, lb, ub, zero(T), one(T))
    y = 1/(1 + exp(-a*(log(x/(1-x))-b)))
    y_out = convertcompactdomain(y, zero(T), one(T), lb, ub)

    return y_out
end

function evalinversecompositelogisticprobit(y::T, a, b) where T <: Real

    #


    return nothing
end

function eval1Dnumericalinverse(f::Function,
    y::T,
    x0::T,
    a::T,
    b::T,
    max_iters::Int) where T <: Real
    
    @assert a < b

    obj_func = xx->((f(xx[1])-y)^2)::T

    op = Optim.Options( iterations = max_iters,
                         store_trace = false,
                         show_trace = false)

    results = Optim.optimize(   obj_func,
                                [x0],
                                Optim.NewtonTrustRegion(),
                                op)

    x_star = results.minimizer
    x_out = clamp(x_star[1], a, b)

    return x_out, results
end