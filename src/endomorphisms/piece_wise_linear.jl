
#### piecewise linear warping function.

mutable struct Piecewise2DLineType{T}
    xs::Vector{T}
    ys::Vector{T}
    ms::Vector{T}
    bs::Vector{T}
    len_s::Vector{T}
    len_z::Vector{T}
end



"""
    getpiecewiselines(intervals_y_st::Vector{T}, intervals_y_fin::Vector{T}, domain_proportion::T;
        lb::T = -one(T), ub::T = one(T))

Computes the parameters of a piece-wise linear function that contain the intervals in intervals_y_st and intervals_y_fin in the range, over the range in c.

Constraints: 0 < domain_proportion < 1.

f = xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)*scale
maps [lb, ub] to [lb, ub]*scale.

# Examples
```jldoctest
julia> c = 1.8
1.8

julia> intervals_y_st = [ -0.12; ]
1-element Vector{Float64}:
 -0.12

julia> intervals_y_fin = [ 0.76; ]
1-element Vector{Float64}:
 0.76

julia> xs, ys, ms, bs, len_s, len_z, scale = IntervalMonoFuncs.getpiecewiselines(intervals_y_st, intervals_y_fin, domain_proportion)
([-1.0, -0.842857142857143, 0.9571428571428571, 1.0], [-1.0, -0.12, 0.76, 1.0], [5.600000000000001, 0.4888888888888889, 5.600000000000001], [4.600000000000001, 0.29206349206349214, -4.600000000000001], [1.8], [0.88])

See piece-wise_linear.jl in the /examples folder for an example.
```
"""
function getpiecewiselines(intervals_y_st::Vector{T}, intervals_y_fin::Vector{T}, domain_proportion::T;
    lb = -one(T), ub = one(T)) where T

    @assert checkzstfin(intervals_y_st, intervals_y_fin) # monotonicity check.
    @assert zero(T) < domain_proportion < one(T)

    # normalize input.
    lb, ub, scale = normalizebounds(lb, ub)

    y_st = intervals_y_st ./scale
    y_fin = intervals_y_fin ./scale

    # 
    N_zones = length(y_st)
    c = domain_proportion*(ub-lb)
    
    ## set up s.
    @assert N_zones == length(y_fin)
    len_z = collect( y_fin[i]-y_st[i] for i = 1:N_zones )

    # sanity check.
    @assert all( len_z .> zero(T) )

    Z = sum(len_z)
    len_s = collect( c * len_z[i]/Z for i = 1:N_zones )

    ## get slope.
    m_zones, m0 = getslope(len_z, len_s, ub-lb)

    xs, ys, ms, bs = buildtransitionpts(lb, ub, m0, m_zones, y_st, y_fin)

    # numerically sensitive when y_st or y_fin contains lb or ub as an element.
    clamp!(xs, lb, ub)
    clamp!(ys, lb, ub)

    return xs, ys, ms, bs, len_s, len_z, scale
end

function normalizebounds(lb::T, ub::T) where T <: Real
    scale = max(abs(lb),abs(ub))

    return lb/scale, ub/scale, scale
end

"""
    checkzstfin(intervals_y_st::Vector{T}, intervals_y_fin::Vector{T}) where T <: Real

Returns true if `intervals_y_st` and `intervals_y_fin` are valid inputs as `intervals_y_st` and `intervals_y_fin` for `getpiecewiselines()`.

"""
function checkzstfin(intervals_y_st::Vector{T}, intervals_y_fin::Vector{T};
    verbose = false) where T <: Real
    
    if length(intervals_y_st) != length(intervals_y_fin)
        
        if verbose
            println("Warning: Invalid intervals_y_st or intervals_y_fin for getpiecewiselines():")
            println("Please have the same number of start points as finish points")
            println()
        end

        return false

    end

    max_val::T = -Inf
    for i in eachindex(intervals_y_st)
        
        if !(intervals_y_st[i] > max_val)
            
            if verbose
                println("Warning: Invalid intervals_y_st or intervals_y_fin for getpiecewiselines():")
                println("Index $i of intervals_y_st is not strictly monotonic than the previous index of intervals_y_fin.")
                println()
            end

            return false
        end
        max_val = intervals_y_st[i]
        
        if !(intervals_y_fin[i] > max_val)
            
            if verbose
                println("Warning: Invalid intervals_y_st or intervals_y_fin for getpiecewiselines():")
                println("Index $i of finish_point is not strictly monotonic than index $i of intervals_y_st.")
                println()
            end
            
            return false
        end
        max_val = intervals_y_fin[i]

    end

    if verbose
        println("Valid intervals_y_st and intervals_y_fin for getpiecewiselines()")
        println()
    end

    return true
end

function evalpiecewise2Dlinearfunc(x, A::Piecewise2DLineType{T})::T where T

    return evalpiecewise2Dlinearfunc(x, A.xs, A.ys, A.ms, A.bs)
end

function evalpiecewise2Dlinearfunc(x_inp::T, A::Piecewise2DLineType{T}, scale::T)::T where T
    return evalpiecewise2Dlinearfunc(x, A.xs, A.ys, A.ms, A.bs, scale)
end

"""
    evalpiecewise2Dlinearfunc(x_inp::T,
        xs::Vector{T}, ys::Vector{T}, ms::Vector{T}, bs::Vector{T}, scale::T)::T where T

Evaluates a piece-wise linear function.

# Select arguments
Two-element 1-D array for the following. The first entry relates to the a parameter, and the second relates to the b parameter.
- `x_inp::T`: input argument to function.
- `xs::Vector{T}`: x-coordinates for the line endpoints. Length L + 1.
- `ys::Vector{T}`: y-coordinates for the line endpoints. Length L + 1.
- `ms::Vector{T}`: slope for the lines. Length L.
- `bs::Vector{T}`: y-intercepts for the lines. Length L.
- `scale::T`: Re-scaling parameter to `x` so that it is in [-1,1], before applying the piece-wise linear function on the domain [-1,1].


***
evalpiecewise2Dlinearfunc(x_inp::T,
    xs::Vector{T}, ys::Vector{T}, ms::Vector{T}, bs::Vector{T})::T where T

In the case when the `scale` argument is absent, it is taken to be 1.

See piece-wise_linear.jl in the /examples folder for an example.
"""
function evalpiecewise2Dlinearfunc(x_inp::T,
    xs::Vector{T}, ys::Vector{T}, ms::Vector{T}, bs::Vector{T}, scale::T)::T where T

    x = clamp(x_inp/scale, -one(T), one(T))
    return evalpiecewise2Dlinearfunc(x, xs, ys, ms, bs)*scale
end

function evalpiecewise2Dlinearfunc(x::T,
    xs::Vector{T}, ys::Vector{T}, ms::Vector{T}, bs::Vector{T})::T where T

    @assert length(xs)-1 == length(ys)-1 == length(ms) == length(bs)

    ind = findfirst(xx->xx>x, xs)

    if typeof(ind) == Int

        return ms[ind-1]*x + bs[ind-1]
    end

    if x < xs[1]
        return ys[1]
    end

    return ys[end]
end

function evalinversepiecewise2Dlinearfunc(y, A::Piecewise2DLineType{T})::T where T

    return evalinversepiecewise2Dlinearfunc(y, A.xs, A.ys, A.ms, A.bs)
end

function evalinversepiecewise2Dlinearfunc(x::T, A::Piecewise2DLineType{T}, scale::T)::T where T
    return evalinversepiecewise2Dlinearfunc(x, A.xs, A.ys, A.ms, A.bs, scale)
end

"""
    evalinversepiecewise2Dlinearfunc(x_inp::T,
        xs::Vector{T}, ys::Vector{T}, ms::Vector{T}, bs::Vector{T}, scale::T)::T where T

Evaluates the inverse of a (forward) piece-wise linear function, given the parameters of the forward function.

# Select arguments
Two-element 1-D array for the following. The first entry relates to the a parameter, and the second relates to the b parameter.
- `x_inp::T`: input argument to function.
- `xs::Vector{T}`: x-coordinates for the line endpoints of the forward function. Length L + 1.
- `ys::Vector{T}`: y-coordinates for the line endpoints of the forward function. Length L + 1.
- `ms::Vector{T}`: slope for the lines of the forward function. Length L.
- `bs::Vector{T}`: y-intercepts for the lines of the forward function. Length L.
- `scale::T`: Re-scaling parameter to `x` so that it is in [-1,1], before applying the piece-wise linear inverse function on the domain [-1,1].


***
evalinversepiecewise2Dlinearfunc(x_inp::T,
    xs::Vector{T}, ys::Vector{T}, ms::Vector{T}, bs::Vector{T})::T where T

In the case when the `scale` argument is absent, it is taken to be 1.

See piece-wise_linear.jl in the /examples folder for an example.
"""
function evalinversepiecewise2Dlinearfunc(x_inp::T,
    xs::Vector{T}, ys::Vector{T}, ms::Vector{T}, bs::Vector{T}, scale::T)::T where T

    x = clamp(x_inp/scale, -one(T), one(T))
    return evalinversepiecewise2Dlinearfunc(x, xs, ys, ms, bs)*scale
end

function evalinversepiecewise2Dlinearfunc(y::T,
    xs::Vector{T}, ys::Vector{T}, ms::Vector{T}, bs::Vector{T})::T where T

    @assert length(xs)-1 == length(ys)-1 == length(ms) == length(bs)

    ind = findfirst(yy->yy>y, ys)

    if typeof(ind) == Int && ind > 1

        return findx2Dline(ms[ind-1], bs[ind-1], y)
    end

    if y < ys[1]
        return xs[1]
    end

    return xs[end]
end

## build transition points.
function buildtransitionpts(lb::T, ub::T, m0::T, m_zones::Vector{T}, intervals_y_st::Vector{T}, intervals_y_fin::Vector{T}) where T
    N_zones = length(m_zones)
    @assert length(intervals_y_st) == length(intervals_y_fin) == N_zones

    # set up y coodinates of the transition points.
    ys::Vector{T} = [lb;]
    xs::Vector{T} = [lb;]

    ms = Vector{T}(undef, 0)
    bs = Vector{T}(undef, 0)

    for i = 1:N_zones
        push!(ms, m0)
        push!(bs, findyintercept(ms[end], xs[end], ys[end]))

        push!(ys, intervals_y_st[i])
        push!(ms, m_zones[i])
        push!(xs, findx2Dline(m0, bs[end], ys[end]))

        push!(bs, findyintercept(ms[end], xs[end], ys[end]))
        push!(ys, intervals_y_fin[i])
        push!(xs, findx2Dline(m_zones[i], bs[end], ys[end]))
    end
    push!(ys, ub)
    push!(xs, ub)
    push!(ms, m0)
    push!(bs, findyintercept(ms[end], xs[end], ys[end]))

    return xs, ys, ms, bs
end

function getslope(len_z::Vector{T}, len_s::Vector{T}, total_len::T) where T

    ms = collect( len_z[i]/len_s[i] for i in eachindex(len_s) )

    m0 = (total_len-sum(len_z))/(total_len-sum(len_s))

    return ms, m0
end

function findyintercept(m::T, x::T, y::T)::T where T
    return y -m*x
end
## test.
# m = randn()
# b = randn()
# f = xx->(m*xx+b)
#
# x = 3.2
# y = f(x)
# b_rec = findyintercept(m,x,y)
# x_rec = findx2Dline(m,b_rec,y)

function findx2Dline(m::T, b::T, y::T)::T where T
    return (y-b)/m
end

#####################

"""
infos, zs, p_range = createendopiewiselines1(p_lb::T,
        p_ub::T,
        range_proportion::T;
        N_itp_samples::Int = 10,
        domain_proportion::T = 0.9) where T <: Real

Get the parameters for a set of `N_itp_samples` piece-wise linear functions that each has a focus interval between `p_lb + range_proportion/2` and `p_ub - range_proportion/2`.

To evaluate the piecewise linear function via an anonymous function:

Constraints: -one(T) <= p_lb < p_ub <= one(T). Hence, the `1` in the name `createendopiewiselines1`.

# Arguments
Constraint: `p_lb`` < `p_ub``
- `p_lb::T`: lower bound for the domain and range for each function.
- `p_ub::T`: upper bound for the domain and range for each function.
- `range_proportion::T`: The percentage of the range to be mapped to for the focus interval.
- `N_itp_samples::Int`: The number of functions to fit, each with a focus interval that uniformly covers `p_lb + range_proportion` and `p_ub - range_proportion`.
- `domain_proportion::T`: The percentage of the domain to map from for the focus interval.

# Outputs
- `infos::Piecewise2DLineType{T}`: an internal datatype for use with evalpiecewise2Dlinearfunc() to evaluate the generated piecewise linear functions. For example:
```
info = infos[m]
f = xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, info)*scale
f_evals = f.(LinRange(p_lb, p_ub, 200))
```
for a given `scale` and `m` index would construct the function `f` that evaluate the m-th piece-wise linear function, then evaluate over a linear range of points between `p_lb` and `p_ub`. `f` maps values in [p_lb, p_ub] to [p_lb, p_ub]*scale.
- `zs::Vector{Vector{T}}`: zs[m][1] is the start of the focus interval on the range of `f` if `scale` is 1, and zs[m][2] is the end of the same interval.
- `p_range`::LinRange{T,Int}: p_range[m] is the center of the focus interval on the range of `f`.

# Example
See /examples/logistic-logit_fit.jl and https://royccwang.github.io/IntervalMonoFuncs.jl/

"""
function createendopiewiselines1(p_lb::T,
    p_ub::T,
    range_proportion::T;
    N_itp_samples::Int = 10,
    domain_proportion::T = 0.9) where T <: Real

    ## normalize input.
    #p_lb, p_ub, scale = normalizebounds(p_lb, p_ub)
    @assert -one(T) <= p_lb < p_ub <= one(T) # just don't allow other combinations for now.

    # set up.
    window = range_proportion*(p_ub-p_lb)/2

    #p_range = LinRange(p_lb + window + ϵ, p_ub - window - ϵ, N_itp_samples)
    p_range = LinRange(p_lb + window, p_ub - window, N_itp_samples)

    infos = Vector{Piecewise2DLineType{T}}(undef, length(p_range))
    zs = Vector{Vector{T}}(undef, length(p_range))

    for (i,p) in enumerate(p_range)

        intervals_y_st = [p - window;]
        intervals_y_fin = [p + window;]

        # returned scale is 1 and ignored here since we already normalized p_lb and p_ub to be in [-1,1].
        xs, ys, ms, bs, len_s, len_z, _ = getpiecewiselines(intervals_y_st, intervals_y_fin, domain_proportion; lb = p_lb, ub = p_ub)
        infos[i] = Piecewise2DLineType(xs, ys, ms, bs, len_s, len_z)

        zs[i] = [intervals_y_st[1]; intervals_y_fin[1]]
    end

    return infos, zs, p_range
end


### information

function getintervalcoverages(start_pts::Vector{Tuple{T,T}}, fin_pts::Vector{Tuple{T,T}}, lb, ub::T) where T <: Real

    #
    X = sort(unique([start_pts; fin_pts]))
    @assert iseven(length(X))

    focus_interval_coverage_domain = zero(T)
    focus_interval_coverage_range = zero(T)

    process_flag = true
    for i in eachindex(X)
        
        if process_flag
            focus_interval_coverage_domain += X[i+1][1] - X[i][1] 
            focus_interval_coverage_range += X[i+1][2] - X[i][2]

            process_flag = false
        else
            process_flag = true
        end
    end
    
    return focus_interval_coverage_domain, focus_interval_coverage_range
end

function getboundarypts(intervals_y_st, intervals_y_fin, lb, ub::T,
    A::Piecewise2DLineType{T}, scale) where T <: Real

    finv = xx->evalinversepiecewise2Dlinearfunc(xx, A, scale)

    start_pts = collect( (finv(y_st), y_st ) for y_st in intervals_y_st )
    fin_pts = collect( (finv(y_fin), y_fin ) for y_fin in intervals_y_fin )
    boundary_pts = sort(unique([start_pts; fin_pts; (lb,lb); (ub,ub)]))

    return start_pts, fin_pts, boundary_pts
end