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



#### piecewise-linear warping function.

mutable struct Piecewise2DLineType{T}
    xs::Vector{T}
    ys::Vector{T}
    ms::Vector{T}
    bs::Vector{T}
    len_s::Vector{T}
    len_z::Vector{T}
end



"""
```
getpiecewiselines( intervals_y_st::Vector{T},
    intervals_y_fin::Vector{T},
    domain_proportion::T;
    lb::T = -one(T),
    ub::T = one(T))
```
Returns (in order):
- `info::Piecewise2DLineType{T}`
- `scale::T`

Computes the parameters (contained in `info`) of a piecewise-linear function. `intervals_y_st` and `intervals_y_fin` specify the starting and finishing y coordinates of line segments in the piecewise-linear function. The boundary points of the specified interval are automatically accounted for if they are not in `intervals_y_st` and `intervals_y_fin`.

Input constraints:
- `0 < domain_proportion < 1`.
- `intervals_y_st[k]` < `intervals_y_fin[k]` for any valid index `k` of `intervals_y_st` and `intervals_y_fin`. See `checkzstfin()` for a function that checks this condition.

# Example use of `info` and `scale`.
```
f = xx->evalpiecewise2Dlinearfunc(xx, info, scale)
finv = yy->evalinversepiecewise2Dlinearfunc(yy, info, scale)
```
`f` is an anonymous function that takes a value from the interval [`lb`, `ub`] to a value in [`lb`, `ub`]. It mathematically implements the piecewise-linear function created by `getpiecewiselines()`.
`finv` is an anonymous function that implements the mathematic inverse of `f`.

# Example
```
julia> intervals_y_st = [-0.82; 0.59];

julia> intervals_y_fin = [0.044; 0.97];

julia> lb = -2.0;

julia> ub = 1.0;

julia> domain_proportion = 0.9;

julia> info, scale = getpiecewiselines(intervals_y_st, intervals_y_fin, domain_proportion; lb = lb, ub = ub)
(Piecewise2DLineType{Float64}([-1.0, -0.8992027334851938, 0.03841784529294124, 0.08505793640911433, 0.4974373576309793, 0.5], [-1.0, -0.41, 0.022, 0.295, 0.485, 0.5], [5.853333333333337, 0.46074074074074073, 5.853333333333337, 0.4607407407407408, 5.853333333333337], [4.853333333333337, 0.004299333502067071, -0.20287245444801622, 0.2558103433729858, -2.4266666666666685], [0.9376205787781351, 0.41237942122186494], [0.432, 0.19]), 2.0)

julia> _, _, boundary_pts = getboundarypts(intervals_y_st, intervals_y_fin, lb, ub, info, scale);

julia> boundary_pts
6-element Vector{Tuple{Float64, Float64}}:
 (-2.0, -2.0)
 (-1.7984054669703875, -0.82)
 (0.07683569058588248, 0.044)
 (0.17011587281822874, 0.59)
 (0.9948747152619589, 0.97)
 (1.0, 1.0)
```

See /examples/piecewise_linear.jl for the full example.
"""
function getpiecewiselines(intervals_y_st::Vector{T}, intervals_y_fin::Vector{T}, domain_proportion::T;
    lb = -one(T), ub = one(T))::Tuple{Piecewise2DLineType{T}, T} where T <: Real

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
    info = Piecewise2DLineType(xs, ys, ms, bs, len_s, len_z)
        
    return info, scale
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

"""
```
evalpiecewise2Dlinearfunc(x::T, A::Piecewise2DLineType{T}, scale::T)::T where T <: Real
```

Evaluates the piecewise-linear function with parameters contained in `A` and `scale` at input `x`.

Obtain `info` and `scale` from getpiecewiselines().
"""
function evalpiecewise2Dlinearfunc(x::T, A::Piecewise2DLineType{T}, scale::T)::T where T <: Real
    return evalpiecewise2Dlinearfunc(x, A.xs, A.ys, A.ms, A.bs, scale)
end


function evalpiecewise2Dlinearfunc(x_inp::T,
    xs::Vector{T}, ys::Vector{T}, ms::Vector{T}, bs::Vector{T}, scale::T)::T where T <: Real

    x = clamp(x_inp/scale, -one(T), one(T))
    return evalpiecewise2Dlinearfunc(x, xs, ys, ms, bs)*scale
end

function evalpiecewise2Dlinearfunc(x::T,
    xs::Vector{T}, ys::Vector{T}, ms::Vector{T}, bs::Vector{T})::T where T <: Real

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

"""
```
evalinversepiecewise2Dlinearfunc(y::T, A::Piecewise2DLineType{T}, scale::T)::T where T <: Real
```

Evaluates the inverse of the piecewise-linear function `f` at input `y`, where `f` has parameters that are contained in `A` and `scale`.

Obtain `info` and `scale` from `getpiecewiselines()`.
"""
function evalinversepiecewise2Dlinearfunc(y, A::Piecewise2DLineType{T})::T where T <: Real

    return evalinversepiecewise2Dlinearfunc(y, A.xs, A.ys, A.ms, A.bs)
end

function evalinversepiecewise2Dlinearfunc(y::T, A::Piecewise2DLineType{T}, scale::T)::T where T <: Real
    return evalinversepiecewise2Dlinearfunc(y, A.xs, A.ys, A.ms, A.bs, scale)
end

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
```
infos, zs, p_range = createendopiewiselines1(p_lb::T,
        p_ub::T,
        range_proportion::T;
        N_itp_samples::Int = 10,
        domain_proportion::T = 0.9) where T <: Real
```

Get the parameters for a set of `N_itp_samples` two-segment piecewise-linear functions. One of the segments is referred to as the focus interval. The inputs `range_proportion` and `range_proportion` specifies the properties of the focus interval/line segment for each constructed piecewise-linear function in the returned set.

The focus intervals of each function is recorded in `p_range`. They are evenly spaced between `p_lb + range_proportion/2` and `p_ub - range_proportion/2`.


# Inputs:
- `p_lb::T`: lower bound for the domain and range for each function.
- `p_ub::T`: upper bound for the domain and range for each function. Constraint: `-one(T) <= p_lb < p_ub <= one(T)`.
- `range_proportion::T`: The proportion of the range for each function's focus interval. Takes a value between 0 and 1.
- `N_itp_samples::Int`: The number of functions to fit,
- `domain_proportion::T`: The proportion of the domain for each function's focus interval. Takes a value between 0 and 1.


# Outputs
- `infos::Piecewise2DLineType{T}`: an internal datatype for use with `evalpiecewise2Dlinearfunc()` to evaluate the generated piecewise-linear functions.

For example, the following creates an anonymous function for the m-th piecewise-linear function:
```
info = infos[m]
f = xx->evalpiecewise2Dlinearfunc(xx, info)
f_evals = f.(LinRange(p_lb, p_ub, 200))
```
Note the usual `scale` input to `evalpiecewise2Dlinearfunc()` is not required for the piecewise-linear functions returned by `createendopiewiselines1()`.

- `zs::Vector{Vector{T}}`: For a given index m in `zs`, 
`first(zs[m])` is the range coordinate of the start of the focus interval for the m-th piecewise-linear function.
`last(zs[m])` is the range coordinate of the end of the focus interval for the m-th piecewise-linear function.


Do the following to the get boundary points of the piecewise-linear function.
````
intervals_y_st = [first(zs[m]);]
intervals_y_fin = [last(zs[m]);]
start_pts, fin_pts, boundary_pts = IntervalMonoFuncs.getboundarypts(intervals_y_st, intervals_y_fin, lb, ub, infos[m], 1.0)
````

- `p_range::LinRange{T,Int}`: `p_range[m]` is the range coordinate of the center of the focus interval for the m-th function.


See `/examples/fit_logistic-logit.jl` in the package repository and the repository document website for other examples.

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
        infos[i], _ = getpiecewiselines(intervals_y_st, intervals_y_fin, domain_proportion; lb = p_lb, ub = p_ub)

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