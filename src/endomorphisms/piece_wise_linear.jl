
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
    getpiecewiselines(z_st::Vector{T}, z_fin::Vector{T}, c::T;
        lb::T = -one(T), ub::T = one(T))

Computes the parameters of a piece-wise linear function that contain the intervals in z_st and z_fin in the range, over the range in c.

f = xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)*scale
maps [lb, ub] to [lb, ub]*scale.

# Examples
```jldoctest
julia> c = 1.8
1.8

julia> z_st = [ -0.12; ]
1-element Vector{Float64}:
 -0.12

julia> z_fin = [ 0.76; ]
1-element Vector{Float64}:
 0.76

julia> xs, ys, ms, bs, len_s, len_z = IntervalMonoFuncs.getpiecewiselines(z_st, z_fin, c)
([-1.0, -0.842857142857143, 0.9571428571428571, 1.0], [-1.0, -0.12, 0.76, 1.0], [5.600000000000001, 0.4888888888888889, 5.600000000000001], [4.600000000000001, 0.29206349206349214, -4.600000000000001], [1.8], [0.88])

See piece-wise_linear.jl in the /examples folder for an example.
```
"""
function getpiecewiselines(z_st::Vector{T}, z_fin::Vector{T}, c::T;
    lb::T = -one(T), ub::T = one(T)) where T

    N_zones = length(z_st)

    @assert zero(T) < c < (ub-lb)

    ## set up s.
    @assert N_zones == length(z_fin)
    len_z = collect( z_fin[i]-z_st[i] for i = 1:N_zones )

    # sanity check.
    @assert all( len_z .> zero(T) )

    Z = sum(len_z)
    len_s = collect( c * len_z[i]/Z for i = 1:N_zones )

    ## get slope.
    m_zones, m0 = getslope(len_z, len_s, ub-lb)

    xs, ys, ms, bs = buildtransitionpts(lb, ub, m0, m_zones, z_st, z_fin)

    # numerical sensitive when z_st or z_fin contains lb or ub as an element.
    clamp!(xs, lb, ub)
    clamp!(ys, lb, ub)

    return xs, ys, ms, bs, len_s, len_z
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

function evalinversepiecewise2Dlinearfunc(x_inp::T, A::Piecewise2DLineType{T}, scale::T)::T where T
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
function buildtransitionpts(lb::T, ub::T, m0::T, m_zones::Vector{T}, z_st::Vector{T}, z_fin::Vector{T}) where T
    N_zones = length(m_zones)
    @assert length(z_st) == length(z_fin) == N_zones

    # set up y coodinates of the transition points.
    ys::Vector{T} = [lb;]
    xs::Vector{T} = [lb;]

    ms = Vector{T}(undef, 0)
    bs = Vector{T}(undef, 0)

    for i = 1:N_zones
        push!(ms, m0)
        push!(bs, findyintercept(ms[end], xs[end], ys[end]))

        push!(ys, z_st[i])
        push!(ms, m_zones[i])
        push!(xs, findx2Dline(m0, bs[end], ys[end]))

        push!(bs, findyintercept(ms[end], xs[end], ys[end]))
        push!(ys, z_fin[i])
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
    getendomorphismpiecewiselinear(p_lb::T, p_ub::T, range_percentage::T;
        N_itp_samples::Int = 10,
        domain_percentage::T = 0.9) where T

Get the parameters for a set of `N_itp_samples` piece-wise linear functions that has a focus interval in its range between `p_lb + range_percentage` and `p_ub - range_percentage`.

f = xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)*scale
maps [lb, ub] to [lb, ub]*scale.

# Arguments
Constraint: -1 <= `p_lb`` < `p_ub`` <= 1
- `p_lb::T`: lower bound for the domain and range for each function.
- `p_ub::T`: upper bound for the domain and range for each function.
- `range_percentage::T`: The percentage of the rain to for the focus interval.
- `N_itp_samples::Int`: The number of functions to fit, each with a focus interval that uniformly covers `p_lb + range_percentage` and `p_ub - range_percentage`.
- `domain_percentage::T`: The percentage of the domain to map to the focus interval in the range.

# Example
See /examples/logistic-logit_fit.jl and https://royccwang.github.io/IntervalMonoFuncs.jl/

"""
function getendomorphismpiecewiselinear(p_lb::T, p_ub::T, range_percentage::T;
    N_itp_samples::Int = 10,
    domain_percentage::T = 0.9) where T

    # set up.
    c = domain_percentage*(p_ub-p_lb)
    window = range_percentage*(p_ub-p_lb)

    #p_range = LinRange(p_lb + window + ϵ, p_ub - window - ϵ, N_itp_samples)
    p_range = LinRange(p_lb + window, p_ub - window, N_itp_samples)

    infos = Vector{Piecewise2DLineType{T}}(undef, length(p_range))
    zs = Vector{Vector{T}}(undef, length(p_range))

    for (i,p) in enumerate(p_range)

        z_st = [p - window;]
        z_fin = [p + window;]

        xs, ys, ms, bs, len_s, len_z = getpiecewiselines(z_st, z_fin, c; lb = p_lb, ub = p_ub)
        infos[i] = Piecewise2DLineType(xs, ys, ms, bs, len_s, len_z)

        zs[i] = [z_st[1]; z_fin[1]]
    end

    return infos, zs, p_range
end
