
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

f = xx->MonotoneMaps.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)*scale
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

julia> xs, ys, ms, bs, len_s, len_z = MonotoneMaps.getpiecewiselines(z_st, z_fin, c)
([-1.0, -0.842857142857143, 0.9571428571428571, 1.0], [-1.0, -0.12, 0.76, 1.0], [5.600000000000001, 0.4888888888888889, 5.600000000000001], [4.600000000000001, 0.29206349206349214, -4.600000000000001], [1.8], [0.88])
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

    return evalpiecewise2Dlinearfunc(y, A.xs, A.ys, A.ms, A.bs)
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

    ms = collect( len_z[i]/len_s[i] for i = 1:length(len_s) )

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

function mergesuchthatnooverlap(z_st_in::Vector{T}, z_fin_in::Vector{T};
    lb::T = -one(T), ub::T = one(T),
    portion_for_shift::T = 0.95) where T

    inds = sortperm(z_st_in)
    z_st = z_st_in[inds]
    z_fin = z_fin_in[inds]

    @assert length(z_st) == length(z_fin)

    out_st = Vector{T}(undef, 0)
    out_fin = Vector{T}(undef, 0)

    fin = -Inf
    for j = 1:length(z_st)

        if fin < z_st[j]
            st = z_st[j]
            fin = findendnooverlap(z_fin[j], z_st[j:end], z_fin[j:end])

            push!(out_st, st)
            push!(out_fin, fin)
        end
    end

    # # sanity check.
    # c = (ub-lb)*portion_for_shift
    # C = sum( out_fin[i]-out_st[i] for i = 1:length(out_fin) )
    # @assert norm(C - c) < 1e-12

    ϵ = 1e-9
    clamp!(out_st, lb+ϵ, ub-ϵ)
    clamp!(out_fin, lb+ϵ, ub-ϵ)

    return out_st, out_fin
end

function findendnooverlap(fin::T, z_st::Vector{T}, z_fin::Vector{T})::T where T
    for i = 2:length(z_st)
        if z_st[i] < fin
            fin = z_fin[i]
        else
            return fin
        end
    end

    return fin
end


"""
Given a few candidate shift zones (a center and window per zone), return a monotonic map that
emphasizes the candidate shift zones. The amount of emphasis is controlled by portion_for_shift.
"""
function setupshiftwarpmap(shift_centers::Vector{T},
    shift_windows::Vector{T};
    portion_for_shift::T = 0.95) where T

    portion_for_shift = clamp(portion_for_shift, 1e-5, one(T)-1e-5)

    # set up.
    N_zones = length(shift_centers)
    @assert length(shift_windows) == N_zones

    c = portion_for_shift*2

    z_pts = collect( [shift_centers[i]-shift_windows[i]/2;
    shift_centers[i]+shift_windows[i]/2] for i = 1:N_zones )

    z_st0 = collect( z_pts[i][1] for i = 1:N_zones )
    z_fin0 = collect( z_pts[i][2] for i = 1:N_zones )

    # combine intervals such that they are not overlapping.
    z_st, z_fin = mergesuchthatnooverlap(z_st0, z_fin0)

    xs, ys, ms, bs, len_s, len_z = getpiecewiselines(z_st, z_fin, c)
    #f = xx->evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)

    return xs, ys, ms, bs, len_s, len_z
end


####### front end.

function getunityobj(dummy_val::T, ::Val{:PiecewiseLinear}) where T <: Real

    c::T = 0.5 # amount of input region used to map to z_lens.
    z::Vector{T} = [-0.5; 0.5]
    z_lens::Vector{T} = ones(2) .* 0.5

    return Piecewise2DLineType(setupshiftwarpmap(z, z_lens; portion_for_shift = c)...)
end

"""
allocates γ percentage of range to be around p0.
"""
function getproximityobj(p0_inp::T,
    input_range_percentage::T, ::Val{:PiecewiseLinear};
    guard_lb = -0.99, guard_ub = 0.99) where T <: Real

    if p0_lb < p0 < p0_ub
        # TODO I am here.
    end
    c = input_range_percentage*2
    z::Vector{T} = [-0.5; 0.5]
    z_lens::Vector{T} = ones(2) .* 0.5

    return Piecewise2DLineType(setupshiftwarpmap(z, z_lens; portion_for_shift = c)...)
end

# function fetchwarpparms(shift_centers_set::Vector{Vector{T}},
#     shift_windows_set::Vector{Vector{T}};
#     portion_for_shift::T = 0.95) where T <: Real
#
#     N_shifts = length(shift_centers_set)
#     @assert length(shift_windows_set) == N_shifts
#
#     warp_param_set = Vector{Piecewise2DLineType{T}}(undef, N_shifts)
#
#     for i = 1:N_shifts
#         shift_centers = shift_centers_set[i]
#         shift_windows = shift_windows_set[i]
#         @assert length(shift_centers) == length(shift_windows)
#
#         tmp = setupshiftwarpmap(shift_centers, shift_windows; portion_for_shift = portion_for_shift)
#         warp_param_set[i] = Piecewise2DLineType(tmp...)
#     end
#
#     return warp_param_set
# end

# piecve-wise lienar version.
function setupsimilarmonotonemap(shift_centers_set::Vector{Vector{T}},
    shift_windows_set::Vector{Vector{T}};
    portion_for_shift::T = 0.95) where T <: Real

    N_shifts = length(shift_centers_set)
    @assert length(shift_windows_set) == N_shifts

    warp_param_set = Vector{Piecewise2DLineType{T}}(undef, N_shifts)

    for i = 1:N_shifts
        shift_centers = shift_centers_set[i]
        shift_windows = shift_windows_set[i]
        @assert length(shift_centers) == length(shift_windows)

        tmp = setupshiftwarpmap(shift_centers, shift_windows; portion_for_shift = portion_for_shift)
        warp_param_set[i] = Piecewise2DLineType(tmp...)
    end

    return warp_param_set
end


################# application as a proximity-encouraging map.

# uses lb = -1, ub = 1.
function setupproximitymap(focus_center::T, focus_radius::T;
    input_range_percentage = 0.9,
    scale::T = one(T)) where T <: Real

    @assert focus_radius < 1
    @assert -1 < focus_center < 1

    c::T = input_range_percentage*2

    focus_lower = focus_center - focus_radius
    focus_upper = focus_center + focus_radius

    xs, ys, ms, bs, len_s, len_z = MonotoneMaps.getpiecewiselines([ focus_lower; ], [ focus_upper; ], c)
    f = xx->MonotoneMaps.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)*scale
    finv = xx->MonotoneMaps.evalinversepiecewise2Dlinearfunc(xx/scale, xs, ys, ms, bs)

    return f, finv
end

function setupproximitymapsitp(focus_lb::T, focus_ub::T, focus_radius::T;
    input_range_percentage::T = 0.9,
    scale::T = one(T)) where T <: Real

    @assert (focus_ub-focus_lb) > focus_radius

    focus_center = focus_lb + focus_radius
    f1, finv1 = setupproximitymap(focus_center, focus_radius;
        input_range_percentage = input_range_percentage,
        scale = scale)

    focus_center = focus_ub - focus_radius
    f2, finv2 = setupproximitymap(focus_center, focus_radius;
        input_range_percentage = input_range_percentage,
        scale = scale)

    #
    f = itpproximitymap(f1, f2, focus_center)

    return f
end

# function evallinearitp(x::T, x1::T, y1::T, x2::T, y2::T)::T where T
#     return (y1*(x2-x) + y2*(x-x1))/(x2-x1)
# end

function itpproximitymap(f1, f2, y::T) where T
    f2_t = f2(t)
    f1_t = f1(t)

    a = (f2_t - y) / (f2_t - f1_t)
    if f1_t > y
        a = zero(T)
    else
        f2_t < y
        a = one(T)
    end

    f = tt->(a*f1(tt) + (1-a)*f2(t))

    return f
end

#####################

function getendomorphismpiecewiselinear(p_lb::T, p_ub::T, window::T;
    N_itp_samples::Int = 10,
    input_range_percentage::T = 0.9) where T

    #ϵ = window/100

    #p_range = LinRange(p_lb + window + ϵ, p_ub - window - ϵ, N_itp_samples)
    p_range = LinRange(p_lb + window, p_ub - window, N_itp_samples)

    lb = zero(T)
    ub = one(T)
    c = input_range_percentage*(ub-lb)

    infos = Vector{Piecewise2DLineType{T}}(undef, length(p_range))
    zs = Vector{Vector{T}}(undef, length(p_range))

    for (i,p) in enumerate(p_range)

        z_st = [p - window;]
        z_fin = [p + window;]

        xs, ys, ms, bs, len_s, len_z = getpiecewiselines(z_st, z_fin, c; lb = lb, ub = ub)
        infos[i] = Piecewise2DLineType(xs, ys, ms, bs, len_s, len_z)

        zs[i] = [z_st[1]; z_fin[1]]
    end

    return infos, zs, p_range
end