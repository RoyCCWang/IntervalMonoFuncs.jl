
#### piecewise linear warping function.

mutable struct Piecewise2DLineType{T}
    xs::Vector{T}
    ys::Vector{T}
    ms::Vector{T}
    bs::Vector{T}
    len_s::Vector{T}
    len_z::Vector{T}
end

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
