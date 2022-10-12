

"""

example usage:
D = 5 #2 # number of regions.
lb = -1
ub = 1
scale = 1.0 #0.5

# amount of input region used to map to z_lens.
input_range_percentage = 0.95
c = input_range_percentage*(ub-lb)

# generate random boundary points that define the regions.
rand_gen_func = xx->IntervalMonoFuncs.convertcompactdomain(xx, 0.0, 1.0, scale*lb, scale*ub)
z = sort(collect(rand_gen_func(rand()) for d = 1:D))

# sums to c.
z_lens = ones(D) .* 2.0 #0.2
z_lens = z_lens .* (c/sum(z_lens))

# process z points.
z_pts = collect( [z[i]-z_lens[i]/2; z[i]+z_lens[i]/2] for i = 1:D )
z_st0 = collect( z_pts[i][1] for i in eachindex(z_pts) )
z_fin0 = collect( z_pts[i][2] for i in eachindex(z_pts) )

# combine intervals such that they are not overlapping.
z_st, z_fin = IntervalMonoFuncs.mergesuchthatnooverlap(z_st0, z_fin0)
"""
function mergesuchthatnooverlap(z_st_in::Vector{T}, z_fin_in::Vector{T};
    lb::T = -one(T),
    ub::T = one(T),
    ϵ = 1e-9) where T

    inds = sortperm(z_st_in)
    z_st = z_st_in[inds]
    z_fin = z_fin_in[inds]

    @assert length(z_st) == length(z_fin)

    out_st = Vector{T}(undef, 0)
    out_fin = Vector{T}(undef, 0)

    fin = -Inf
    for j in eachindex(z_st)

        if fin < z_st[j]
            st = z_st[j]
            fin = findendnooverlap(z_fin[j], z_st[j:end], z_fin[j:end])

            push!(out_st, st)
            push!(out_fin, fin)
        end
    end

    # # sanity check.
    #portion_for_shift::T = 0.95
    # c = (ub-lb)*portion_for_shift
    # C = sum( out_fin[i]-out_st[i] for i in eachindex(out_fin) )
    # @assert norm(C - c) < 1e-12
    
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


#unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
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

    xs, ys, ms, bs, len_s, len_z = IntervalMonoFuncs.getpiecewiselines([ focus_lower; ], [ focus_upper; ], c)
    f = xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)*scale
    finv = xx->IntervalMonoFuncs.evalinversepiecewise2Dlinearfunc(xx/scale, xs, ys, ms, bs)

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