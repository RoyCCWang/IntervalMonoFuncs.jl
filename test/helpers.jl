"""
convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real
returns (x-a)*(d-c)/(b-a)+c

converts x ∈ [a,b] to y ∈ [c,d].
"""
function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end

function generatestartfinishpts(N_intervals::Int, lb::T, ub::T) where T <: Real
    N_pts = 2*N_intervals
    X = sort(collect( convertcompactdomain(rand(T), zero(T), one(T), lb, ub) for d = 1:N_pts ))

    start_pts = Vector{T}(undef, N_intervals)
    finish_pts = Vector{T}(undef, N_intervals)

    j = 0 # X is for internal use, assume 1-indexing.
    for i in eachindex(start_pts)
        j += 1
        start_pts[i] = X[j]

        j += 1
        finish_pts[i] = X[j]
    end

    return start_pts, finish_pts
end


#### test bench for piecewise-linear functions. Based on /examples/piecewise-linear.jl.
function runpiecewiselineartests(dummy_val::T;
    N_trials = 100,
    N_intervals = 4,
    N_eval_pts = 5000,
    use_lb_as_start = true, # special case involving boundary.
    ZERO_TOL = 1e-10,
    bound_min::T = -200.0, # must be efinite.
    bound_max::T = 200.0) where T <: Real # must be finite.

    for n = 1:N_trials

        ## generate setup.
        
        # parameters.
        # tmp = randn(val_type, 2) .* 50 # randn() can't do BigFloat.
        # lb = minimum(tmp)
        # ub = maximum(tmp)

        tmp1 = convertcompactdomain(rand(T), zero(T), one(T), convert(T, bound_min), convert(T, bound_max))
        tmp2 = convertcompactdomain(rand(T), zero(T), one(T), convert(T, bound_min), convert(T, bound_max))
        lb = min(tmp1,tmp2)
        ub = max(tmp1,tmp2)
        
        domain_proportion = rand(T)

        # points.
        intervals_y_st, intervals_y_fin = generatestartfinishpts(N_intervals, lb, ub)
        
        if use_lb_as_start
            intervals_y_st[1] = lb
        end

        ## get piecewise-linear function.
        info, scale = IntervalMonoFuncs.getpiecewiselines(intervals_y_st, intervals_y_fin, domain_proportion; lb = lb, ub = ub)
        
        f = xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, info, scale)
        finv = xx->IntervalMonoFuncs.evalinversepiecewise2Dlinearfunc(xx, info, scale)

        x_range = LinRange(lb, ub, N_eval_pts)
        f_x = f.(x_range)
        finv_y = finv.(f_x)

        ## tests.
        # check for monotonicity of forward evaluations.
        @test (norm(sort(f_x)-f_x) < ZERO_TOL)

        # check inverse.
        @test norm(sort(finv_y)-x_range) < ZERO_TOL

        # check domain and range coverage.
        start_pts, fin_pts, boundary_pts = IntervalMonoFuncs.getboundarypts(intervals_y_st, intervals_y_fin, lb, ub, info, scale)

        focus_interval_coverage_domain, focus_interval_coverage_range = IntervalMonoFuncs.getintervalcoverages(start_pts, fin_pts, lb, ub)
        @test isapprox(focus_interval_coverage_range, sum(intervals_y_fin-intervals_y_st))
        @test isapprox(focus_interval_coverage_domain, domain_proportion*(ub-lb))
    end

    return nothing
end

##### tests for logisticprobit inverse. Based on /examples/logistic.jl
function runlogistictests(dummy_val::T;
    N_trials = 100,
    a_bounds::Vector{T} = [0.5, 0.6],
    b_bounds::Vector{T} = [-5.0, 5.0],
    N_eval_pts = 5000,
    ZERO_TOL = 1e-10) where T <: Real

    val_type = T

    @assert a_bounds[2] > a_bounds[1]
    @assert b_bounds[2] > b_bounds[1]

    lb = -one(T)
    ub = one(T)

    for n = 1:N_trials
        a = convertcompactdomain(rand(T), zero(T), one(T), a_bounds[1], a_bounds[2])
        b = convertcompactdomain(rand(T), zero(T), one(T), b_bounds[1], b_bounds[2])

        x = LinRange(lb, ub, N_eval_pts)
        q = tt->IntervalMonoFuncs.evalcompositelogisticprobit(tt, a, b, lb, ub)
        q_inv = yy->IntervalMonoFuncs.evalinversecompositelogisticprobit(yy, a, b, lb, ub)

        @test norm(q_inv.(q.(x)) - x) < ZERO_TOL
    end

    return nothing
end



##### tests for createendopiewiselines1(). Based on /examples/fit_logistic-logit.jl and /examples/piecewise-linear.jl.
function runcreateendopiewiselines1tests(;
    val_type = Float64,
    N_trials = 100,
    max_N_itp_samples = 10,
    ZERO_TOL = 1e-10)

    T = val_type
    p_lb = -one(T) # logistic-logit maps [0,1]->[0,1], so we use 0 and 1 for our bounds.
    p_ub = one(T)
    scale = one(T) # since -1 <= p_lb < p_ub <= 1.0, which is the required setup for createendopiewiselines1(). See /examples/piecewise-linear.jl for more insight on scale.

    for k = 1:N_trials

        range_proportion = rand(T) # the focus interval should have this much coverage on the range (vertical axis). In proportion units, i.e., 0 to 1.
        N_itp_samples = rand(2:max_N_itp_samples) # must be a positive integer greater than 1, up to max_N_itp_samples.
        domain_proportion = rand(T) # the focus interval should have this much coverage on the domain (horizontal axis). In proportion units, i.e., 0 to 1.
        
        infos, zs, p_range = IntervalMonoFuncs.createendopiewiselines1(p_lb, p_ub, range_proportion; N_itp_samples = N_itp_samples, domain_proportion = domain_proportion)

        # test.
        for n in eachindex(zs)

            lb = p_lb
            ub = p_ub
            intervals_y_st = [first(zs[n]);]
            intervals_y_fin = [last(zs[n]);]
            
            start_pts, fin_pts, _ = IntervalMonoFuncs.getboundarypts(intervals_y_st, intervals_y_fin, lb, ub, infos[n], scale)

            focus_interval_coverage_domain, focus_interval_coverage_range = IntervalMonoFuncs.getintervalcoverages(start_pts, fin_pts, lb, ub)
            @test abs(focus_interval_coverage_range - sum(intervals_y_fin-intervals_y_st)) < ZERO_TOL
            @test abs(focus_interval_coverage_domain - domain_proportion*(ub-lb)) < ZERO_TOL
        end
    end

    return nothing
end