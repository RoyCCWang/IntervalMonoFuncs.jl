
include("../src/IntervalMonoFuncs.jl")
import .IntervalMonoFuncs
#import IntervalMonoFuncs

using LinearAlgebra
using Test

import Random
Random.seed!(25)


include("./helpers.jl")


@testset "piecewise-linear" begin
    # Write your tests here.

    ZERO_TOL = 1e-10
    N_eval_pts = 5000
    N_trials = 100
    N_intervals = 4
    use_lb_as_start = true # special case involving boundary.

    for n = 1:N_trials

        # case parameters.
        tmp = randn(2) .* 50
        lb = minimum(tmp)
        ub = maximum(tmp)

        domain_proportion = rand()

        intervals_y_st, intervals_y_fin = generatestartfinishpts(N_intervals, lb, ub)
        
        if use_lb_as_start
            intervals_y_st[1] = lb
        end

        # get piece-wise linear function.
        xs, ys, ms, bs, len_s, len_z, scale = IntervalMonoFuncs.getpiecewiselines(intervals_y_st, intervals_y_fin, domain_proportion; lb = lb, ub = ub)
        info = IntervalMonoFuncs.Piecewise2DLineType(xs, ys, ms, bs, len_s, len_z)
        
        f = xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs, scale)
        finv = xx->IntervalMonoFuncs.evalinversepiecewise2Dlinearfunc(xx, xs, ys, ms, bs, scale)

        
        x_range = LinRange(lb, ub, N_eval_pts)
        f_x = f.(x_range)
        finv_y = finv.(f_x)

        # check for monotonicity of forward evaluations.
        @test norm(sort(f_x)-f_x) < ZERO_TOL

        # check inverse.
        @test norm(sort(finv_y)-x_range) < ZERO_TOL

        # check domain and range coverage.
        start_pts, fin_pts, boundary_pts = IntervalMonoFuncs.getboundarypts(intervals_y_st, intervals_y_fin, lb, ub, info, scale)

        boundary_xs = collect(boundary_pts[i][1] for i in eachindex(boundary_pts) )
        boundary_ys = collect(boundary_pts[i][2] for i in eachindex(boundary_pts) )

        focus_interval_coverage_domain, focus_interval_coverage_range = IntervalMonoFuncs.getintervalcoverages(start_pts, fin_pts, lb, ub)
        @test isapprox(focus_interval_coverage_range, sum(intervals_y_fin-intervals_y_st))
        @test isapprox(focus_interval_coverage_domain, domain_proportion*(ub-lb))

    end


end
