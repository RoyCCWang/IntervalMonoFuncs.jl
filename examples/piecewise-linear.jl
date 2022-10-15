# explore warping functions.

include("../src/IntervalMonoFuncs.jl")
import .IntervalMonoFuncs
#import IntervalMonoFuncs

import PyPlot
import Random
using LinearAlgebra
import Interpolations

PyPlot.close("all")
fig_num = 1

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

Random.seed!(25)

# # focus interval can be on the boundary, -1.
# intervals_y_st = [-1; -0.82; 0.59] # This is {c_l} in the documentation. All elements must be between lb and ub.
# intervals_y_fin = [-0.9; 0.044; 0.97] # This is {d_l} in the documentation. All elements must be between lb and ub.

intervals_y_st = [-0.82; 0.59] # This is {c_l} in the documentation. All elements must be between lb and ub.
intervals_y_fin = [0.044; 0.97] # This is {d_l} in the documentation. All elements must be between lb and ub.
println("total proportion of range used by the focus intervals is ", sum(intervals_y_fin-intervals_y_st))

IntervalMonoFuncs.checkzstfin(intervals_y_st, intervals_y_fin)

#
lb = -2.0
ub = 1.0

# specify the total amount of domain for covering the intervals specified by intervals_y_st and intervals_y_fin. In proportion units, i.e., 0 to 1.
domain_proportion = 0.9
println("total proportion of domain used by the focus intervals is ", domain_proportion*(ub-lb))

# the returned scale is 1.0 if -1 <= lb < ub <= 1 by design of getpiecewiselines(). Otherwise it returns lb_normalized, ub_normalized, scale = IntervalMonoFuncs.normalizebounds(lb, ub).
info, scale = IntervalMonoFuncs.getpiecewiselines(intervals_y_st, intervals_y_fin, domain_proportion; lb = lb, ub = ub)

f = xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, info, scale)
finv = xx->IntervalMonoFuncs.evalinversepiecewise2Dlinearfunc(xx, info, scale)

x_range = LinRange(lb, ub, 5000)
f_x = f.(x_range)
finv_y = finv.(f_x)

# check for monotonicity.
ZERO_TOL = 1e-12
sanity_check = norm(sort(f_x)-f_x)
println("f_x: monotonicity sanity_check = ", sanity_check)
@assert sanity_check < ZERO_TOL

# check to see if inverse is working.
sanity_check = norm(sort(finv_y)-x_range)
println("finv_y: monotonicity sanity_check = ", sanity_check)
@assert sanity_check < ZERO_TOL

# get the boundary points {(x,y)}.



start_pts, fin_pts, boundary_pts = IntervalMonoFuncs.getboundarypts(intervals_y_st, intervals_y_fin, lb, ub, info, scale)

boundary_xs = collect(boundary_pts[i][1] for i in eachindex(boundary_pts) )
boundary_ys = collect(boundary_pts[i][2] for i in eachindex(boundary_pts) )

focus_interval_coverage_domain, focus_interval_coverage_range = IntervalMonoFuncs.getintervalcoverages(start_pts, fin_pts, lb, ub)
@assert isapprox(focus_interval_coverage_range, sum(intervals_y_fin-intervals_y_st))
@assert isapprox(focus_interval_coverage_domain, domain_proportion*(ub-lb))



PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x_range, f_x)
PyPlot.plot(boundary_xs, boundary_ys, "o", label = "boundary points")

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("f")
PyPlot.title("target warp func")

#### inverse.
PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(f_x, finv_y)

PyPlot.legend()
PyPlot.xlabel("y")
PyPlot.ylabel("x")
PyPlot.title("inverse")



### use monotonic interpolation.

X = LinRange(lb, ub, 50)
Y = f.(X)
f_itp = Interpolations.extrapolate(Interpolations.interpolate(X,
Y, Interpolations.SteffenMonotonicInterpolation()), Interpolations.Flat());


t = LinRange(lb, ub, 5000)
f_itp_t = map(f_itp, t)

sanity_check = norm(sort(f_itp_t)-f_itp_t)
println("f_itp_t: sanity_check = ", sanity_check)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(t, f_itp_t)

PyPlot.legend()
PyPlot.xlabel("t")
PyPlot.ylabel("f_itp")
PyPlot.title("Steffen Monotonic itp")
#PyPlot.axis("scaled")
