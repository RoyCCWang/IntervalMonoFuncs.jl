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

z_st = [-0.82; 0.59] # This is {c_l} in the documentation. All elements must be between lb and ub.
z_fin = [0.044; 0.97] # This is {d_l} in the documentation. All elements must be between lb and ub.

# allowed values for lb, ub, -1 <= lb < ub <= 1
lb = -0.9
ub = 1.0
scale = 2.34

# amount of input region used to map to the intervals specified by z_st and z_fin.
input_range_percentage = 0.95

c = input_range_percentage*(ub-lb)

xs, ys, ms, bs, len_s, len_z = IntervalMonoFuncs.getpiecewiselines(z_st, z_fin, c; lb = lb, ub = ub)
f = xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs, scale)
finv = xx->IntervalMonoFuncs.evalinversepiecewise2Dlinearfunc(xx, xs, ys, ms, bs, scale)

x_range = LinRange(lb*scale, ub*scale, 5000)
f_x = f.(x_range)
finv_y = finv.(f_x)

sanity_check = norm(sort(f_x)-f_x)
println("f_x: monotonicity sanity_check = ", sanity_check)

sanity_check = norm(sort(finv_y)-x_range)
println("finv_y: monotonicity sanity_check = ", sanity_check)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x_range, f_x)

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

X = LinRange(lb*scale, ub*scale, 50)
Y = f.(X)
f_itp = Interpolations.extrapolate(Interpolations.interpolate(X,
Y, Interpolations.SteffenMonotonicInterpolation()), Interpolations.Flat());


t = LinRange(lb*scale, ub*scale, 5000)
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
