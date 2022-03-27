# explore warping functions.

include("../src/MonotoneMaps.jl")
import .MonotoneMaps

using FFTW
import PyPlot
import BSON
import Optim
import Random
using LinearAlgebra

import Interpolations

PyPlot.close("all")
fig_num = 1

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

Random.seed!(25)


# # gives the identity map.
# D = 2 #5 #2
# lb = -1
# ub = 1
# scale = 1.0 # 0.01
# c = 0.5*(ub-lb) # amount of input region used to map to z_lens.
# z = [-0.5; 0.5]
# z_lens = ones(D) .* 0.5

## gives a random map.
D = 5 #2 # number of regions.
lb = -1
ub = 1
scale = 1.0 #0.5

# amount of input region used to map to z_lens.
input_range_percentage = 0.95
c = input_range_percentage*(ub-lb)

# generate random boundary points that define the regions.
rand_gen_func = xx->MonotoneMaps.convertcompactdomain(xx, 0.0, 1.0, scale*lb, scale*ub)
z = sort(collect(rand_gen_func(rand()) for d = 1:D))

# sums to c.
z_lens = ones(D) .* 2.0 #0.2
z_lens = z_lens .* (c/sum(z_lens))

# process z points.
z_pts = collect( [z[i]-z_lens[i]/2; z[i]+z_lens[i]/2] for i = 1:D )
z_st0 = collect( z_pts[i][1] for i = 1:length(z_pts) )
z_fin0 = collect( z_pts[i][2] for i = 1:length(z_pts) )

# combine intervals such that they are not overlapping.
z_st, z_fin = MonotoneMaps.mergesuchthatnooverlap(z_st0, z_fin0)
# z_st is start of segment, in domain's coordinate.
# z_fin is end of segment, in range's coordinate.

xs, ys, ms, bs, len_s, len_z = MonotoneMaps.getpiecewiselines(z_st, z_fin, c)
f = xx->MonotoneMaps.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)*scale
finv = xx->MonotoneMaps.evalinversepiecewise2Dlinearfunc(xx/scale, xs, ys, ms, bs)

x_range = LinRange(lb, ub, 5000)
f_x = f.(x_range)
finv_y = finv.(f_x)

sanity_check = norm(sort(f_x)-f_x)
println("f_x: sanity_check = ", sanity_check)

sanity_check = norm(sort(finv_y)-x_range)
println("finv_y: sanity_check = ", sanity_check)

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
PyPlot.title("SteffenMonoton itp")
#PyPlot.axis("scaled")
