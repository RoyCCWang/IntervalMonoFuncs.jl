# explore warping functions.

include("../src/IntervalMonoFuncs.jl")
import .IntervalMonoFuncs

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

## gives a random map.
lb = -1.0
ub = 1.0
scale = 1.0

# amount of input region used to map to z_lens.
input_range_percentage = 0.9
c = input_range_percentage*(ub-lb)

# generate random boundary points that define the regions.

# for lb = -1.0
z_st = [ -0.12; ]
z_fin = [ 0.76; ]

xs, ys, ms, bs, len_s, len_z = IntervalMonoFuncs.getpiecewiselines(z_st, z_fin, c; lb = lb, ub = ub)
f = xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)*scale
finv = xx->IntervalMonoFuncs.evalinversepiecewise2Dlinearfunc(xx/scale, xs, ys, ms, bs)

x_range = LinRange(lb, ub, 5000)
f_x = f.(x_range)
finv_y = finv.(f_x)

sanity_check = norm(sort(f_x)-f_x)
println("f_x: sanity_check = ", sanity_check)
@assert sanity_check < 1e-12

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x_range, f_x)

PyPlot.xlabel("x")
PyPlot.ylabel("f")
PyPlot.title("target warp func")
