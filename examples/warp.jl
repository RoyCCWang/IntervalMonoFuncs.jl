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

## gives a random map.
D = 1 #2 # number of regions.
lb = 0.0
ub = 1.0
scale = 1.0 #0.5

# amount of input region used to map to z_lens.
input_range_percentage = 0.9
c = input_range_percentage*(ub-lb)

# generate random boundary points that define the regions.

# # for lb = -1.0
# z_st = [ -0.12; ]
# z_fin = [ 0.76; ]

# for lb = 0.0
# z_st = [ 0.12; ]
# z_fin = [ 0.76; ]

# z_st = [ 0.1; ]
# z_fin = [ 0.3; ]
#
#
# z_st = [ 0.6; ]
# z_fin = [ 0.8; ]

# z_st = [ -0.999999999999; ]
# z_fin = [ -0.9; ]
#
# z_st = [ 0.9; ]
# z_fin = [ 0.999; ]

xs, ys, ms, bs, len_s, len_z = MonotoneMaps.getpiecewiselines(z_st, z_fin, c; lb = lb, ub = ub)
f = xx->MonotoneMaps.evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)*scale
finv = xx->MonotoneMaps.evalinversepiecewise2Dlinearfunc(xx/scale, xs, ys, ms, bs)

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
