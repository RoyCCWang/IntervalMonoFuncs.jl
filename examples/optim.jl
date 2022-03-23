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


unity_obj = MonotoneMaps.getunityobj(1.0, Val(:PiecewiseLinear))
f = xx->MonotoneMaps.evalpiecewise2Dlinearfunc(xx, unity_obj)
finv = yy->MonotoneMaps.evalinversepiecewise2Dlinearfunc(yy, unity_obj)


lb = -1
ub = 1

x_range = LinRange(lb, ub, 500)
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
