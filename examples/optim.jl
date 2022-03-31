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
import NLopt
import Zygote

PyPlot.close("all")
fig_num = 1

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

Random.seed!(25)

# get the oracle single-region piece-wise linear functions, and visualize.
p_lb = 0.0
p_ub = 1.0
window = 0.1
N_itp_samples = 10
input_range_percentage = 0.7 #0.9
infos, zs, p_range = MonotoneMaps.getendomorphismpiecewiselinear(p_lb, p_ub, window; N_itp_samples = N_itp_samples, input_range_percentage = input_range_percentage)

fs = collect( xx->MonotoneMaps.evalpiecewise2Dlinearfunc(xx, infos[i]) for i = 1:length(infos) )

display_t = LinRange(0.0, 1.0, 5000)

PyPlot.figure(fig_num)
fig_num += 1

for i = length(fs):length(fs)
    PyPlot.plot(display_t, fs[i].(display_t), label = "f[$(i)]")
end

PyPlot.xlabel("x")
PyPlot.ylabel("fs")
PyPlot.title("piece-wise linear warp funcs")



# fit the compact sigmoids (composite function of applying probit then logistic functions).
p0 = [0.5; 0.0]
p_lb = [0.1; -5.0]
p_ub = [0.6; 5.0]
optim_algorithm = :LN_BOBYQA
costfuncs, minxs, rets = MonotoneMaps.getcompactsigmoidparameters(infos; p0 = p0, p_lb = p_lb, p_ub = p_ub, optim_algorithm = optim_algorithm)
qs = collect( tt->MonotoneMaps.evalcompositelogisticprobit(tt, minxs[i][1], minxs[i][2]) for i = 1:length(minxs) )


# visualize oracle vs. fitted sigmoids.
PyPlot.figure(fig_num)
fig_num += 1

for l = 1:length(qs)
    PyPlot.plot(display_t, fs[l].(display_t), label = "f[$(l)]")
    PyPlot.plot(display_t, qs[l].(display_t), "--", label = "q[$(l)]")
end

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("")
PyPlot.title("target vs fit")

# print fit cost.
for l = 1:length(costfuncs)
    println("cost of p_star $(l) is ", costfuncs[l](minxs[l]))
end
