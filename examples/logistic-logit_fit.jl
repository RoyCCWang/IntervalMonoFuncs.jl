# In this example, we create piece-wise linear functions that has a single line segment that has a small slope. The interpretation of using such a function as a transport map is to move mass from a large coverage to a concentrated coverage. The intervals in the domain and range that correspond to this segment is the *focus interval*, and is controlled by `range_percentage` and `domain_percentage`. We then fit a logistic-logit function to each of the piece-wise linear functions. The fitted parameters can then be interpolated so that we get a logistic-logit function for any given focus interval.

include("../src/MonotoneMaps.jl")
import .MonotoneMaps
#import MonotoneMaps

import PyPlot # works better in REPL.

#using Plots; pyplot() # easier to work with in .jmd / Weave.jl.

import Random
using LinearAlgebra


PyPlot.close("all")
fig_num = 1

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

Random.seed!(25)

# Get the set of piece-wise linear functions.
p_lb = 0.0
p_ub = 1.0
range_percentage = 0.1
N_itp_samples = 10
domain_percentage = 0.7
infos, zs, p_range = MonotoneMaps.getendomorphismpiecewiselinear(p_lb, p_ub, range_percentage; N_itp_samples = N_itp_samples, domain_percentage = domain_percentage)

# construct the set of piece-wise functions.
fs = collect( xx->MonotoneMaps.evalpiecewise2Dlinearfunc(xx, infos[i]) for i = 1:length(infos) )

# visualize.
display_t = LinRange(0.0, 1.0, 5000)

PyPlot.figure(fig_num)
fig_num += 1

for i = length(fs):length(fs)
    PyPlot.plot(display_t, fs[i].(display_t), label = "f[$(i)]")
end

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("y")
PyPlot.title("Piece-wise linear functions, each with a single focus subinterval")

# display_mat = [ fs[c](display_t[r]) for r = 1:length(display_t), c = 1:length(fs) ]
# display_labels = ["$(i)-th function" for j = 1:1, i = 1:length(fs)]
#
# default(titlefont = (20, "times"), legendfontsize = 15, guidefont = (18, :black),
# tickfont = (12, :black))
#
# plot_handle = plot(display_t, display_mat,
# label = display_labels,
# title = "Piece-wise linear functions, each with a single focus subinterval",
# xlabel = "x",
# ylabel = "y",
# linewidth = 2, legend = :outerright, aspect_ratio=:equal, size = (800,800))
# display(plot_handle)


# fit the compact sigmoids (composite function of applying probit then logistic functions).
p0 = [0.5; 0.0]
p_lb = [0.1; -5.0]
p_ub = [0.6; 5.0]
optim_algorithm = :LN_BOBYQA # a local derivative-free algorithm. For other algorithms in NLopt, see https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
costfuncs, minxs, rets = MonotoneMaps.getcompactsigmoidparameters(infos; p0 = p0, p_lb = p_lb, p_ub = p_ub, optim_algorithm = optim_algorithm)
qs = collect( tt->MonotoneMaps.evalcompositelogisticprobit(tt, minxs[i][1], minxs[i][2]) for i = 1:length(minxs) )


## visualize oracle vs. fitted sigmoids.
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

# display_mat = [ fs[c](display_t[r]) for r = 1:length(display_t), c = 1:length(fs) ]
# display_labels = ["$(i)-th piece-wise linear" for j = 1:1, i = 1:length(fs)]
#
# display_mat2 = [ qs[c](display_t[r]) for r = 1:length(display_t), c = 1:length(fs) ]
# display_labels2 = ["$(i)-th logistic-logit" for j = 1:1, i = 1:length(qs)]
#
# plot_handle2 = plot(display_t, display_mat,
# label = display_labels,
# title = "Piece-wise linear and logistic-logit functions",
# xlabel = "x",
# ylabel = "y",
# linewidth = 2, legend = :outerright, aspect_ratio=:equal, size = (800,800))
#
# plot!(display_t, display_mat2,
# label = display_labels2,
# style = :dash)
#
# display(plot_handle2)

# print fit cost.
for l = 1:length(costfuncs)
    println("cost of p_star $(l) is ", costfuncs[l](minxs[l]))
end
