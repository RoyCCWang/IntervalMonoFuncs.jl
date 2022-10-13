# In this example, we create piece-wise linear functions that has a single line segment that has a small slope. The interpretation of using such a function as a transport map is to move mass from a large coverage to a concentrated coverage. The intervals in the domain and range that correspond to this segment is the *focus interval*, and is controlled by `range_proportion` and `domain_proportion`. We then fit a logistic-logit function to each of the piece-wise linear functions. The fitted parameters can then be interpolated so that we get a logistic-logit function for any given focus interval.

include("../src/IntervalMonoFuncs.jl")
import .IntervalMonoFuncs
#import IntervalMonoFuncs

import PyPlot # works better in REPL.

#using Plots; pyplot() # easier to work with in .jmd / Weave.jl.

import Random
using LinearAlgebra


PyPlot.close("all")
fig_num = 1

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

Random.seed!(25)

# Get the set of piece-wise linear functions.
p_lb = 0.0 # logistic-logit maps [0,1]->[0,1], so we use 0 and 1 for our bounds.
p_ub = 1.0

# 0.9 is very poor for logistic-logit. try logit-logistic instead.
range_proportion = 0.1 # the focus interval should have this much coverage on the range (vertical axis). In proportion units, i.e., 0 to 1.
N_itp_samples = 10
domain_proportion = 0.7 # the focus interval should have this much coverage on the domain (horizontal axis). In proportion units, i.e., 0 to 1.
infos, zs, p_range = IntervalMonoFuncs.getendomorphismpiecewiselinear(p_lb, p_ub, range_proportion; N_itp_samples = N_itp_samples, domain_proportion = domain_proportion)

# construct the set of piece-wise functions.
fs = collect( xx->IntervalMonoFuncs.evalpiecewise2Dlinearfunc(xx, infos[i]) for i in eachindex(infos) )

# visualize.
display_t = LinRange(p_lb, p_ub, 5000)

PyPlot.figure(fig_num)
fig_num += 1

for i in eachindex(fs)
    @show i
    PyPlot.plot(display_t, fs[i].(display_t), label = "f[$(i)]")
end

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("y")
PyPlot.title("Piece-wise linear functions, each with a single focus subinterval")

# display_mat = [ fs[c](display_t[r]) for r in eachindex(display_t), c in eachindex(fs) ]
# display_labels = ["$(i)-th function" for j = 1:1, i in eachindex(fs)]
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
costfuncs, minxs, rets = IntervalMonoFuncs.getcompactsigmoidparameters(infos; p0 = p0, p_lb = p_lb, p_ub = p_ub, optim_algorithm = optim_algorithm)
qs = collect( tt->IntervalMonoFuncs.evalcompositelogisticprobit(tt, minxs[i][1], minxs[i][2]) for i in eachindex(minxs) )


## visualize oracle vs. fitted sigmoids.
PyPlot.figure(fig_num)
fig_num += 1

for l in eachindex(qs)
    PyPlot.plot(display_t, fs[l].(display_t), label = "f[$(l)]")
    PyPlot.plot(display_t, qs[l].(display_t), "--", label = "q[$(l)]")
end

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("")
PyPlot.title("target vs fit")

# display_mat = [ fs[c](display_t[r]) for r in eachindex(display_t), c in eachindex(fs) ]
# display_labels = ["$(i)-th piece-wise linear" for j = 1:1, i in eachindex(fs)]
#
# display_mat2 = [ qs[c](display_t[r]) for r in eachindex(display_t), c in eachindex(fs) ]
# display_labels2 = ["$(i)-th logistic-logit" for j = 1:1, i in eachindex(qs)]
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
for l in eachindex(costfuncs)
    println("cost of p_star $(l) is ", costfuncs[l](minxs[l]))
end



#### transport uniform distribution.

function applytransport(f::Function, N_samples::Int)
    X = rand(N_samples)
    return X, collect( f(X[i]) for i in eachindex(X) )
end

N_bins = 300
N_his_samples = 5000


function plothistograms(qs, fs, N_his_samples, N_bins, fig_num;
    transparency_alpha = 0.5)

    for l in eachindex(qs)

        PyPlot.figure(fig_num)
        fig_num += 1

        X, Y = applytransport(fs[l], N_his_samples)
        PyPlot.hist(Y, N_bins, label = "f[$(l)]", alpha = transparency_alpha)

        X, Y = applytransport(qs[l], N_his_samples)
        PyPlot.hist(Y, N_bins, label = "q[$(l)]", alpha = transparency_alpha)

        PyPlot.legend()
        PyPlot.xlabel("x")
        PyPlot.ylabel("")
        PyPlot.title("transported samples for the $l-th function. f is piecewise-linear, q is logistic-logit")
    end

    return fig_num
end

fig_num = plothistograms(qs, fs, N_his_samples, N_bins, fig_num)

# inverse transport map, back to uniform. make as test bench for inverse picelinear and logistic-logit.
#  next, show case interpolation to get any desired focus range center.

# then write test that assures inverse is correct.

# merge logistic.jl with the logistic-logit_fit.jl example.

# write tests, code coverage and build setup, citation, then last documentation.


# first, testbench of getendomorphismpiecewiselinear() based triangle area calculations and zs, p_range.