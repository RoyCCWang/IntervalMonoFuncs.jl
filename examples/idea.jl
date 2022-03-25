# explore warping functions.

# include("../src/MonotoneMaps.jl")
# import .MonotoneMaps

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

a = [1.0; 20.0]
b = [1.0; 1.0]
fs = collect( xx->((xx-b[i])/sqrt(a[i]+(xx-b[i])^2)) for i = 1:length(a) )

function evalinvf(y::T, a::T, b::T)::T where T <: Real
    A = b*y^2-a
    B = y^2-1
    C = sqrt(y^2*(a^2-2*a*b-a*y^2+a+b^2))
    D = y^2-1

    term1 = A/B
    term2 = C/D

    return term1+term2, term1+term2
end

invfs = collect( yy->evalinvf(yy, a[i], b[i]) for i = 1:length(a) )



x = LinRange(-3.0, 3.0, 500)

PyPlot.figure(fig_num)
fig_num += 1

for i = 1:length(fs)
    PyPlot.plot(x, fs[i].(x), label = "f[$(i)]")
end

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("f")
PyPlot.title("target warp func")


y = LinRange(-1+1e-5, 1-1e-5, 500)

PyPlot.figure(fig_num)
fig_num += 1

for i = 1:length(fs)
    PyPlot.plot(y, invfs[i].(y), label = "f[$(i)]")
end

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("f")
PyPlot.title("target warp func")

@assert 1==2
import NLopt

function runNLopt!(  opt,
    p0::Vector{T},
    obj_func,
    grad_func,
    p_lbs,
    p_ubs;
    max_iters = 10000,
    xtol_rel = 1e-12,
    ftol_rel = 1e-12,
    maxtime = Inf) where T

    @assert length(p0) == length(p_lbs) == length(p_ubs)

    opt.maxeval = max_iters
    opt.lower_bounds = p_lbs
    opt.upper_bounds = p_ubs
    opt.xtol_rel = xtol_rel
    opt.ftol_rel = ftol_rel
    opt.maxtime = maxtime


    opt.min_objective = (xx, gg)->genericNLoptcostfunc!(xx, gg, obj_func, grad_func)

    # optimize.
    (minf, minx, ret) = NLopt.optimize(opt, p0)

    N_evals = opt.numevals

    return minf, minx, ret, N_evals
end

function genericNLoptcostfunc!(x::Vector{T}, df_x, f, df)::T where T <: Real

    if length(df_x) > 0
        df_x[:] = df(x)
    end

    return f(x)
end

function evalalgebraicsigmoid(x, a, b)
    return (x[1]-b[1])/sqrt(a[1]+(x[1]-b[1])^2)
end

a = [1.0;]
b = [0.0;]

f = xx->evalalgebraicsigmoid(xx, a, b)

domain_lb = 0.5
g = aa->(evalalgebraicsigmoid(-1.0, aa, b)-range_lb)
# using ChainRules
# function evaldf(x)
#
#     a, a_pullback = rrule(sin, x);
#     b, b_pullback = rrule(+, 0.2, a);
#     c, c_pullback = rrule(asin, b)
#
#     #### Then the backward pass calculating gradients
#     c̄ = 1;                    # ∂c/∂c
#     _, b̄ = c_pullback(c̄);     # ∂c/∂b = ∂c/∂b ⋅ ∂c/∂c
#     _, _, ā = b_pullback(b̄);  # ∂c/∂a = ∂c/∂b ⋅ ∂b/∂a
#     _, x̄ = a_pullback(ā);     # ∂c/∂x = ∂c/∂a ⋅ ∂a/∂x
#     return x̄                         # ∂c/∂x = ∂foo/∂x
# end

import Zygote

dg = xx->Zygote.gradient(f, xx)

p0 = ones(1)
p_lb = [1e-5;]
p_ub = [50.0;]

optim_algorithm = :GN_ESCH
#optim_algorithm = :LN_BOBYQA

opt = NLopt.Opt(optim_algorithm, 1)

minf, minx, ret, N_evals = runNLopt!(opt,
    p0,
    g,
    dg,
    p_lb,
    p_ub;
    max_iters = 5000,
    xtol_rel = 1e-5,
    ftol_rel = 1e-5,
    maxtime = Inf)
