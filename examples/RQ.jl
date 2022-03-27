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
fs = collect( xx->(1/sqrt(a[i]+(xx-b[i])^2)^3) for i = 1:length(a) )
int_fs = collect( xx->((xx-b[i])/sqrt(a[i]+(xx-b[i])^2)^3) for i = 1:length(a) )

function evalinvf(y::T, a::T, b::T)::Tuple{T,T} where T <: Real

    term = 1/(sqrt(y)^3) - a

    return b - term, b + term
end

inv_fs = collect( yy->evalinvf(yy, a[i], b[i]) for i = 1:length(a) )



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

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
y = LinRange(-1+1e-5, 1-1e-5, 500)

PyPlot.figure(fig_num)
fig_num += 1

for i = 1:length(inv_fs)
    x1, x2 = unzip(inv_fs[i].(y))

    PyPlot.plot(y, x1, label = "x1, inv f[$(i)]")
end

PyPlot.legend()
PyPlot.xlabel("y")
PyPlot.ylabel("inv f")
PyPlot.title("inverse")


PyPlot.figure(fig_num)
fig_num += 1

for i = 1:length(int_fs)

    PyPlot.plot(x, int_fs[i].(x), label = "int f[$(i)]")
end

PyPlot.legend()
PyPlot.xlabel("y")
PyPlot.ylabel("int f")
PyPlot.title("integral")

@assert 1==2
import NLopt



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
