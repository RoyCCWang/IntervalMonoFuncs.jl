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

a = [2; 5] # higher makes sharper transition.
b = [0.0; 3.0]

function evalf(x::T, a::Int, b::T)::T where T <: Real
    # term = 1 + 1/(1/(x-b) - 1)^a
    #
    # return 1- 1/term

    h = log(1/(x-b)-1)

    return 1 - 1/(1+exp(-a*h))
end

function evalinvf(y::T, a::T, b::T)::T where T <: Real

    term = a*b + (a+b)*y + y

    return term/(a*y)
end

fs = collect( xx->( evalf(xx, a[i], b[i]) ) for i = 1:length(a) )

inv_fs = collect( yy->evalinvf(yy, a[i], b[i]) for i = 1:length(a) )

g = xx->inv_fs[2](fs[1](xx))

x = LinRange(1e-5, 1.0-1e-5, 500)

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
y = LinRange(1e-5, 1-1e-5, 500)

PyPlot.figure(fig_num)
fig_num += 1

for i = 1:length(inv_fs)

    PyPlot.plot(y, inv_fs[i].(y), label = "inv f[$(i)]")
end

PyPlot.legend()
PyPlot.xlabel("y")
PyPlot.ylabel("inv f")
PyPlot.title("inverse")
