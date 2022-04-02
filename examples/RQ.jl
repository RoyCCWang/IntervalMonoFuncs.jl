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

# a = [1.0; 20.0]
# b = [1.0; 1.0]
# fs = collect( xx->(1/sqrt(a[i]+(xx-b[i])^2)^3) for i = 1:length(a) )
# int_fs = collect( xx->((xx-b[i])/sqrt(a[i]+(xx-b[i])^2)^3) for i = 1:length(a) )
# function evalinvf(y::T, a::T, b::T)::Tuple{T,T} where T <: Real
#
#     term = 1/(sqrt(y)^3) - a
#
#     return b - term, b + term
# end
# unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
# y = LinRange(-1+1e-5, 1-1e-5, 500)
# #
# PyPlot.figure(fig_num)
# fig_num += 1
# for i = 1:length(inv_fs)
#     x1, x2 = unzip(inv_fs[i].(y))
#
#     PyPlot.plot(y, x1, label = "x1, inv f[$(i)]")
# end
# PyPlot.legend()
# PyPlot.xlabel("y")
# PyPlot.ylabel("inv f")
# PyPlot.title("inverse")

function evalg(x::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T})::T where T <: Real

    out = zero(T)
    for i = 1:length(zs)
        τ = x-zs[i]
        out += w[i]*τ/sqrt(bs[i] + τ^2)
    end

    return out
end

function evalZ(α::T, β::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T})::T where T <: Real

    return evalg(β, w, zs, bs) - evalg(α, w, zs, bs)
end

function evalB(α::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T}, Z::T)::T where T <: Real

    return evalg(α, w, zs, bs)/Z
end

function evalCDF(x::T, B::T, Z::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T})::T where T <: Real
    return evalg(x, w, zs, bs)/Z - B
end

function evalPDF(x::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T}, Z)::T where T <: Real

    return sum( w[n]*bs[n]/sqrt(bs[n]+(x-zs[n])^2 )^3 for n = 1:length(bs) )/Z
end

function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end


lb = -3.0
ub = 4.0

L = 5
zs = collect( convertcompactdomain(rand(), 0.0, 1.0, lb, ub) for l = 1:L )
bs = rand(L) .* 5.0
w = rand(L) .* 7.0

Z = evalZ(lb, ub, w, zs, bs)
B = evalB(lb, w, zs, bs, Z)
CDF_func = xx->evalCDF(xx, B, Z, w, zs, bs)
PDF_func = xx->evalPDF(xx, w, zs, bs, Z)


### visualize.

x = LinRange(lb, ub, 500)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x, CDF_func.(x))


PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("Q inv")
PyPlot.title("CDF")


PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(x, PDF_func.(x))


PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("p(x)")
PyPlot.title("PDF")

###### NiLang goes here.


function evalg2(x::T, w::Vector{T}, zs::Vector{T}, bs::Vector{T})::T where T <: Real

    out = zero(T)
    for i = 1:length(zs)
        τ = x-zs[i]
        out += w[i]*τ/sqrt(bs[i] + τ^2)
    end

    return out
end