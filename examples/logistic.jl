# explore warping functions.



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

a = [0.5; 5.0; 1/5] # higher makes sharper transition.
b = [1.0; 1.0; 1.0] .*1

c = [1.0; 1.0; 1.0] # higher makes sharper transition.
d = [0.0; 0.0; 0.0]

fs = collect( xx->( evalf2(evalf1(xx, a[i], b[i]), c[i], c[i]) ) for i = 1:length(a) )
#fs = collect( xx->( evalf1(0.5*evalf2(xx, a[i], b[i])+0.5, c[i], c[i]) ) for i = 1:length(a) )

function evalf1(x::T, a, b)::T where T
    return a*log(x/(1-x))+b
end

## algebraic sigmoid function. f: ℝ → [-1, 1]
# function evalf2(x::T, a, b)::T where T
#     return (x-b)/sqrt(a+(x-b)^2)
# end

## logistic function. f: ℝ → [-1, 1]
function evalf2(x::T, a, b)::T where T
    return 1/(1+exp(-a*(x-b)))
end

x = LinRange(0 + 1e-2, 1 - 1e-2, 500)
#x = LinRange(-30 , 30, 500)

PyPlot.figure(fig_num)
fig_num += 1

for i = 1:length(fs)
    PyPlot.plot(x, fs[i].(x), label = "f[$(i)]")
end

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("f")
PyPlot.title("target warp func")


##### streamline evalf2 ∘ evalf1.

include("../src/MonotoneMaps.jl")
import .MonotoneMaps


q = tt->MonotoneMaps.evalcompositelogisticprobit(tt, 0.25, 0.83, -1.0, 1.0)


x = LinRange(-1 + 1e-2, 1 - 1e-2, 500)

PyPlot.figure(fig_num)
fig_num += 1

ax = PyPlot.axes()
ax[:set_ylim]([-1,1])
PyPlot.plot(x, q.(x), label = "q")

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("q")
PyPlot.title("evalcompositelogisticprobit()")



## inverse.