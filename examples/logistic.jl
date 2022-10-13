

##### streamline evalf2 ∘ evalf1.

include("../src/IntervalMonoFuncs.jl")
import .IntervalMonoFuncs


q = tt->IntervalMonoFuncs.evalcompositelogisticprobit(tt, 0.25, 0.83, -1.0, 1.0)


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

q_inv = yy->IntervalMonoFuncs.evalinversecompositelogisticprobit(yy, 0.25, 0.83, -1.0, 1.0)


x = LinRange(-1 + 1e-2, 1 - 1e-2, 500)

PyPlot.figure(fig_num)
fig_num += 1

ax = PyPlot.axes()
ax[:set_ylim]([-1,1])
PyPlot.plot(x, q_inv.(x), label = "q_inv")

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("q")
PyPlot.title("evalinversecompositelogisticprobit()")

# check.
@show q_inv(q(-0.94)) 
@assert isapprox(q_inv(q(-0.94)), -0.94)