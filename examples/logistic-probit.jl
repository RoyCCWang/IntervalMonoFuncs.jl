

##### streamline evalf2 ∘ evalf1.

include("../src/IntervalMonoFuncs.jl")
import .IntervalMonoFuncs

import PyPlot

import Random
using LinearAlgebra


PyPlot.close("all")
fig_num = 1

PyPlot.matplotlib["rcParams"][:update](["font.size" => 22, "font.family" => "serif"])

Random.seed!(25)

### parameters.

a = 0.25 # the sign of a dictates if q and q_inv is monotonically increasing (positive sign)  or decreasing (negative sign).
b = 0.83
lb = -1.0 # must be greater than or equal to -1.
ub = 1.0 # must be less than or equal to 1.

# combinations of large magnitudes of a and b seem to have numerically issues. It might base to reach ZERO_TOL in the check at the end of this example script.
# We recommend a and b to satisfy one of the following scenarios:
# * both in [-2,2]
# * a in [0.1, 0.6] and b in [-5, 5].
# use the testbench function runlogistictests() in /test/ to explore other combinations for custom ZERO_TOL and N_eval_pts than the ones here.


### forward evaluation.
q = tt->IntervalMonoFuncs.evalcompositelogisticprobit(tt, a, b, lb, ub)

#x = LinRange(lb + 1e-2, ub - 1e-2, 5000)
x = LinRange(lb, ub, 5000)

PyPlot.figure(fig_num)
fig_num += 1

ax = PyPlot.axes()
ax[:set_ylim]([-1,1])
PyPlot.plot(x, q.(x), label = "q")

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("q")
PyPlot.title("evalcompositelogisticprobit()")



### inverse.
q_inv = yy->IntervalMonoFuncs.evalinversecompositelogisticprobit(yy, a, b, lb, ub)


#x = LinRange(lb + 1e-2, ub - 1e-2, 5000)
x = LinRange(lb, ub, 5000)

PyPlot.figure(fig_num)
fig_num += 1

ax = PyPlot.axes()
ax[:set_ylim]([-1,1])
PyPlot.plot(x, q_inv.(x), label = "q_inv")

PyPlot.legend()
PyPlot.xlabel("x")
PyPlot.ylabel("q")
PyPlot.title("evalinversecompositelogisticprobit()")

### numerical check on inverse.
ZERO_TOL = 1e-10
@show norm(q_inv.(q.(x)) - x)
@assert norm(q_inv.(q.(x)) - x) < ZERO_TOL