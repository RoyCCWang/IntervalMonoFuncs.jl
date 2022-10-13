module IntervalMonoFuncs

# Write your package code here.

import NLopt

#import IJulia # for notebook examples.



include("../src/endomorphisms/piece_wise_linear.jl")
include("../src/endomorphisms/composite_sigmoid.jl")

include("../src/fit/fit_to_linear.jl")

include("../src/utils.jl")

export getpiecewiselines,
    evalpiecewise2Dlinearfunc,
    evalinversepiecewise2Dlinearfunc,

    createendopiewiselines1,
    getcompactsigmoidparameters,
    evalcompositelogisticprobit

end
