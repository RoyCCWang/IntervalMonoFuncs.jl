# Copyright (c) 2022 Roy Wang

# Exhibit A - Source Code Form License Notice
# -------------------------------------------

#   This Source Code Form is subject to the terms of the Mozilla Public
#   License, v. 2.0. If a copy of the MPL was not distributed with this
#   file, You can obtain one at http://mozilla.org/MPL/2.0/.

# If it is not possible or desirable to put the notice in a particular
# file, then You may include the notice in a location (such as a LICENSE
# file in a relevant directory) where a recipient would be likely to look
# for such a notice.

# You may add additional accurate notices of copyright ownership.


module IntervalMonoFuncs


include("../src/endomorphisms/piece_wise_linear.jl")
include("../src/endomorphisms/composite_sigmoid.jl")

include("../src/fit/fit_to_linear.jl")

include("../src/utils.jl")

export getpiecewiselines,
    checkzstfin,
    evalpiecewise2Dlinearfunc,
    evalinversepiecewise2Dlinearfunc,

    createendopiewiselines1,
    getlogisticprobitparameters,
    evalcompositelogisticprobit,
    evalinversecompositelogisticprobit

# internal functions are not intended for external use.
end
