
include("../src/IntervalMonoFuncs.jl")
import .IntervalMonoFuncs
#import IntervalMonoFuncs

using LinearAlgebra
using Test

import Random
Random.seed!(25)


include("./helpers.jl")




@testset "piecewise-linear" begin

    runpiecewiselineartests(;
        val_type = Float64,
        N_eval_pts = 5000,
        N_trials = 100,
        N_intervals = 40,
        use_lb_as_start = true, # special case involving boundary.
        ZERO_TOL = 1e-10)

    runpiecewiselineartests(;
        val_type = Float64,
        N_eval_pts = 5000,
        N_trials = 100,
        N_intervals = 5,
        use_lb_as_start = true, # special case involving boundary.
        ZERO_TOL = 1e-10)
        
    runpiecewiselineartests(;
        val_type = Float64,
        N_eval_pts = 5000,
        N_trials = 100,
        N_intervals = 40,
        use_lb_as_start = false,
        ZERO_TOL = 1e-10)

    runpiecewiselineartests(;
        N_eval_pts = 5000,
        N_trials = 100,
        N_intervals = 5,
        use_lb_as_start = false,
        ZERO_TOL = 1e-10)
end

@testset "logistic-probit" begin

    # test different combination of feasible intervals for a and b. See /examples/logistic.jl for motivation.
    runlogistictests(;
        val_type = Float64,
        N_trials = 10000,
        a_bounds = [0.5, 0.6],
        b_bounds = [-5.0, 5.0],
        N_eval_pts = 5000,
        ZERO_TOL = 1e-10)

    runlogistictests(;
        val_type = Float64,
        N_trials = 10000,
        a_bounds = [-2.0, 2.0],
        b_bounds = [-2.0, 2.0],
        N_eval_pts = 5000,
        ZERO_TOL = 1e-10)

    runlogistictests(;
        val_type = Float64,
        N_trials = 10000,
        a_bounds = [0.01, 1.0],
        b_bounds = [-5.0, 5.0],
        N_eval_pts = 5000,
        ZERO_TOL = 1e-10)
end


@testset "createendopiewiselines1()" begin
    
    # test domain and range coverage of the created endomorphic piecewise-linear functions.
    # max_N_itp_samples is the number of piecewise-linear functions to generate on the domain [0,1].
    runcreateendopiewiselines1tests(;
        val_type = Float64,
        N_trials = 10000,
        max_N_itp_samples = 100,
        ZERO_TOL = 1e-10)

end