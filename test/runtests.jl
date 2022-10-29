using IntervalMonoFuncs

using LinearAlgebra
using Test

import Random
Random.seed!(25)


include("./helpers.jl")


val_type = Float64
ZERO_TOL = 1e-10

# val_type = Float32
# ZERO_TOL = 1e-2

# # takes a long time to run. leave for future.
# val_type = BigFloat
# ZERO_TOL = 1e-10

@testset "piecewise-linear" begin

    runpiecewiselineartests(one(val_type);
        N_eval_pts = 5000,
        N_trials = 100,
        N_intervals = 40,
        use_lb_as_start = true, # special case involving boundary.
        ZERO_TOL = ZERO_TOL,
        bound_min = convert(val_type, -200.0),
        bound_max = convert(val_type, 200.0))

    runpiecewiselineartests(one(val_type);
        N_eval_pts = 5000,
        N_trials = 100,
        N_intervals = 5,
        use_lb_as_start = true, # special case involving boundary.
        ZERO_TOL = ZERO_TOL,
        bound_min = convert(val_type, -200.0),
        bound_max = convert(val_type, 200.0))
        
    runpiecewiselineartests(one(val_type);
        N_eval_pts = 5000,
        N_trials = 100,
        N_intervals = 40,
        use_lb_as_start = false,
        ZERO_TOL = ZERO_TOL,
        bound_min = convert(val_type, -200.0),
        bound_max = convert(val_type, 200.0))

    runpiecewiselineartests(one(val_type);
        N_eval_pts = 5000,
        N_trials = 100,
        N_intervals = 5,
        use_lb_as_start = false,
        ZERO_TOL = ZERO_TOL,
        bound_min = convert(val_type, -200.0),
        bound_max = convert(val_type, 200.0))
end

@testset "logistic-probit" begin

    # test different combination of feasible intervals for a and b. See /examples/logistic.jl for motivation.
    runlogistictests(one(val_type);
        N_trials = 10000,
        a_bounds = convert(Vector{val_type}, [0.5, 0.6]),
        b_bounds = convert(Vector{val_type}, [-5.0, 5.0]),
        N_eval_pts = 5000,
        ZERO_TOL = ZERO_TOL)

    runlogistictests(one(val_type);
        N_trials = 10000,
        a_bounds = convert(Vector{val_type}, [-2.0, 2.0]),
        b_bounds = convert(Vector{val_type}, [-2.0, 2.0]),
        N_eval_pts = 5000,
        ZERO_TOL = ZERO_TOL)

    runlogistictests(one(val_type);
        N_trials = 10000,
        a_bounds = convert(Vector{val_type}, [0.01, 1.0]),
        b_bounds = convert(Vector{val_type}, [-5.0, 5.0]),
        N_eval_pts = 5000,
        ZERO_TOL = ZERO_TOL)
end


@testset "createendopiewiselines1()" begin
    
    # test domain and range coverage of the created endomorphic piecewise-linear functions.
    # max_N_itp_samples is the number of piecewise-linear functions to generate on the domain [0,1].
    runcreateendopiewiselines1tests(;
        val_type = val_type,
        N_trials = 10000,
        max_N_itp_samples = 100,
        ZERO_TOL = ZERO_TOL)

end