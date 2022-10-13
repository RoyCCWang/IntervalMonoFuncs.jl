### routines for fitting a composite sigmoid to piece-wise linear functions.

function evalcompositelogisticprobitcost(t_range, f::Function, p::Vector{T})::T where T

    s = zero(T)
    for (i,t) in enumerate(t_range)
        s += ( f(t) - evalcompositelogisticprobit(t, p[1], p[2]) )^2
    end

    return s
end

"""
    getcompactsigmoidparameters(infos::Vector{IntervalMonoFuncs.Piecewise2DLineType{T}};
        N_fit_positions::Int = 15,
        max_iters = 5000,
        xtol_rel = 1e-5,
        ftol_rel = 1e-5,
        maxtime = Inf,
        p0 = [0.5; 0.0],
        p_lb = [0.1; -5.0],
        p_ub = [0.6; 5.0]

Given a set of single region piece-wise linear function parameter objects, infos, fit the `a` and `b` parameters of the compact sigmoid (see function `evalcompositelogisticprobit`) for each piece-wise linear function.

∀ i ∈ [length(infos)], infos[i].xs must be 4 elements, with the first and last being the boundary, and the second and third being the end points of the single region.

p0, p_lb, p_ub are two element 1-D arrays.
...
# Select arguments
Two-element 1-D array for the following. The first entry relates to the a parameter, and the second relates to the b parameter.
- `p0::Vector{T}`: initial guess to the optimization.
- `p_lb::Vector{T}`: lower bounds to the optimization.
- `p_ub::Vector{T}`: upper bounds to the optimization.
- `optim_algorithm::Symbol` can be :GN_ESCH, :GN_ISRES, :LN_BOBYQA, :GN_DIRECT_L. Must be a derivative-free NLopt optimization method. See the NLopt library website for more details about these algorithms.

...


See optim.jl in the examples folder for usage details.
"""
function getcompactsigmoidparameters(infos::Vector{IntervalMonoFuncs.Piecewise2DLineType{T}};
    N_fit_positions::Int = 15,
    max_iters = 5000,
    xtol_rel = 1e-5,
    ftol_rel = 1e-5,
    maxtime = Inf,
    optim_algorithm::Symbol = :GN_ESCH,
    p0 = [0.5; 0.0],
    p_lb = [0.1; -5.0],
    p_ub = [0.6; 5.0],
    evalcostfunc = evalcompositelogisticprobitcost) where T

    

    fs = collect( xx->evalpiecewise2Dlinearfunc(xx, infos[i]) for i in eachindex(infos) )

    L = length(fs)
    gs = Vector{Function}(undef, L)
    p_star_set = Vector{Vector{T}}(undef, L)
    rets = Vector{Symbol}(undef, L)

    for l = 1:L

        t_range = LinRange(infos[l].xs[2], infos[l].xs[3], N_fit_positions)

        g = pp->evalcostfunc(t_range, fs[l], pp)
        #dg = xx->Zygote.gradient(f, xx)
        dg = xx->xx # we're not using gradient-based optimization.


        opt = NLopt.Opt(optim_algorithm, length(p0))

        minf, minx, ret, N_evals = runNLopt!(opt,
            p0,
            g,
            dg,
            p_lb,
            p_ub;
            max_iters = max_iters,
            xtol_rel = xtol_rel,
            ftol_rel = ftol_rel,
            maxtime = maxtime)
        # update.
        p_star_set[l] = minx
        gs[l] = g
        rets[l] = ret
    end

    return gs, p_star_set, rets
end


function getlogisticprobitparameters(infos::Vector{IntervalMonoFuncs.Piecewise2DLineType{T}};
    N_fit_positions::Int = 15,
    max_iters = 5000,
    xtol_rel = 1e-5,
    ftol_rel = 1e-5,
    maxtime = Inf,
    optim_algorithm::Symbol = :GN_ESCH,
    a_lb::T = 0.1,
    a_ub::T = 0.6,
    b_lb::T = -5.0,
    b_ub::T = 5.0,
    a_initial = (a_ub-a_lb)/2,
    b_initial = (b_ub-b_lb)/2) where T <: Real

    @assert a_lb <= a_initial <= a_ub
    @assert b_lb <= b_initial <= b_ub

    return getcompactsigmoidparameters(infos,
        N_fit_positions = N_fit_positions,
        max_iters = max_iters,
        xtol_rel = xtol_rel,
        ftol_rel = ftol_rel,
        maxtime = maxtime,
        optim_algorithm = optim_algorithm,
        p0 = [a_initial; b_initial],
        p_lb = [a_lb; b_lb],
        p_ub = [a_ub; b_ub],
        evalcostfunc = evalcompositelogisticprobitcost)
end