"""
    convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T

converts compact domain x ∈ [a,b] to compact domain out ∈ [c,d].
"""
function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end

function convertcompactdomain(x::Vector{T}, a::T, b::T, c::T, d::T)::Vector{T} where T <: Real

    return collect( convertcompactdomain(x[i], a, b, c, d) for i = 1:length(x) )
end

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))



##### experimental.

function intervalsigmoid(x::T, a, b)::T where T <: Real

    return 1/(1 + exp(-a*(log(x/(1-x))-b)))
end

function prepareboxboundwarping(p_lb::T, p_ub::T, window::T;
    N_itp_samples::Int = 10,
    input_range_percentage::T = 0.9,
    scale::T = 1.0) where T

    #ϵ = window/100

    #p_range = LinRange(p_lb + window + ϵ, p_ub - window - ϵ, N_itp_samples)
    p_range = LinRange(p_lb + window, p_ub - window, N_itp_samples)

    lb = zero(T)
    ub = one(T)
    c = input_range_percentage*(ub-lb)

    fs = Vector{Function}(undef, length(p_range))
    infos = Vector{Piecewise2DLineType{T}}(undef, length(p_range))
    zs = Vector{Vector{T}}(undef, length(p_range))

    for (i,p) in enumerate(p_range)

        z_st = [p - window;]
        z_fin = [p + window;]

        xs, ys, ms, bs, len_s, len_z = getpiecewiselines(z_st, z_fin, c; lb = lb, ub = ub)
        fs[i] = xx->evalpiecewise2Dlinearfunc(xx, xs, ys, ms, bs)*scale

        infos[i] = Piecewise2DLineType(xs, ys, ms, bs, len_s, len_z)
        zs[i] = [z_st[1]; z_fin[1]]
    end

    return fs, infos, zs
end


function runNLopt!(  opt,
    p0::Vector{T},
    obj_func,
    grad_func,
    p_lbs,
    p_ubs;
    max_iters = 10000,
    xtol_rel = 1e-12,
    ftol_rel = 1e-12,
    maxtime = Inf) where T

    @assert length(p0) == length(p_lbs) == length(p_ubs)

    opt.maxeval = max_iters
    opt.lower_bounds = p_lbs
    opt.upper_bounds = p_ubs
    opt.xtol_rel = xtol_rel
    opt.ftol_rel = ftol_rel
    opt.maxtime = maxtime


    opt.min_objective = (xx, gg)->genericNLoptcostfunc!(xx, gg, obj_func, grad_func)

    # optimize.
    (minf, minx, ret) = NLopt.optimize(opt, p0)

    N_evals = opt.numevals

    return minf, minx, ret, N_evals
end

function genericNLoptcostfunc!(x::Vector{T}, df_x, f, df)::T where T <: Real

    if length(df_x) > 0
        df_x[:] = df(x)
    end

    return f(x)
end

function costfunc(t_range, f, p::Vector{T})::T where T

    s = zero(T)
    for (i,t) in enumerate(t_range)
        s += ( f(t) - intervalsigmoid(t, p[1], p[2]) )^2
    end

    return s
end


function getcompactsigmoidparameters(fs, infos::Vector{MonotoneMaps.Piecewise2DLineType{T}};
    M::Int = 15,
    max_iters = 5000,
    xtol_rel = 1e-5,
    ftol_rel = 1e-5,
    maxtime = Inf,
    p0 = [0.5; 0.0],
    p_lb = [0.1; -5.0],
    p_ub = [0.6; 5.0]) where T

    #
    L = length(fs)
    gs = Vector{Function}(undef, L)
    p_star_set = Vector{Vector{T}}(undef, L)
    rets = Vector{Symbol}(undef, L)

    for l = 1:L

        t_range = LinRange(infos[l].xs[2], infos[l].xs[3], M)

        g = pp->costfunc(t_range, fs[l], pp)
        dg = xx->Zygote.gradient(f, xx)

        optim_algorithm = :GN_ESCH
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
