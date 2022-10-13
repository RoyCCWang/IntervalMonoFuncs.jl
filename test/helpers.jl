"""
convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real
returns (x-a)*(d-c)/(b-a)+c

converts x ∈ [a,b] to y ∈ [c,d].
"""
function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end

function generatestartfinishpts(N_intervals::Int, lb::T, ub::T) where T <: Real
    N_pts = 2*N_intervals
    X = sort(collect( convertcompactdomain(rand(), zero(T), one(T), lb, ub) for d = 1:N_pts ))

    start_pts = Vector{T}(undef, N_intervals)
    finish_pts = Vector{T}(undef, N_intervals)

    j = 0 # X is for internal use, assume 1-indexing.
    for i in eachindex(start_pts)
        j += 1
        start_pts[i] = X[j]

        j += 1
        finish_pts[i] = X[j]
    end

    return start_pts, finish_pts
end


