abstract type Mean end

# apply columnwise
function mean(m::Mean, xs::AbstractMatrix{Float64})
    d, N = size(xs)
    μ = Array{Float64}(undef, d, N)
    for i in 1:N
        μ[:,i] .= _mean(m, xs[:,i])
    end
    return μ
end

# point ∈ ℜ^d evaluations
mean(m::Mean, xs::AbstractVector{Float64}) = mean(m, reshape(xs, :, 1))

# point ∈ ℜ evaluations
mean(m::Mean, x::Float64) = mean(m, [x])


struct FunctionMean <: Mean
    f::Function
end

_mean(m::FunctionMean, xs::AbstractVector{Float64}) = m.f(xs)

function MeanZero()
    return FunctionMean(x -> 0.)
end
