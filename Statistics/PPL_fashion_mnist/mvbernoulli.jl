
struct MvBernoulli <: Distribution{Vector{Bool}} end
const mvbernoulli = MvBernoulli()

function Gen.logpdf(::MvBernoulli, x::AbstractArray{Bool,1}, probs::AbstractArray{U,1}) where {U <: Real}
    l = 0.
    for (i, prob) in enumerate(probs)
        l += x[i] ? log(prob) : log(1. - prob)
    end
    return l
end

function Gen.logpdf_grad(::MvBernoulli, x::AbstractArray{Bool,1}, probs::AbstractArray{U,1}) where {U <: Real}
    probs_grads = zeros(length(probs))
    for (i, prob) in enumerate(probs)
        probs_grads[i] = x[i] ? 1. / prob : -1. / (1-prob)
    end
    (nothing, probs_grads)
end

function Gen.random(::MvBernoulli, probs::AbstractArray{U,1}) where {U <: Real}
    return rand(length(probs)) .< probs
end

Gen.is_discrete(::MvBernoulli) = true

(::MvBernoulli)(probs) = Gen.random(MvBernoulli(), probs)

Gen.has_output_grad(::MvBernoulli) = false
Gen.has_argument_grads(::MvBernoulli) = (true,)
