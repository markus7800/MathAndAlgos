using Plots
using Distributions
using Random
include("kernel.jl")
include("mean.jl")


mutable struct GP
    mean::Mean
    kernel::Kernel
end

#=
 observations and evalution points are given as
 xs ∈ ℜ^{d × N}
 where d is the dimension of the GP
 and N is the number of observations.
=#


function make_posdef!(K::AbstractMatrix; chances=10)
    if isposdef(K)
        return
    end
    for _ in 1:chances
        ϵ = 1e-6 * tr(K) / size(K,1)
        for i in 1:size(K,1)
            K[i,i] += ϵ
        end
        if isposdef(K)
            return
        end
    end
    throw(ArgumentError("K is not positive definite."))
end

function sample_GP_1D(gp::GP, xs::AbstractMatrix{Float64}, n_sample=1)
    d, N = size(xs)
    μ = mean(gp.mean, xs) # 1 × N
    K = cov(gp.kernel, xs, xs)
    make_posdef!(K)

    Σ = cholesky(K).U
    Z = randn(N, n_sample)
    return μ' .+ Σ'Z
end

# 1 dimensional case
sample_GP(gp::GP, xs::AbstractVector{Float64}, n_sample=1) = sample_GP_1D(gp, xs', n_sample)

function plot_GP_1D(gp::GP, xs::AbstractVector; q=0.95)
    μs = map(x -> mean(gp.mean, x)[1], xs)
    s = quantile(Normal(), (q+1)/2)
    σs = map(x -> s*sqrt(cov(gp.kernel,x,x)[1]), xs)
    plot(xs, μs, ribbon=σs, fillalpha=0.25, lw=3, legend=false)
end

function posterior(gp::GP, xtrain::AbstractMatrix{Float64}, ytrain::AbstractMatrix{Float64}; σ=0.)
    d, n = size(xtrain)
    K = cov(gp.kernel, xtrain, xtrain) # n × n
    A = inv(K + σ^2*I(n)) # n × n
    v = A * (ytrain .- mean(gp.mean, xtrain))' # n × d
    function m_new(x::AbstractVector{Float64}) # x ∈ ℜ^d
        X = Array{Float64}(undef, length(x), 1)
        X[:,1] .= x # d × 1

        #     (1 × d) + (1 × n) * (n × d)
        val = mean(gp.mean, x)' .+ cov(gp.kernel,x,xtrain)*v
        return vec(val)
    end
    function k_new(x::AbstractVector{Float64}, x´::AbstractVector{Float64}) # x, x´ ∈ ℜ^d
        L = cov(gp.kernel,x,xtrain) # 1 × n
        R = cov(gp.kernel,xtrain,x´) # n × 1
        val = cov(gp.kernel, x, x´) # 1 × 1

        #     (1 × n) * (n × n) * (n × 1)
        val .-= (L*A*R)
        return val[1]
    end

    return GP(FunctionMean(m_new), FunctionKernel(k_new))
end

# 1D
posterior(gp::GP, xtrain::AbstractVector{Float64}, ytrain::AbstractVector{Float64}; σ=0.) = posterior(gp, xtrain', ytrain', σ=σ)

function predict(gp::GP, xpred::AbstractMatrix{Float64}, xtrain::AbstractMatrix{Float64}, ytrain::AbstractMatrix{Float64}; σ=0.)
    d, n = size(xtrain)
    d, N = size(xpred)

    K = cov(gp.kernel, xtrain, xtrain) # n × n
    A = inv(K + σ^2*I(n)) # n × n
    v = A * (ytrain .- mean(gp.mean, xtrain))' # n × d
    K12 = cov(gp.kernel,xtrain,xpred) # n × N

    #           (N × d)         +  (N × n) * (n × d)
    μ = mean(gp.mean, xpred)' + cov(gp.kernel, xpred, xtrain)*v
    #           (N × N)             + (N×n)*(n×n)*(n×N)
    Σ = cov(gp.kernel,xpred,xpred) - K12'A*K12

    return μ, Σ
end

predict(gp::GP, xpred::AbstractVector{Float64}, xtrain::AbstractVector{Float64}, ytrain::AbstractVector{Float64}; kw...) =
    predict(gp, xpred', xtrain', ytrain', kw...)

function plot_predict_1D(gp::GP, xs::AbstractVector{Float64}, xtrain::AbstractVector{Float64}, ytrain::AbstractVector{Float64};
            σ=0., q=0.95, obsv=true)

    n = length(xs)
    μ, Σ = predict(gp, xs', xtrain', ytrain', σ=σ)
    μs = vec(μ)
    s = quantile(Normal(), (q+1)/2)
    σs = map(i -> s*sqrt(Σ[i,i]), 1:n)
    plot(xs, μs, ribbon=σs, fillalpha=0.25, lw=3, legend=false)
    obsv && scatter!(xtrain, ytrain)
end


gp = GP(FunctionMean(x -> -x), SE(1.,1.))

xs = LinRange(-3,3,100)

plot_GP_1D(gp, xs)

Random.seed!(1)
fs = sample_GP_1D(gp, xs', 10)
plot!(xs, fs, legend=false)

xtrain = [-1., 0., 1.]
ytrain = [1., 2., 3.]
gp_pos = posterior(gp, xtrain, ytrain)

plot_GP_1D(gp_pos, xs)
scatter!(xtrain, ytrain)

predict(gp, xs, xtrain, ytrain)
plot_predict_1D(gp, xs, xtrain, ytrain)
