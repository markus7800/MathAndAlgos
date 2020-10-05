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

# TODO n dim
function sample_GP_1D(gp::GP, xs::AbstractMatrix{Float64}, n_sample=1)
    d, N = size(xs)
    μ = mean(gp.mean, xs) # 1 × N
    K = cov(gp.kernel, xs, xs)
    make_posdef!(K)

    Z = randn(N, n_sample)
    UW = unwhiten(K, Z)

    return μ' .+ UW
end

#=
 if Z is a standardnormal col vector (X_1, ..., X_N)^T ∼ N(0,I)
 then U^T * Z ∼ N(0,Σ)
 where U^T * U = Σ
=#
function unwhiten(Σ::AbstractMatrix, Z::AbstractMatrix)
    U = cholesky(Σ).U
    return U'Z
end

#=
 if Z is a standardnormal col vector (X_1, ..., X_N)^T ∼ N(0,Σ)
 then U^T * Z ∼ N(0,I)
 where U^T * U = Σ
=#

function unwhiten(Σ::AbstractMatrix, Z::AbstractMatrix)
    U = cholesky(Σ).U
    return U'Z
end

#=
 if Z is a gaussian col vector (X_1, ..., X_N)^T ∼ N(0,Σ)
 then Y = U^T \ Z ∼ N(0,I)
 where U^T * U = Σ
 (U^T * Y = Z)
=#
function whiten(Σ::AbstractMatrix, Z::AbstractMatrix)
    U = cholesky(Σ).U
    return U' \ Z
end


# 1 dimensional case
sample_GP(gp::GP, xs::AbstractVector{Float64}, n_sample=1) = sample_GP_1D(gp, xs', n_sample)

function plot_GP_1D(gp::GP, xs::AbstractVector; q=0.95)
    μs = map(x -> mean(gp.mean, x)[1], xs)
    s = quantile(Normal(), (q+1)/2)
    σs = map(x -> s*sqrt(cov(gp.kernel,x,x)[1]), xs)
    plot(xs, μs, ribbon=σs, fillalpha=0.25, lw=3, legend=false)
end

function posterior_naive(gp::GP, xtrain::AbstractMatrix{Float64}, ytrain::AbstractMatrix{Float64}; σ=0.)
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

function predict_naive(gp::GP, xpred::AbstractMatrix{Float64}, xtrain::AbstractMatrix{Float64}, ytrain::AbstractMatrix{Float64}; σ=0.)
    d, n = size(xtrain)
    d, N = size(xpred)

    K = cov(gp.kernel, xtrain, xtrain) # n × n
    A = inv(K + σ^2*I(n)) # n × n <- INVERSION IS SILLY
    v = A * (ytrain .- mean(gp.mean, xtrain))' # n × d
    K12 = cov(gp.kernel,xtrain,xpred) # n × N

    #           (N × d)         +  (N × n) * (n × d)
    μ = mean(gp.mean, xpred)' + cov(gp.kernel, xpred, xtrain)*v
    #           (N × N)             + (N×n)*(n×n)*(n×N)
    Σ = cov(gp.kernel,xpred,xpred) - K12'A*K12

    return μ, Σ
end

function posterior(gp::GP, xtrain::AbstractMatrix{Float64}, ytrain::AbstractMatrix{Float64}; σ=0.)
    d, n = size(xtrain)
    Ktrain = cov(gp.kernel, xtrain, xtrain) + σ^2*I(n) # n × n
    v = Ktrain \ (ytrain .- mean(gp.mean, xtrain))' # n × d

    function m_new(x::AbstractVector{Float64}) # x ∈ ℜ^d
        Kcross = cov(gp.kernel,xtrain,x) # n × 1

        #     (1 × d) + (1 × n) * (n × d)
        val = mean(gp.mean, x)' .+ Kcross'v
        return vec(val)
    end
    function k_new(x::AbstractVector{Float64}, x´::AbstractVector{Float64}) # x, x´ ∈ ℜ^d
        xpred = Matrix{Float64}(undef, length(x), 2)
        xpred[:,1] .= x; xpred[:,2] .= x´

        Kcross = cov(gp.kernel,xtrain,xpred) # n × N
        Kpred = cov(gp.kernel,xpred,xpred)

        L = whiten(Ktrain, Kcross)
        K = Kpred .- L'L # see predict

        return K[1,2]
    end

    return GP(FunctionMean(m_new), FunctionKernel(k_new))
end

function predict(gp::GP, xpred::AbstractMatrix{Float64}, xtrain::AbstractMatrix{Float64}, ytrain::AbstractMatrix{Float64}; σ=0.)
    d, n = size(xtrain)
    d, N = size(xpred)

    Ktrain = cov(gp.kernel, xtrain, xtrain) + σ^2*I(n) # n × n
    Kcross = cov(gp.kernel,xtrain,xpred) # n × N
    Kpred = cov(gp.kernel,xpred,xpred)

    v = Ktrain \ (ytrain .- mean(gp.mean, xtrain))' # n × d

    L = whiten(Ktrain, Kcross) # U^T L = Kcross where U^T U = Ktrain

    #           (N × d)         +  (N × n) * (n × d)
    μ = mean(gp.mean, xpred)' + Kcross'v

    # L = (U^-1)^T Kcross
    # L^T L = Kcross^T (U^-1) (U^-1)^T Kcross = Kcross^T Ktrain^-1 Kcross
    # since  Ktrain (U^-1) (U^-1)^T = U^T U (U^-1) (U^-1)^T = I
    Σ = Kpred - L'L

    return μ, Σ
end

predict(gp::GP, xpred::AbstractVector{Float64}, xtrain::AbstractVector{Float64}, ytrain::AbstractVector{Float64}; σ=0.) =
    predict(gp, xpred', xtrain', ytrain', σ=σ)

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
gp = GP(MeanZero(), SE(1.,1.))

xs = LinRange(-4,4,100)

plot_GP_1D(gp, xs)
savefig("posterior_sample.svg")

Random.seed!(1)
fs = sample_GP_1D(gp, xs', 10)
plot!(xs, fs, legend=false)

xtrain = [-2., -1., 0., 1., 2.]
ytrain = [0.5, 1., 2., 1.5, 2.]
gp_pos = posterior(gp, xtrain, ytrain, σ=.0)

plot_GP_1D(gp_pos, xs)
scatter!(xtrain, ytrain)
Random.seed!(1)
fs = sample_GP_1D(gp_pos, xs', 10)
plot!(xs, fs, legend=false)

plot_predict_1D(gp, xs, xtrain, ytrain)
