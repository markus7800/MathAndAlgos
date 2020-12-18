abstract type ExpectationMaximation end

function expectation(EM::ExpectationMaximation, θ_old)::AbstractMatrix
    error("Not implemented!")
end

function maximisation(EM::ExpectationMaximation, ps::AbstractMatrix, θ_old::T)::T where T
    error("Not implemented!")
end

function log_likelihood(EM::ExpectationMaximation, θ)::Float64
    error("Not implemented!")
end

function solve(EM::ExpectationMaximation, θ_0; max_iter=10^4)
    θ = θ_0
    ll = 0.
    for i in 1:max_iter
        ps = expectation(EM, θ)
        θ = maximisation(EM, ps, θ)
        ll_new = log_likelihood(EM, θ)

        if abs(ll - ll_new) ≤ 1e-6
            return θ
        end
        println("$i: $ll")
        ll = ll_new
    end
end

using Distributions
mutable struct GaussianMixtureEM <: ExpectationMaximation
    X::Array{Float64}
    K::Int
end

include("../../utils/posdef.jl")

function expectation(GMEM::GaussianMixtureEM, θ_old)::Array{Float64}
    πs, μ, Σ = θ_old
    X = GMEM.X
    N, D = size(X)
    K = GMEM.K

    ps = Array{Float64}(undef, N, K) # responsibilities

    for n in 1:N, k in 1:K
        ps[n,k] = πs[k] * pdf(MvNormal(μ[:,k], Σ[:,:,k]), X[n,:])
    end
    ps ./= sum(ps, dims=2)
    return ps
end

using LinearAlgebra
function maximisation(EM::GaussianMixtureEM, ps::AbstractMatrix, θ_old)
    πs, μ, Σ = θ_old
    X = GMEM.X
    N, D = size(X)
    K = GMEM.K

    Ns = vec(sum(ps, dims=1))

    πs_new = Ns ./ N

    # ps ∈ ℜ^{N × K}, X ∈ ℜ^{N × D},  μ ∈ ℜ^{D × K}

    μ_new = similar(μ)
    Σ_new = similar(Σ)
    for k in 1:K
        μ_new[:,k] = X'ps[:,k] / Ns[k] # ∈ ℜ^{D × 1}
        A = X' .- μ_new[:,k]
        B = A*Diagonal(ps[:,k])*A' / Ns[k] # ∈ ℜ^{D × D × 1}
        make_symmetric!(B)
        make_posdef!(B)
        Σ_new[:,:,k] = B
    end

    return πs_new, μ_new, Σ_new
end

# not probability for continuous pdf
function log_likelihood(EM::GaussianMixtureEM, θ)
    πs, μ, Σ = θ
    X = GMEM.X
    N, D = size(X)
    K = GMEM.K
    return sum(log(
                sum(πs[k]*pdf(MvNormal(μ[:,k], Σ[:,:,k]), X[n,:]) for k in 1:K)
                )
            for n in 1:N)
end


using Plots
using Random
using StatsBase

K = 3
πs_true = [0.5, 0.3, 0.2]
μ_true = [
    0.2 0.5 0.7
    0.4 0.5 0.6
]
Σ_true = zeros(2,2,K)

Σ_true[:,:,1] = [
    3 1
    1 3
] / 500

Σ_true[:,:,2] = [
    3 -1
    -1 3
] / 500

Σ_true[:,:,3] = [
    3 1
    1 3
] / 500

N = 500
X = zeros(N, 2)
ks = zeros(Int,N)

Random.seed!(1)
for n in 1:N
    k = sample(1:3, Weights(πs_true))
    ks[n] = k
    X[n,:] .= rand(MvNormal(μ_true[:,k], Σ_true[:,:,k]))
end

scatter(X[:,1], X[:,2], mc=ks)

GMEM = GaussianMixtureEM(X, K)

Random.seed!(1)
A = zeros(2,2,K)
B = randn(2,2); A[:,:,1] = B'B
B = randn(2,2); A[:,:,2] = B'B
B = randn(2,2); A[:,:,3] = B'B
Θ_0 = (fill(1/K, K), randn(2,K), A)

πs_est, μ_est, Σ_est = solve(GMEM, Θ_0)
log_likelihood_1(GMEM, (πs_est, μ_est, Σ_est))

pdf_est(x, y) = sum(πs_est[k] * pdf(MvNormal(μ_est[:,k], Σ_est[:,:,k]), [x,y]) for k in 1:K)
contour(LinRange(0,1,100), LinRange(0,1,100), pdf_est)
scatter!(X[:,1], X[:,2], mc=ks, ms=2)

pdf_est_1(x, y) = pdf(MvNormal(μ_est[:,1], Σ_est[:,:,1]), [x,y])
pdf_est_2(x, y) = pdf(MvNormal(μ_est[:,2], Σ_est[:,:,2]), [x,y])
pdf_est_3(x, y) = pdf(MvNormal(μ_est[:,3], Σ_est[:,:,3]), [x,y])
contour(LinRange(0,1,100), LinRange(0,1,100), pdf_est_1)
contour!(LinRange(0,1,100), LinRange(0,1,100), pdf_est_2)
contour!(LinRange(0,1,100), LinRange(0,1,100), pdf_est_3)
scatter!(X[:,1], X[:,2], mc=ks, ms=2)


mutable struct BernoulliMixtureEM <: ExpectationMaximation
    X::AbstractArray
    K::Int
end


function expectation(BMEM::BernoulliMixtureEM, θ_old)::Array{Float64}
    πs, μs = θ_old
    X = BMEM.X
    N = length(X)
    K = BMEM.K

    # log_ps = Array{Float64}(undef, N, K) # responsibilities
    ps = Array{Float64}(undef, N, K) # responsibilities


    for n in 1:N, k in 1:K
        p = 0.
        for (x, μ) in zip(X[n], μs[k])
            p *= pdf(Bernoulli(μ), x)
        end
        # log_ps[n,k] = p
        ps[n,k] = πs[k] * p + 1e6
    end
    # log_ps .-= log(sum(exp.(log_ps), dims=2))
    ps ./= (sum(ps, dims=2))
    return ps
end

function maximisation(EM::BernoulliMixtureEM, ps::AbstractMatrix, θ_old)
    πs, μs = θ_old
    X = BMEM.X
    N = length(X)
    K = BMEM.K

    Ns = vec(sum(ps, dims=1))

    πs_new = Ns ./ N

    μs_new = similar(μs)
    for k in 1:K
        μs_new[k] = sum(X[n]*ps[n,k] for n in 1:N) / Ns[k]
    end
    return πs_new, μs_new
end

function log_likelihood(EM::BernoulliMixtureEM, θ)
    πs, μs = θ
    X = BMEM.X
    N = length(X)
    K = BMEM.K
    display(μs)

    px = Array{Float64}(undef, N, K)
    for n in 1:N, k in 1:K
        p = 0.
        for (x, μ) in zip(X[n], μs[k])
            p *= pdf(Bernoulli(μ), x)
        end
        px[n,k] = p
    end

    return sum(log(
                sum(πs[k] * px[n,k] for k in 1:K)
                )
            for n in 1:N)
end

K = 2
μs_true = [
    [0.1, 0.5, 0., 0., 0.9],
    [0.9, 0.5, 0.1, 0.5, 0.1]
]

πs_true = [0.4, 0.6]

N = 500
X = Array{Vector{Int}}(undef, N)

Random.seed!(1)
for n in 1:N
    k = sample(1:2, Weights(πs_true))

    x = Vector{Int}(undef, 5)
    for (i,μ) in enumerate(μs_true[k])
        x[i] = rand(Bernoulli(μ))
    end
    X[n] = x
end

BMEM = BernoulliMixtureEM(X, K)

Random.seed!(1)
Θ_0 = (fill(1/K, K), fill(rand(5),2))

solve(BMEM, Θ_0)
