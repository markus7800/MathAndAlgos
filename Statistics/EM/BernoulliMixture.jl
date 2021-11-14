include("EM.jl")

using Random
using Distributions
using StatsBase
using Plots


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
        p = 1.
        for (x, μ) in zip(X[n], μs[k])
            p *= pdf(Bernoulli(μ), x)
        end
        # log_ps[n,k] = p
        ps[n,k] = πs[k] * p# + 1e6
    end
    # log_ps .-= log(sum(exp.(log_ps), dims=2))
    ps ./= (sum(ps, dims=2))
    #println(ps)
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

    px = Array{Float64}(undef, N, K)
    for n in 1:N, k in 1:K
        p = 1.
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

function predict_class(μs::Vector, classes::Vector, X::Vector)
    N = length(X)
    K = length(μs)
    pred = Vector{eltype(classes)}(undef, N)

    for n in 1:N
        lls = Array{Float64}(undef, K)
        for k in 1:K
            p = 1.
            for (x, μ) in zip(X[n], μs[k])
                p *= pdf(Bernoulli(μ), x)
            end
            lls[k] = p
        end
        pred[n] = classes[argmax(lls)]
    end

    return pred
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
Θ_0 = (fill(1/K, K), [rand(Uniform(0.25,0.75), 5),rand(Uniform(0.25,0.75), 5)])

solve(BMEM, Θ_0)

log_likelihood(BMEM, Θ_0)


using Flux.Data.MNIST

images = MNIST.images()
labels = MNIST.labels()

classes = [2,3,4]
K = length(classes)
is = [l in classes for l in labels]

images_subset = images[is]
labels_subset = labels[is]

images_train = map(img -> Float32.(img) .> 0.5, images_subset)[1:1000]
labels_train = labels_subset[1:1000]

images_test = map(img -> Float32.(img) .> 0.5, images_subset)[1001:end]
labels_test = labels_subset[1001:end]

D = 28

BMEM = BernoulliMixtureEM(images_train, K)

Random.seed!(1)
Θ_0 = (
    fill(1/K, K),
    [
        rand(Uniform(0.25,0.75), D, D),
        rand(Uniform(0.25,0.75), D, D),
        rand(Uniform(0.25,0.75), D, D)
    ])



Θ_est = solve(BMEM, Θ_0)

πs_est, μs_est = Θ_est

πs_est

Gray.(μs_est[1]) # 3
Gray.(μs_est[2]) # 2
Gray.(μs_est[3]) # 4

pred_train = predict_class(μs_est, [3,2,4], images_train)
sum(labels_train .== pred_train) / length(pred_train)

pred_test = predict_class(μs_est, [3,2,4], images_test)
sum(labels_test .== pred_test) / length(pred_test)
