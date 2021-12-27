

using Turing
using StatsPlots

#=
 Turing.jl Introduction
 from https://turing.ml/dev/docs/using-turing/guide
=#

# Models are definded with the @model macro and ~ syntax.
# The arguments of a model are used to condition the model.
# Use @macroexpand1 (before @model) to inspect generated code.
@model function gdemo(x, y)
    s² ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
    return x, y
end

# passing 'missing' to the model leaves the parameters unconditioned
unconditioned = gdemo(missing, missing)
# get a sample
unconditioned()


# Let's condition the model on x = 1.5 and y = 2
conditioned = gdemo(1.5, 2)

# Use Sequential Monte Carlo (SMC) to infer s and m
chain = sample(conditioned, SMC(), 10_000)

# get summary statistics
describe(chain)

# show summary plot
plot(chain)



#=
 Bayesian Logistic Regression
=#

# Model assumptions:
# Prior over parameters w ∈ R^d w ~ p(w) arbitrary
# Observations x ∈ R^d with target y ∈ {0,1}

# P(y = 1 | x) = σ(x'w)
# P(y = 0 | x) = 1 - σ(x'w)
# where σ(t) = 1 / (1 + exp(-t)) logistic sigmoid

# In summary:
# y | x ∼ Bernoulli(p = σ(x'w))

# Find posterior over w
# p(w | X, Y) ∝ p(Y|w,X) p(w)


# generate syntetic data set
using Random
Random.seed!(0)
N = 1000
x1 = rand(N) .* 2 .- 1
x2 = rand(N) .* 2 .- 1
ϵ = rand(Normal(0, 0.25), N)

x = hcat(x1,x2)
y = @. Int(x1^2 + 2 * x2^2 + ϵ < 0.75)

scatter(x1, x2, mc = y.+1, legend=false, aspect_ratio=:equal, xlim=(-1,1), ylim=(-1,1))



# define model
σ(t) = 1 / (1 + exp(-t))

# gaussian priors
@model function logreg_gauss(x, y, μ, Σ)
    w ~ MvNormal(μ, Σ)

    nrow, ncol = size(x)
    for i in 1:nrow
        p = σ(w[1] * x[i,1]^2 + w[2] * x[i,2]^2 + w[3])

        y[i] ~ Bernoulli(p)
    end

    return y
end

# dependent uniform priors
@model function logreg_unif(x, y)
    w1 ~ Uniform(-10, 0)
    w2 ~ Uniform(w1, 1)
    w3 ~ Normal(0, 10)

    nrow, ncol = size(x)
    for i in 1:nrow
        p = σ(w1 * x[i,1]^2 + w2 * x[i,2]^2 + w3)

        y[i] ~ Bernoulli(p)
    end

    return y
end


# inference step
using LinearAlgebra
μ = fill(0., 3)
Σ = 10 * Matrix{Float64}(I(3)) # board prior

m = logreg_gauss(x, y, μ, Σ)
chain_gauss = sample(m, HMC(0.05, 10), 10_000)

describe(chain_gauss)

plot(chain_gauss)

m = logreg_unif(x, y)
chain_unif = sample(m, HMC(0.05, 10), 10_000)

describe(chain_unif)

plot(chain_unif)

# make predictions
function MAP_prediction(x, chain, threshold)
    w = [mean(chain, param) for param in chain.name_map[:parameters]] # mean ≈ MAP

    nrow, ncol = size(x)
    ŷ = zeros(nrow)

    for i in 1:nrow
        p = σ(w[1] * x[i,1]^2 + w[2] * x[i,2]^2 + w[3])

        ŷ[i] = p ≥ threshold ? 1 : 0
    end

    return ŷ
end

chain = chain_gauss

# find best threshold
thresholds = LinRange(0, 1, 101)
accuracies = Float64[]
for threshold in thresholds
    ŷ = MAP_prediction(x, chain, threshold)
    push!(accuracies, mean(ŷ .== y))
end

best_treshold = thresholds[argmax(accuracies)]

plot(thresholds, accuracies, legend=false, xlabel="threshold", ylabel="accuarcy", ylim=(0,1))


function bernoulli_prob(x, chain)
    w = [mean(chain, param) for param in chain.name_map[:parameters]] # mean ≈ MAP
    p = σ(w[1] * x[1]^2 + w[2] * x[2]^2 + w[3])
    return p
end
scatter(x1, x2, mc = y.+1, legend=false, alpha=0.3, aspect_ratio=:equal, xlim=(-1,1), ylim=(-1,1));
contour!(
    LinRange(-1,1,201),
    LinRange(-1,1,201),
    (x1, x2) -> bernoulli_prob([x1,x2], chain),
    levels=[best_treshold])
