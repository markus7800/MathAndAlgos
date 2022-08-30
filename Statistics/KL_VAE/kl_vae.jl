
using StatsPlots
using Distributions

using Optim
using QuadGK

p(x) = 0.4 * pdf(Normal(-1., 0.75), x) + 0.6 * pdf(Normal(4., 0.5), x)


q(x, μ, σ) = pdf(Normal(μ, σ), x)


xlimits = (-4,8)

plot(p, xlims=xlimits, label="p");
plot!(x -> q(x, 2, 1), label="q")


KL(p,q) = sum(@. p * log(p / q))

xs = LinRange(xlimits..., 1000)


function inclusive_sampled(x)
    μ, σ = x
    if σ < 0
        return Inf
    end

    n = 10000
    bs = rand(Bernoulli(0.6), n)
    n1s = rand(Normal(-1., 0.75), n)
    n2s = rand(Normal(4., 0.5), n)
    xs = n1s
    xs[bs] = n2s[bs]

    return sum(@. log(p(xs) / q(xs, μ, σ))) / n
end

function inclusive(x)
    μ, σ = x
    if σ < 0
        return Inf
    end
    return quadgk(x -> p(x) * log(p(x) / q(x, μ, σ)), xlimits...)[1]
end

function exclusive_sampled(x)
    μ, σ = x
    if σ < 0
        return Inf
    end

    n = 10000
    xs = rand(Normal(μ, σ), n)

    return sum(@. log(q(xs, μ, σ) / p(xs))) / n
end

function exclusive(x)
    μ, σ = x

    if σ < 0
        return Inf
    end

    return quadgk(x -> q(x, μ, σ) * log(q(x, μ, σ) / p(x)), μ-4*σ, μ+4*σ)[1]
end


res0 = Optim.optimize(inclusive_sampled, [2.,1.], Optim.Options(iterations=10000))
plot(p, xlims=xlimits, label="p",
    title="DK=$(round(res0.minimum,digits=2)) μ=$(round(res0.minimizer[1],digits=2)) σ=$(round(res0.minimizer[2],digits=2))");
plot!(x -> q(x, res0.minimizer...), label="q")

res1 = Optim.optimize(exclusive_sampled, [2.,1.], Optim.Options(iterations=10000))
plot(p, xlims=xlimits, label="p",
    title="DK=$(round(res1.minimum,digits=2)) μ=$(round(res1.minimizer[1],digits=2)) σ=$(round(res1.minimizer[2],digits=2))");
plot!(x -> q(x, res1.minimizer...), label="q")

res2 = Optim.optimize(inclusive, [2.,1.], Optim.Options(iterations=10000))
plot(p, xlims=xlimits, label="p",
    title="DK=$(round(res2.minimum,digits=2)) μ=$(round(res2.minimizer[1],digits=2)) σ=$(round(res2.minimizer[2],digits=2))");
plot!(x -> q(x, res2.minimizer...), label="q")
savefig("inclusive.svg")


res3 = Optim.optimize(exclusive, [2.,1.], Optim.Options(iterations=10000))
plot(p, xlims=xlimits, label="p",
    title="DK=$(round(res3.minimum,digits=2)) μ=$(round(res3.minimizer[1],digits=2)) σ=$(round(res3.minimizer[2],digits=2))");
plot!(x -> q(x, res3.minimizer...), label="q")
savefig("exclusive.svg")

inclusive_sampled(res0.minimizer)
inclusive(res0.minimizer)
inclusive_sampled(res2.minimizer)
inclusive(res2.minimizer)

exclusive_sampled(res1.minimizer)
exclusive(res1.minimizer)
exclusive_sampled(res3.minimizer)
exclusive(res3.minimizer)


include("../MetropolisHastings/MetropolisND.jl")

l(x::Float64, μ1::Float64, μ2::Float64)::Float64 = 0.5 * pdf(Normal(μ1, 0.5), x) + 0.5 * pdf(Normal(μ2, 0.5), x)
logl(x::Float64, μ1::Float64, μ2::Float64)::Float64 = log(l(x, μ1, μ2))

sample_true()::Float64 = rand() < 0.5 ? rand(Normal(-1, 0.5)) : rand(Normal(1, 0.5))

using LinearAlgebra
prior_σ = 3.
prior(μ1::Float64, μ2::Float64)::Float64 = pdf(Normal(0., prior_σ), μ1)*pdf(Normal(0., prior_σ), μ2)
logprior(μ1::Float64, μ2::Float64)::Float64 = logpdf(Normal(0., prior_σ), μ1) + logpdf(Normal(0., prior_σ), μ2)


prior2(μ1::Float64, μ2::Float64)::Float64 = pdf(Uniform(-2.,2.), μ1)*pdf(Uniform(-3.,3.), μ2)
logprior2(μ1::Float64, μ2::Float64)::Float64 = logpdf(Uniform(-2.,2.), μ1) + logpdf(Uniform(-3.,3.), μ2)

using Random
Random.seed!(0)
D = [sample_true() for _ in 1:10]

# not normalised
function posterior(μ::Vector{Float64})::Float64
    μ1, μ2 = μ
    return prod(l(x, μ1, μ2) for x in D) * prior(μ1, μ2)
end
function logposterior(μ::Vector{Float64}, D::Vector{Float64}, logprior::Function)::Float64
    μ1, μ2 = μ
    return logprior(μ1, μ2) + sum(logl(x, μ1, μ2) for x in D)
end


Random.seed!(0)
posterior_samples = log_Metropolis_nD(μ -> logposterior(μ, D, logprior2); n_iter=1010_000, var=2., q_init=[0.,0.])
posterior_samples = posterior_samples[10_001:end,:]
mean(posterior_samples, dims=1)

using KernelDensity
dens = kde(posterior_samples)

# res = Optim.maximize(μ -> pdf(dens, μ[1], μ[2]), [1.,-1.])
# Optim.maximum(res)
# Optim.maximizer(res)

kde_amax = argmax(μ -> pdf(dens, μ[1], μ[2]),
    Iterators.product(LinRange(-2,2,100), LinRange(-2,2,100)))

heatmap(dens, c=:blues, xlims=(-2,2), ylims=(-2,2));
scatter!(kde_amax, label="MAD")
savefig("mcmc_posterior.png")
# E_{z∼q}[log(p(z,D) / q(z))]
# p(z,D) = p(D|z)p(z)
# p(z|D) = p(D|z)p(z)/p(D) ∝ p(D|z)p(z)

mode(posterior_samples)

vae_q(x::AbstractVector{Float64}, μ::AbstractVector{Float64}, σ::Float64)::Float64 = 0.5 * pdf(MvNormal(μ, σ * I(2)), x) + 0.5 * pdf(MvNormal(reverse(μ), σ * I(2)), x)

vae_q([0.,0.], [0.,0.], 1.)

using HCubature
function ELBO(μ::Vector{Float64}, D::Vector{Float64}, vae_q::Function, logprior::Function, σ::Float64)::Float64
    return hcubature(
        x -> vae_q(x, μ, σ) * (logposterior(μ, D, logprior) - log(vae_q(x, μ, σ))),
        (μ[1]-4*σ, μ[2]-4*σ),
        (μ[1]+4*σ, μ[2]+4*σ)
    )[1]
end

ELBO([0., 0.], D, vae_q, logprior2, 1.)
σ = 0.1
res = Optim.maximize(μ -> ELBO(μ, D, vae_q, logprior2, σ), [0.,0.])
Optim.maximum(res)
μ = Optim.maximizer(res)

X = LinRange(-2,2,250)
Y = LinRange(-2,2,250)
Z = [vae_q([x,y], Optim.maximizer(res), σ) for x in X, y in Y]

heatmap(X,Y,Z, c=:blues, xlims=(-2,2), ylims=(-2,2));
scatter!([μ[1], μ[2]], [μ[2], μ[1]], label="MAD")
savefig("vae_posterior.png")
