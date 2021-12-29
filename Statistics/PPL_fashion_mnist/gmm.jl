using Distributions, StatsPlots, Random

using ReverseDiff


# Set a random seed.
Random.seed!(3)

# Construct 30 data points for each cluster.
N = 30

# Parameters for each cluster, we assume that each cluster is Gaussian distributed in the example.
μs = [-3.5, 0.0]

# Construct the data points.
x = mapreduce(c -> rand(MvNormal([μs[c], μs[c]], 1.), N), hcat, 1:2)

# Visualization.
scatter(x[1,:], x[2,:], legend = false, title = "Synthetic Dataset")

using Turing, MCMCChains

# Turn off the progress monitor.
Turing.setprogress!(false);
Turing.setadbackend(:reversediff)

@model function GaussianMixtureModel(x, counter)
    counter[1] += 1

    D, N = size(x)

    # Draw the parameters for cluster 1.
    μ1 ~ Normal()

    # Draw the parameters for cluster 2.
    μ2 ~ Normal()

    μ = [μ1, μ2]

    # Uncomment the following lines to draw the weights for the K clusters
    # from a Dirichlet distribution.

    # α = 1.0
    # w ~ Dirichlet(2, α)

    # Comment out this line if you instead want to draw the weights.
    w = [0.5, 0.5]

    # Draw assignments for each datum and generate it from a multivariate normal.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ Categorical(w)
        x[:,i] ~ MvNormal([μ[k[i]], μ[k[i]]], 1.)
    end
    return k
end;

counter = Int[0]
gmm_model = GaussianMixtureModel(x, counter);

gmm_sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ1, :μ2))
tchain = sample(gmm_model, gmm_sampler, 100);
# tchain = sample(gmm_model, gmm_sampler, MCMCThreads(), 100, 3);

ids = findall(map(name -> occursin("μ", string(name)), names(tchain)));
p = plot(tchain[:, ids, :]; legend=true, labels=["Mu 1" "Mu 2"], colordim=:parameter)

tchain = tchain[:, :, 1];

# Helper function used for visualizing the density region.
function Turing.predict(x, y, w, μ)
    # Use log-sum-exp trick for numeric stability.
    return Turing.logaddexp(
        log(w[1]) + logpdf(MvNormal([μ[1], μ[1]], 1.), [x, y]),
        log(w[2]) + logpdf(MvNormal([μ[2], μ[2]], 1.), [x, y])
    )
end;

contour(range(-5, stop = 3), range(-6, stop = 2),
    (x, y) -> predict(x, y, [0.5, 0.5], [mean(tchain[:μ1]), mean(tchain[:μ2])])
)
scatter!(x[1,:], x[2,:]; legend=false, title="Synthetic Dataset")

assignments = mean(MCMCChains.group(tchain, :k)).nt.mean
scatter(x[1,:], x[2,:]; legend=false,
    title="Assignments on Synthetic Dataset", zcolor=assignments)
