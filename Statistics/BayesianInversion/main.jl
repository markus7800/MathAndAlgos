
include("Metropolis.jl")
include("PopulationDynamics.jl")
using Random
using Plots

p = plot(fox_pop, label="fox", lc=1);
plot!(rabbit_pop, label="rabbit", lc=2);
p = scatter(fox_data, label="fox data", mc=1, ms=3, markerstrokecolor=1);
scatter!(rabbit_data, label="rabbit data", mc=2, ms=3, markerstrokecolor=2);
display(p)
savefig("foxrabbitdata.svg")

Random.seed!(1)
q, = Metropolis_1D(prior, likelihood,
         10^5, 10^-5,
         δ_min, δ_max, (δ_min+δ_max)/2)

δ, QI_min, QI_max, bins, counts = MAP(q, burn_in=10^3, bin_width=0.0001)

plot(1:10^3, q[1:10^3], lc=:black)
plot!(10^3+1:10^5, q[10^3+1:10^5], lc=1)

# the MAP for δ is ≈ 0.1207
# the 95%-confidence interval is [0.1195, 0.1218]
# the true value of δ = 0.12 is estimated quite good.

# visualising MAP
colors = Base.map(b -> QI_min ≤ b && b < QI_max ? 1 : :gray, bins)
map_bin = filter(i -> bins[i] < δ && δ < bins[i+1], 1:length(bins)-1)[1]
colors[map_bin] = 2
p = bar(bins, counts, fc=colors, legend=false)
display(p)
savefig("deltahist.svg")

# visualising goodness of fit
fox_pop_est, rabbit_pop_est = model(α_exact, β_exact, γ_exact, δ)


p = scatter(fox_data, label="fox data", mc=1, ms=3, markerstrokecolor=1);
scatter!(rabbit_data, label="rabbit data", mc=2, ms=3, markerstrokecolor=2);
p = plot!(fox_pop, label="true fox", lc=:black, ls=:dot, lw=2);
plot!(rabbit_pop, label="true rabbit", lc=:black, ls=:dot, lw=2);
plot!(fox_pop_est, label="est fox", lc=1);
plot!(rabbit_pop_est, label="est rabbit", lc=2);
display(p)
savefig("comptotrue.svg")
