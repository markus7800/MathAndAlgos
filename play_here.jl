using Distributions
using Random
using Plots

p = 0.3
n = 100

Random.seed!(1)
X = 1/n*rand(Binomial(n, p), 1000)

plot(x -> pdf(Binomial(n,p),x), 0, 100)

histogram(X)

var(X)
p*(1-p) / n



p = plot(xlims=(-2,2))

for n in [1,2,5,10,20,50,100]
    plot!(x -> pdf(TDist(n), x), label="", lc=2)
end
plot!(x->pdf(Normal(), x), lc=1, lw=2)
