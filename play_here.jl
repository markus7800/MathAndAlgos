using Distributions
using Random
using Plots
using BenchmarkTools

p = 0.3
n = 100

Random.seed!(1)
X = 1/n*rand(Binomial(n, p), 1000)

plot(x -> pdf(Binomial(n,p),x), 0, 100)

histogram(X)

var(X)
p*(1-p) / n


Random.seed!(1)
A = randn(Float32,500,1000)
B = randn(Float32,1000,500)
C = randn(Float32,500,1000)

@btime A*B
# F64 5.740 ms
# F32 2.719 ms
# F16 7.615 s

@btime A+C
# F64 431.216 μs
# F32 177.652 μs
# F16 6.895 ms


p = plot(xlims=(-2,2))

for n in [1,2,5,10,20,50,100]
    plot!(x -> pdf(TDist(n), x), label="", lc=2)
end
plot!(x->pdf(Normal(), x), lc=1, lw=2)
