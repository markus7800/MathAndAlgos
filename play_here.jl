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



using Statistics

N = 100
p = 0.3
X = Float64.(rand(N) .<  p)

s = sum(X)

mean(X)
s/N

1/(N-1) * (s*(1-s/N)^2 + (N-s)*(s/N)^2)

var(X)


p*(1-p) # var


using StatsBase

means = [mean(sample(X, length(X), replace=true)) for _ in 1:1000]
var(means)
p*(1-p) / N
# X_i ~ Bernoulli(p)
# ̂p = 1/N sum_{i=1}^N X_i ~ 1/N Binom(p, N)
# -> Var(̂p) = 1/N^2 N p(1-p) = p (1-p) / N
histogram(means, xlims=[0,1])

# Beta(1,1) = Unif(0,1) p-prior
# Bernoulli conjugate
# Beta(1 + ∑X_i, 1 + N - ∑ X_i) posterior
# var = α β / ((α + β)^2 (α + β + 1)) ≈ α β / N^3 ≈ p (1-p) / N

using Distributions

posterior = Beta(1+s, 1+N-s)
var(posterior)

plot(x -> pdf(posterior, x), xlims=(0,1))

mean(posterior)

using Roots


function ci(posterior, α, t)
    return cdf(posterior, min(1, mean(posterior)+t)) - cdf(posterior, max(0,mean(posterior)-t)) - α
end

cdf(posterior,0)

ci(posterior, 0.95, 0)
ci(posterior, 0.95, 1)

α = 0.95
t = find_zero(t -> ci(posterior, α, t), (0,1))

cdf(posterior, min(1, mean(posterior)+t)) - cdf(posterior, max(0,mean(posterior)-t))


plot(x -> pdf(posterior, x), xlims=(0,1))
vline!([min(1, mean(posterior)+t), max(0,mean(posterior)-t)])



using BenchmarkTools

using Random

Random.seed!(0)
N = 4096
A = randn(Float32, N, N)
B = randn(Float32, N, N)

stats = @timed A * B

GFLOPS = N^3 / 10^9 / stats.time


Random.seed!(0)
A = randn(Float32, N, N)
x1 = Float32.(rand(N) .> 0.5)
x2 = copy(x1)
x2[1] = 1 - x1[1]
x2[2] = 1 - x1[2]
x2[3] = 1 - x1[3]
Δx = x2 - x1

@btime A * x # 3.4ms
@btime A * x2 # 3.4ms
@btime A * Δx # 3.4ms

Δ1 = x2 .> x1
Δ2 = x2 .< x1

A*x2

A*x2 - A*x1


sum(A[:, Δ1], dims=2) - sum(A[:, Δ2], dims=2)  # = A*x2 - A*x1

A*x1 + A*(x2 .- x1)

z = A*x1 + sum(A[:, Δ1], dims=2) - sum(A[:, Δ2], dims=2)
maximum(abs.(z - A*x2))

y = A*x1
@btime y + sum(A[:, Δ1], dims=2) - sum(A[:, Δ2], dims=2) # 16.100 μs (17 allocations: 112.89 KiB)

@btime y + sum((@view A[:, Δ1]), dims=2) - sum((@view A[:, Δ2]), dims=2) # 14.900 μs (18 allocations: 64.97 KiB)

d = x2 - x1
@btime y + A*d # 3.394 ms (2 allocations: 32.25 KiB)

@btime y + A[:,d .!= 0]*d[d .!= 0] # 52.000 μs (16 allocations: 90.22 KiB)

@btime y + (@view A[:,d .!= 0]) * (@view d[d .!= 0]) # 43.500 μs (17 allocations: 42.36 KiB)

using Random
Random.seed!(0)

W1 = randn(Float32, N÷2, N)
b1 = randn(Float32, N÷2)

W2 = randn(Float32, N÷4, N÷2)
b2 = randn(Float32, N÷4)

W3 = randn(Float32, 10, N÷4)
b3 = randn(Float32, 10)

x1 = Float32.(rand(N) .> 0.5)
x2 = copy(x1)
x2[1] = 1 - x1[1]
x2[2] = 1 - x1[2]

relu(x) = max.(0, x)

function evalNN(x)
    x = relu(W1 * x + b1)
    x = relu(W2 * x + b2)
    x = relu(W3 * x + b3)
    return x
end

function evalNNstate(x)
    res = []

    y = W1 * x + b1
    push!(res, (copy(x), copy(y)))
    x = relu(y)

    y = W2 * x + b2
    push!(res, (copy(x), copy(y)))
    x = relu(y)

    y = W3 * x + b3
    push!(res, (copy(x), copy(y)))
    x = relu(y)

    return x, res
end

@btime evalNN(x1) # 2.173 ms (9 allocations: 37.12 KiB)

res, state = evalNNstate(x1)


# sum(W1[:, diff], dims=2) = W1*x1 - W1*x2
# W1*x2 + b1 = W1*x1 - sum(W1[:, diff], dims=2) + b1
Δ1 = x2 .> x1
Δ2 = x2 .< x1
z = A*x1 + sum(A[:, Δ1], dims=2) - sum(A[:, Δ2], dims=2)


W1*x1 + b1 + sum(W1[:, Δ1], dims=2) - sum(W1[:, Δ2], dims=2)


W1*x2 + b1

function evalNN(x, state)

    x_, y_ = state[1]
    Δ1 = x .> x_
    Δ2 = x .< x_
    x = relu(y_ + sum((@view W1[:, Δ1]), dims=2) - sum((@view W1[:, Δ2]), dims=2))
    x = relu(W2 * x + b2)
    x = relu(W3 * x + b3)

    return x
end

sum(x1 .!= x2)

res, state = evalNNstate(x1)

@btime evalNN(x2, state) # 591.600 μs (37 allocations: 63.69 KiB)
@btime evalNN(x2) # 1.463 ms (9 allocations: 37.12 KiB)

x = x2

x_, y_ = state[1]

Δ1 = x .> x_
Δ2 = x .< x_
sum(Δ1) + sum(Δ2)
@btime y = y_ + sum(W1[:, Δ1], dims=2) - sum(W1[:, Δ2], dims=2)
maximum(abs.(W1 * x + b1 - y))
x = relu(y)

x_, y_ = state[2]


(x_ .== 0) .& (x_ .!= x)

@btime W2 * x  # 544.000 μs (1 allocation: 4.12 KiB)

d = vec(x .!= 0.)
@btime W2[:, d] * x[d] # 1.470 ms (5 allocations: 4.07 MiB)
@btime W2[:, d] * x[d, :] # 1.599 ms (5 allocations: 4.07 MiB)
@btime (@view W2[:, d]) * (@view x[d, :]) # 2.322 ms (13 allocations: 21.08 KiB)

W2[:, d]
x[d, :]
