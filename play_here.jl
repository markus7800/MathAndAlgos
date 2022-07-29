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



begin
    ps = []
    rs = []

    for i in 1:10_000_000
        p = rand() < 1/3 ? 1. : 0.5
        r = rand() < p ? 1. : 0.
        push!(ps, p)
        push!(rs, r)
    end
end

ps_cond = ps[rs .== 1]
sum(ps_cond .== 1.) / length(ps_cond)

function prog()
    a = 0
    b = 0
    x = 2
    y = 0

    if x > 0 #while x > 0
        a = rand() < 1/3 ? 0 : 1
        b = rand() < 1/4 ? 1 : 2

        if !( (a + b) > 1 )
            return NaN
        end
        x = x - 1
        y = y + a + b
    end
    return y
end

ys = begin
    ys = []
    for _ in 1:10_000_000
        y = prog()
        if !isnan(y)
            push!(ys, y)
        end
    end
    ys
end

function BN()
    p1 = rand() < 0.4

    if p1
        p2 = rand() < 0.8
    else
        p2 = rand() < 0.5
    end

    if p2
        p3 = rand() < 0.2
    else
        p3 = rand() < 0.3
    end

    return p1, p2, p3
end

using Statistics
p1s = begin
    p1s = []
    N = 100_000_000
    for _ in 1:N
        p1, p2, p3 = BN()

        if p2 && !p3
            push!(p1s, p1)
        end
    end
    @info("Stats:", mean(p1s), length(p1s) / N)
    p1s
end




using LinearAlgebra

A = Matrix(Float64[
    4 -1 2
    -1 5 -2
    1 1 -4
])

b = Vector(Float64[8, 3, -9])

x = A\b

A * x

function jacobi(A, b, x)
    m, n = size(A)
    @assert m == n
    x_new = zeros(n)
    for i in 1:n
        for j in 1:n
            i == j && continue
            x_new[i] -= A[i,j]*x[j]
        end
        x_new[i] += b[i]
        x_new[i] /= A[i,i]
    end
    return x_new
end

x = jacobi(A, b, [0., 0., 0.])

jacobi(A, b, x)

using Plots
using Statistics

t = Float64[1950, 1960, 1970, 1980, 1990, 2000, 2010]
ft = Float64[2.54, 3.03, 3.70, 4.46, 5.33, 6.14, 6.69]
lnft = log.(ft)

n = length(t)
a = (t'lnft - n*mean(t)*mean(lnft)) / (t't - n*mean(t)^2)
lnc = mean(lnft) - a * mean(t)

scatter(t, lnft, legend=false);
plot!(t -> lnc + a*t)


scatter(t, ft, legend=false);
plot!(t -> exp(lnc) * exp(a*t), xlims=(1940, 2030))

exp(lnc) * exp(a*2020)



f(x) = 4/(1+x^2)
n = 10
h = 1/n

x = [h*i for i in 0:n]

y = f.(x)

h * sum(y) - h/2*(y[1] + y[end])

Q = 0
for i in 1:(n+1)
    if (i-1) % 2 == 0
        Q += 2 * y[i]
    else
        Q += 4 * y[i]
    end
end
Q -= y[1]
Q -= y[end]
Q * h/3


x = 0
y = 0
f(x,y) = 1 + x - y^3

h = 0.0001
xs = Float64[x]
ys = Float64[y]
for n in 1:Int(ceil(1/h))
    y = y + h * f(x,y)
    x = x+h
    println(x, ", ", y)
    push!(xs, x)
    push!(ys, y)
end

plot(xs, ys, legend=false)
