using Random
using Plots
include("linreg.jl")


β = 3.
N = 25
X = collect(LinRange(0,1,N))

Random.seed!(1)
y = -1.2 .+ 2.3 * X .+ randn(N)/β

lm_ml = LM_ML(X, y)
plot_lm(lm_ml)
plot!(t-> -1.2 + 2.3*t, label="true", legend=:topleft)
summary_stats(lm_ml)

lm_bayes = LM_Bayes(X, y, α=0.1, β=β)
p = plot_lm(lm_bayes)

ws = sample(lm_bayes, 5)

for i in 1:size(ws, 2)
    w = ws[:,i]
    plot!(t -> w[1] + w[2]*t, lc=:black, label="")
end

display(p)



β = 5.
N = 50
X = collect(LinRange(0,1,N))

Random.seed!(1)
y = sin.(2*π*X) .+ randn(N)/β

scatter(X,y, mc=2, label="data")
savefig("obsdata.svg")

function gauss_basis(x_vec)
    x = x_vec[1]
    exp.([0, x^2/2, (x-1)^2, (x+1)^2, (x-2)^2/4, (x+2)^2/4, (x-3)^2/9, (x+3)^2/9])
end

lm_ml = LM_ML(X, y, basis=gauss_basis)
plot_lm(lm_ml)
plot!(t->sin(2π*t), label="true", lc=:black, ls=:dot, lw=3)
savefig("ml.svg")

Random.seed!(1)
is = rand(1:N, 5)
lm_bayes = LM_Bayes(X[is], y[is], basis=gauss_basis, α=10^-6, β=β)
plot_lm(lm_bayes, xlims=(0,1))
savefig("bayes5.svg")

Random.seed!(1)
is = rand(1:N, 10)
lm_bayes = LM_Bayes(X[is], y[is], basis=gauss_basis, α=10^-6, β=β)
plot_lm(lm_bayes, xlims=(0,1))
savefig("bayes10.svg")


lm_bayes = LM_Bayes(X, y, basis=gauss_basis, α=10^-6, β=β)
plot_lm(lm_bayes)
plot!(t->sin(2π*t), label="true", lc=:black, ls=:dot, lw=3)
savefig("bayes.svg")

ws = sample(lm_bayes, 10)
p = plot();
ts = collect(LinRange(0,1,500))
for i in 1:size(ws, 2)
    w = ws[:,i]
    plot!(ts, predict(lm_bayes, ts, w), lc=:black, label="")
end
display(p)
scatter!(X,y, mc=2, label="data", ms=2)


Random.seed!(1)

function monom_basis(x_vec)
    x = x_vec[1]
    [x^j for j in 0:24]
end

function gauss_basis_2(a,b,N,σ)
    μ = LinRange(a,b,N)
    function F(x_vec)
        x = x_vec[1]
        vcat(1, exp.((x .- μ).^2 / σ^2))
    end
end

using Statistics
function plot_bias_var(λ, n_iter)

    ts = collect(LinRange(0,1,500))
    ys = zeros(n_iter, 500)

    β = 100000
    N = 25
    X = collect(LinRange(0,1,N))

    p = plot(ts, sin.(2*π*ts), ls=:dot, lc=:black, label="true", lw=3)
    # plot!(ts, sin.(2*π*ts).+1/β, label="", lc=:black, ls=:dot)
    # plot!(ts, sin.(2*π*ts).-1/β, label="", lc=:black, ls=:dot)
    for i in 1:n_iter
        y = sin.(2*π*X) .+ randn(N)/β
        lm_ml = LM_ML(X, y, basis=gauss_basis_2(0,1,N-1,1.), λ=λ)
        ys´ = predict(lm_ml, ts)
        ys[i,:] = ys´
        if (i % 5 == 0)
            plot!(ts, ys´, label="", lc=1)
        end
    end
    y_mean = zeros(500)
    y_sd = zeros(500)
    for i in 1:500
        y_mean[i] = Statistics.mean(ys[:,i])
        y_sd[i] = Statistics.std(ys[:,i])
    end
    plot!(ts, y_mean, ribbon=y_sd, fc=2, label="average", lc=:black, lw=2)

    # plot!(t->sin(2π*t), label="true", lc=:black, ls=:dot, lw=3)
end
Random.seed!(1)
plot_bias_var(0.,100)

Random.seed!(1)
plot_bias_var(1.,100)

function interval_basis(a,b,N)
    μ = collect(LinRange(a,b,N))
    function F(x_vec)
        x = x_vec[1]
        xs = x .- μ
        xs[xs .> (b-a)/(N-1)] .= 0
        xs[xs .< 0] .= 0
        vcat(1, xs)
    end
end

β = 1.
N = 25
X = collect(LinRange(0,1,N))
y = sin.(2*π*X) .+ randn(N)/β
lm_ml = LM_ML(X, y, basis=gauss_basis_2(0,1,100,.1), λ=0.)
plot_lm(lm_ml)

lm_ml.w

F = interval_basis(0,1,11)
for x in 0:0.04:1
    println(x, ": ", F(x))
end

Random.seed!(1)
β = 1.
N = 25
X = collect(LinRange(0,1,N))
y = sin.(2*π*X) .+ randn(N)/β
lm_ml = LM_ML(X, y, basis=interval_basis(0,1,100), λ=0.)
plot_lm(lm_ml)
