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

function plot_bias_var(λ)
    ts = collect(LinRange(0,1,500))
    ys = zeros(500)

    β = 1.
    N = 25
    X = collect(LinRange(0,1,N))

    p = plot()
    for i in 1:100
        y = sin.(2*π*X) .+ randn(N)/β
        lm_ml = LM_ML(X, y, basis=monom_basis, λ=λ)
        ys´ = predict(lm_ml, ts)
        ys += ys´/100
        if i % 5 == 0
            plot!(ts, ys´, label="", lc=1)
        end
    end
    plot!(ts, ys, label="average", lc=:black, lw=2)

    plot!(t->sin(2π*t), label="true", lc=:black, ls=:dot, lw=3)
end

Random.seed!(1)
plot_bias_var(0.)

Random.seed!(1)
plot_bias_var(1.)


β = 1.
N = 25
X = collect(LinRange(0,1,N))
y = sin.(2*π*X) .+ randn(N)/β
lm_ml = LM_ML(X, y, basis=monom_basis, λ=0.)
plot_lm(lm_ml)