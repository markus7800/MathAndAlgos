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

lm_bayes = LM_Bayes(X, y, α=0.1, β=β)
p = plot_lm(lm_bayes)

ws = sample(lm_bayes, 5)

for i in 1:size(ws, 2)
    w = ws[:,i]
    plot!(t -> w[1] + w[2]*t, lc=:black, label="")
end

display(p)
