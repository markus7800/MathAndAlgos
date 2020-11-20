using Random
using Plots
include("logreg.jl")


N = 500

Random.seed!(1)
xs, ys = rand(N) * 5 .- 3, rand(N) * 5 .- 3

function classification(xs, ys, color=false)
    rs = sqrt.(xs.^2 .+ ys.^2)
    class = Int.(xs .+ ys .+ 2rs .> 2)
    if color
        class .+= 1
    end
    return class
end

scatter(xs, ys, mc=classification(xs,ys, true), legend=false)

rs = sqrt.(xs.^2 + ys.^2)

function my_basis(x)
    [1, x[1], x[2], sqrt(x[1]^2 + x[1]^2)]
end

X = hcat(xs, ys, rs)
y = classification(xs, ys)

lg = BIN_LOGR_ML(X, y)

class_pred = predict(lg, X)
n_wrong = sum(class_pred .!= y)

scatter(xs, ys, mc=class_pred, legend=false)
