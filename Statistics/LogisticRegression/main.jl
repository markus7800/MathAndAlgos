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


X = hcat(xs, ys, rs)
y = classification(xs, ys)

lg = BIN_LOGR_ML(X, y)

class_pred = predict(lg, X)
n_wrong = sum(class_pred .!= y)

scatter(xs, ys, mc=class_pred, legend=false)

Y = hcat(y, 1 .- y)

lg = MULTI_LOGR_ML(X, Y)

Ŷ = predict(lg, X)

n_wrong = sum(Ŷ .!= Y) / 2

function classification_multi(xs, ys)
    rs = sqrt.(xs.^2 .+ ys.^2)
    class = 1*(xs .+ ys .+ 2rs .> 2) + 2 * (xs .- ys .+ rs .< 1)
    class .+= 1
    return class
end

function onehot(y)
    levels = sort(unique(y))
    Y = zeros(Int, length(y), length(levels))
    for (i,level) in enumerate(levels)
        Y[:,i] .= y .== level
    end
    return Y
end

function onecold(Y)
    map(c -> c[2], argmax(Y, dims=2))
end

scatter(xs, ys, mc=classification_multi(xs,ys), legend=false)
savefig("Statistics/LogisticRegression/multiclassproblem.svg")

Y = onehot(classification_multi(xs,ys))

lg = MULTI_LOGR_ML(X, Y)

N = 100
lin = LinRange(-3,2,N)
grid = zeros(N^2, 3)
for (i,(x,y)) in enumerate(Iterators.product(lin, lin))
    grid[i,:] = [x,y, sqrt(x^2+y^2)]
end

grid_pred = onecold(predict(lg, grid))
probs = predict_probs(lg, grid)
scatter(grid[:,1],grid[:,2],
    mc=grid_pred, ma=probs,
    markerstrokecolor = grid_pred,
    legend=false, ms=2)
scatter!(xs, ys, mc=classification_multi(xs,ys), legend=false, ms=2)
savefig("Statistics/LogisticRegression/multiclasssolved.svg")

lg = MULTI_LOGR_ML(hcat(xs,ys), Y)

N = 100
lin = LinRange(-3,2,N)
grid = zeros(N^2, 2)
for (i,(x,y)) in enumerate(Iterators.product(lin, lin))
    grid[i,:] = [x,y]
end

grid_pred = onecold(predict(lg, grid))
probs = predict_probs(lg, grid)
scatter(grid[:,1],grid[:,2],
    mc=grid_pred, ma=probs,
    markerstrokecolor = grid_pred,
    legend=false, ms=2)
scatter!(xs, ys, mc=classification_multi(xs,ys), legend=false, ms=2)
savefig("Statistics/LogisticRegression/multiclassnotsolved.svg")
