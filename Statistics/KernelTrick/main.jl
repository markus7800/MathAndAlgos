using Random
using Plots

N = 250

Random.seed!(1)
xs, ys = rand(N) * 5 .- 3, rand(N) * 5 .- 3

function classification(xs, ys)
    rs = sqrt.(xs.^2 .+ ys.^2)
    return (xs .+ ys .+ 2rs .> 2) * 2 .- 1
end

scatter(xs, ys, mc=classification(xs,ys), legend=false)

rs = sqrt.(xs.^2 + ys.^2)

scatter(xs, ys, rs, mc=classification(xs,ys), camera=(45,65))

dist(x,y) = sqrt(x^2 + y^2)
surface(LinRange(-3,2,100),LinRange(-3,2,100),dist)

plane(x,y) = (2-(x+y))/2
surface!(LinRange(-3,2,100),LinRange(-3,2,100),plane)


X = Matrix(hcat(xs, ys, rs)')
y = classification(xs, ys)
β = [1, 1, 2]
β0 = -2
pred = svm_predict(β0, β, X)
pred == y

scatter(xs, ys, mc=classification(xs,ys), legend=false)
scatter(xs, ys, mc=pred, legend=false)

β0, β = naive_svm(X, y)

pred = svm_predict(β0, β, X)
pred == y

svm(X, y)
