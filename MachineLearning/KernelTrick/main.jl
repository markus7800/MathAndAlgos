using Random
using Plots
include("svm.jl")


N = 500

Random.seed!(1)
xs, ys = rand(N) * 5 .- 3, rand(N) * 5 .- 3

function classification(xs, ys, color=false)
    rs = sqrt.(xs.^2 .+ ys.^2)
    class = (xs .+ ys .+ 2rs .> 2)
    if color
        class .+ 1
    else
        class .* 2 .- 1
    end
end

scatter(xs, ys, mc=classification(xs,ys, true), legend=false)
savefig("svm_2d.svg")

rs = sqrt.(xs.^2 + ys.^2)

scatter(xs, ys, rs, mc=classification(xs,ys,true), camera=(75,65), legend=false)
savefig("svm_3d_1.svg")



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


β0, β = svm_coordinate_descent(X, y, 250)
pred = svm_predict(β0, β, X)
pred == y
sum(pred .== y)


β0, β = svm(X, y)
pred = svm_predict(β0, β, X)
pred == y



X = Matrix(hcat(xs, ys)')
y = classification(xs, ys)


β0, β = svm(X, y)
pred = svm_predict(β0, β, X)
pred == y
sum(pred .== y) / length(y)



rs = sqrt.(xs.^2 + ys.^2)
X_aug = Matrix(hcat(xs, ys, rs)')

β0, β, α = svm(X_aug, y)
pred = svm_predict(β0, β, X_aug)
pred == y
sum(pred .== y) / length(y)

sv_is = findall(α .> 1e-3)
svs = X_aug[:, sv_is]
β0s = vec(-β'svs)


function kernel(u,v)
    return u[1]*v[1] + u[2]*v[2] + √(u[1]^2+u[2]^2)*√(v[1]^2+v[2]^2)
end

(X_aug[:,1])'X_aug[:,2]
kernel(X[:,1], X[:,2])

β01, β1, α1 = svm(X, y, kernel)
sum(α .- α1)

pred = svm_predict(α1, X, y, kernel, X)

pred == y
sum(pred .== y) / length(y)


using BenchmarkTools
@btime pred = svm_predict(α1, X, y, kernel, X, fast=true) # 413.633 μs
@btime pred = svm_predict(α1, X, y, kernel, X, fast=false) # 61.605 ms

N = 50
lin = LinRange(-3,2,N)
grid = zeros(3, N^2)
for (i,(x,y)) in enumerate(Iterators.product(lin, lin))
    grid[:,i] = [x,y, sqrt(x^2+y^2)]
end
@time grid_pred = svm_predict(β0, β, grid)

scatter(grid[1,:],grid[2,:],mc=Int.(grid_pred), legend=false)
scatter(grid[1,:],grid[2,:],grid[3,:],
    mc=Int.((grid_pred.+ 1) ./2 .+1), legend=false, camera=(75,65))

savefig("svm_3d.svg")
