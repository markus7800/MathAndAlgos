
using Random
using BenchmarkTools
include("lin_solve.jl")

L = LowerTriangular([1 0; 2 2])
b = [2, 2]

L \ b
gauss(L, b)

U = UpperTriangular(adjoint(L))

U \ b
gauss(U, b)


A = Matrix{Float64}([1 2 3; 2 0 1; 0 1 3])
b = [1, 1, 1]
x = A \ b

x_gauss = solve(A, b, method=:gauss)
sum(abs.(x_gauss .- x))

x_lu = solve(A, b, method=:lu, pivot=false)
sum(abs.(x_lu .- x))

x_lu_pivot = solve(A, b, method=:lu, pivot=true)
sum(abs.(x_lu_pivot .- x))

x_qr = solve(A, b, method=:qr)
sum(abs.(x_qr .- x))

x_l2 = solve(A, b, method=:L2)
sum(abs.(x_l2 .- x))


Random.seed!(1)
A = Float64.(rand(-25:25, 1000, 1000))
rank(A)
x = Float64.(rand(-25:25, 1000))
b = A*x


@btime x_gauss = solve(A, b, method=:gauss)
# 7.569 s

@btime x_lu = solve(A, b, method=:lu, pivot=false)
# 5.098 s

@btime x_lu_pivot = solve(A, b, method=:lu, pivot=true)
# 5.209 s

@time x_qr = solve(A, b, method=:qr)
# 59.611 s

x_l2 = solve(A, b, method=:L2)
# does not work
