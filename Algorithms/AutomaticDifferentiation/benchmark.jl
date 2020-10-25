include("AD.jl")

using Random

using Flux

function f1(_A::Matrix{Float64},_B::Matrix{Float64},_x::Vector{Float64}; v=false)
    A = DMat(_A)
    B = DMat(_B)
    x = DVec(_x)

    r = x ⋅ ((A * B) * x)
    backward(r, v=v)
    return A.∇, B.∇, x.∇
end

function f2(_A::Matrix{Float64},_B::Matrix{Float64},_x::Vector{Float64})
    A = DVal.(_A)
    B = DVal.(_B)
    x = DVal.(_x)

    r = reshape(x,1,:) * ((A * B) * x)
    backward(r[1], v=true)
    return map(a -> a.∇, A), map(b -> b.∇, B), map(v -> v.∇, x)
end

function f3(_A::Matrix{Float64},_B::Matrix{Float64},_x::Vector{Float64})
    f(A,B,x) = x' * ((A * B) * x)
    g = gradient(f, _A, _B, _x)
    g
end

using BenchmarkTools

N = 100

Random.seed!(1)
_A = rand(N,2*N)
_B = rand(2*N,N)
_x = rand(N)
_x' * ((_A * _B) * _x)

∇A1, ∇B1, ∇x1 = f1(_A, _B, _x)

∇A2, ∇B2, ∇x2 = f2(_A, _B, _x)

∇A3, ∇B3, ∇x3 = f3(_A, _B, _x)

sum(abs.(∇A1 - ∇A3))
sum(abs.(∇A2 - ∇A3))

sum(abs.(∇B1 - ∇B3))
sum(abs.(∇B2 - ∇B3))

sum(abs.(∇x1 - ∇x3))
sum(abs.(∇x2 - ∇x3))

@btime f1(_A, _B, _x) # 310.830 μs, N=10^2
@time f2(_A, _B, _x)  # 3.527538 s, N=10^2
@btime f3(_A, _B, _x) # 234.012 μs, N=10^2

@btime f1(_A, _B, _x) # 159.072 ms, N=10^3
@time f2(_A, _B, _x) # DNF, N=10^3
@btime f3(_A, _B, _x) # 129.113 ms, N=10^3
