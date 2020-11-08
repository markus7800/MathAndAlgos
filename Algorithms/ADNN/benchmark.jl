import Flux
using BenchmarkTools
using Random
include("NN.jl")

function Conv(c::Flux.Conv)
    W = Float64.(flip(c.weight))
    b = Float64.(c.bias)
    σ = c.σ
    stride = c.stride
    Conv(DTensor(W),DTensor(b),σ,stride)
end

function convolve_slice(W::AbstractArray, b::AbstractArray, stride::Tuple{Int,Int}, σ::Function, A::AbstractArray)
	kx, ky, kd1, kd2 = size(W)
	inx, iny, = size(A)
	m, n = size_after_conv((inx, iny), stride, (kx, ky))
	output = Array{eltype(W), 3}(undef, m, n, kd2)
	for i in 1:m, j in 1:n
		x = 1+(i-1)*stride[1]
		y = 1+(j-1)*stride[2]
		w = sum(W .* A[x:x+kx-1,y:y+ky-1,:], dims=(1,2,3))
		w = reshape(w, :)
		output[i,j,:] .= σ.(b .+ w)
	end
	return output
end

function convolve_loop(W::AbstractArray, b::AbstractArray, stride::Tuple{Int,Int}, σ::Function, A::AbstractArray)
	kx, ky, kd1, kd2 = size(W)
	inx, iny, = size(A)
	m, n = size_after_conv((inx, iny), stride, (kx, ky))
	output = Array{eltype(W), 3}(undef, m, n, kd2)
	@inbounds for i in 1:m, j in 1:n, k in 1:kd2
		x = 1+(i-1)*stride[1]
		y = 1+(j-1)*stride[2]
		acc = eltype(W)(0)
		for (i´,x´) in enumerate(x:x+kx-1), (j´,y´) in enumerate(y:y+ky-1), l in 1:kd1
			acc += W[i´,j´,l,k] * A[x´,y´,l]
		end
		output[i,j,k] = σ(b[k] + acc)
	end
	return output
end

Random.seed!(1)
conv_flux = Flux.Conv((5,5), 5=>10)
my_conv = Conv(conv_flux)
X = randn(100, 50, 5, 1)
X_ = reshape(X, 100, 50, 5)

sum(abs.(conv_flux(X) .- my_conv(X_).s))
sum(abs.(convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, my_conv.σ, X_) .-
	convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, my_conv.σ, X_)))

@btime conv_flux(X) # 5.526 ms (45 allocations: 692.05 KiB)
@btime my_conv(X_) # 214.307 ms (1364570 allocations: 144.67 MiB)
@btime convolve_slice(my_conv.W.s, my_conv.b.s, my_conv.stride, my_conv.σ, X_) # 17.400 ms (70659 allocations: 50.67 MiB)
@btime convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, my_conv.σ, X_) # 12.331 ms (3 allocations: 345.11 KiB)
@btime convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, my_conv.σ, X_) # 5.515 ms (3 allocations: 345.11 KiB) (inbounds macro)
214.307 / 5.526 # faster than mine

Y = Float32.(X)
@btime conv_flux(Y) # 449.736 μs (48 allocations: 4.55 MiB)
5.526 / (0.449736) # faster than Float64
214.307 / (0.449736) # faster than mine

################################################################################

Random.seed!(1)
X = randn(100, 50, 5, 128)
@btime conv_flux(X) # 720.447 ms (45 allocations: 86.25 MiB)
5.526 * 128 # scales approximately as expected
(214.307 * 128) / 1000 # expected time for mine

Y = Float32.(X)
@btime conv_flux(Y) # 39.134 ms (302 allocations: 47.35 MiB)
(0.449736 * 128) / 39.134 # faster in batch
(214.307 * 128) / 39.134 # faster than mine iter

################################################################################
