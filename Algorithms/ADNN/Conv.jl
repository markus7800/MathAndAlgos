mutable struct Conv
    W::DTensor # store "flipped" version to make convolution easy
    b:: DTensor
    σ::Function
    stride::Tuple{Int,Int}
end

function Conv(size::Tuple{Int,Int}, ch::Pair{Int,Int}, σ::Function=identity; stride=1, init=:glorot)
	strides = (0,0) .+ stride
	if init == :glorot
		W = glorot(size..., ch...)
	elseif init == :normal
		W = randn(size..., ch...)
	end
	b = zeros(ch[2])

	Conv(DTensor(W), DTensor(b), σ, strides)
end

function (c::Conv)(x::Union{DTensor, AbstractArray})
	convolve(c.W, c.b, c.stride, c.σ, x)
end

function update_GDS!(c::Conv; η)
	c.W.s .-= η * c.W.∇
	c.b.s .-= η * c.b.∇
	c.W.∇ .= 0
	c.b.∇ .= 0
end

function flip(W)
	kx, ky, kd1, kd2 = size(W)
	[W[kx-i+1,kx-j+1,k,l] for i in 1:ky, j in 1:ky, k in 1:kd1, l in 1:kd2]
end

function size_after_conv(input_size::T, stride::T, size::T) where T <: Tuple{Int,Int}
	function n(inps, size, stride)
		return Int(floor((inps - size) / stride) + 1)
	end
	return n(input_size[1], size[1], stride[1]), n(input_size[2], size[2], stride[2])
end

# W is flipped kernel to agree with standard definition of convolution
function convolve(W::DTensor, b::DTensor, stride::Tuple{Int,Int}, σ::Function, A::Union{DTensor, AbstractArray})
	kx, ky, kd1, kd2 = size(W)
	inx, iny, = size(A)
	m, n = size_after_conv((inx, iny), stride, (kx, ky))
	output = Array{DTensor, 2}(undef, m, n)
	for i in 1:m, j in 1:n
		x = 1+(i-1)*stride[1]
		y = 1+(j-1)*stride[2]
		w = sum(W * A[x:x+kx-1,y:y+ky-1,:], dims=(1,2,3))
		w = reshape(w, :)
		output[i,j] = σ(b + w)
	end
	return DTensor(output)
end

import Flux
using Random
Random.seed!(1)
conv = Flux.Conv((10, 10), (1=>3), Flux.relu, stride=4)
conv.weight
conv.bias
X = rand(Float32, 60, 60, 1, 1)
C1 = conv(X)

W = DTensor(Float64.(flip(conv.weight)))
b = DTensor(Float64.(conv.bias))
A=X[:,:,1,1]
C2 = convolve(W, b, conv.stride, relu, A)

all(C1 .≈ C2.s)
