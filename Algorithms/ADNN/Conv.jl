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

# function (c::Conv)(x::Union{DTensor, AbstractArray})
# 	convolve(c.W, c.b, c.stride, c.σ, x)
# end

function (c::Conv)(x::DTensor)
	# convolve(c.W, c.b, c.stride, c.σ, x)
	s = convolve_loop(c.W.s, c.b.s, c.stride, x.s)
	res = DTensor(s, prev=[c.W, c.b, x], op="conv")
	res.backward = function bw(∇)
		∇W, ∇b, ∇x = ∇convolve_loop(c.W.s, c.b.s, c.stride, x.s, ∇)
		c.W.∇ .+= ∇W
		c.b.∇ .+= ∇b
		x.∇ += ∇x
	end
	c.σ(res)
end

function (c::Conv)(x::AbstractArray)
	# convolve(c.W, c.b, c.stride, c.σ, x)
	s = convolve_loop(c.W.s, c.b.s, c.stride, x)
	res = DTensor(s, prev=[c.W, c.b], op="conv")
	res.backward = function bw(∇)
		∇W, ∇b, ∇x = ∇convolve_loop(c.W.s, c.b.s, c.stride, x, ∇)
		c.W.∇ .+= ∇W
		c.b.∇ .+= ∇b
	end
	c.σ(res)
end

function update_GDS!(c::Conv, opt)
	update!(opt, c.W.s, c.W.∇)
    update!(opt, c.b.s, c.b.∇)
	c.W.∇ .= 0
	c.b.∇ .= 0
end

function zero_∇!(c::Conv)
	c.W.∇ .= 0
	c.b.∇ .= 0
end

function flip(W)
	kx, ky, kd1, kd2 = size(W)
	[W[kx-i+1,ky-j+1,k,l] for i in 1:kx, j in 1:ky, k in 1:kd1, l in 1:kd2]
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


function convolve_loop(W::AbstractArray, b::AbstractArray, stride::Tuple{Int,Int}, A::AbstractArray)
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
		output[i,j,k] = b[k] + acc
	end
	return output
end

# without sigma
function ∇convolve_loop(W::AbstractArray, b::AbstractArray, stride::Tuple{Int,Int}, A::AbstractArray, ∇::AbstractArray)
	∇W = zeros(size(W)); ∇b=zeros(size(b)); ∇A = zeros(size(A))
	kx, ky, kd1, kd2 = size(W)
	inx, iny, = size(A)
	m, n = size_after_conv((inx, iny), stride, (kx, ky))
	@assert (m, n, kd2) == size(∇)
	@inbounds for i in 1:m, j in 1:n, k in 1:kd2
		x = 1+(i-1)*stride[1]
		y = 1+(j-1)*stride[2]
		for (i´,x´) in enumerate(x:x+kx-1), (j´,y´) in enumerate(y:y+ky-1), l in 1:kd1
			∇W[i´,j´,l,k] +=  A[x´,y´,l] * ∇[i,j,k]
			∇A[x´,y´,l] += W[i´,j´,l, k] * ∇[i,j,k]
		end
		∇b[k] += ∇[i,j,k]
	end
	return ∇W, ∇b, ∇A
end

# function convolve_slice(W::AbstractArray, b::AbstractArray, stride::Tuple{Int,Int}, σ::Function, A::AbstractArray)
# 	kx, ky, kd1, kd2 = size(W)
# 	inx, iny, = size(A)
# 	m, n = size_after_conv((inx, iny), stride, (kx, ky))
# 	output = Array{eltype(W), 3}(undef, m, n, kd2)
# 	for i in 1:m, j in 1:n
# 		x = 1+(i-1)*stride[1]
# 		y = 1+(j-1)*stride[2]
# 		w = sum(W .* A[x:x+kx-1,y:y+ky-1,:], dims=(1,2,3))
# 		w = reshape(w, :)
# 		output[i,j,:] .= b .+ w
# 	end
# 	return output
# end
