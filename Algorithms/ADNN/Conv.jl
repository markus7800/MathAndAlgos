function convolve(A, kernel)
    height, width = size(kernel)

    B = similar(A)

    # (i, j) loop over the original image
	m, n = size(A)


    return B
end

using Images

Random.seed!(1)
A = rand(10,10)
k, = Kernel.sobel()

convolve(A,k)


convolve(A,k)




using Flux
conv = Conv((10, 10), (1=>1), relu, stride=4)
conv.weight
conv.bias

X = rand(Float32, 60, 60, 1, 1)

function size_after_conv(input_size::T, stride::T, size::T) where T <: Tuple{Int,Int}
	function n(inps, size, stride)
		return Int(floor((inps - size) / stride) + 1)
	end
	return n(input_size[1], size[1], stride[1]), n(input_size[2], size[2], stride[2])
end

function convolute(conv, input)
	kx, ky = size(conv.weight)
	bias = conv.bias[1]
	# flip weights
	kernel = [conv.weight[kx-i+1,kx-j+1] for i in 1:ky, j in 1:ky]
	stride = conv.stride
	input_size = (size(input,1), size(input,2))
	m, n = size_after_conv(input_size, stride, (kx, ky))
	output = Array{Float32, 2}(undef, m, n)
	for i in 1:m, j in 1:n
		x = 1+(i-1)*stride[1]
		y = 1+(j-1)*stride[2]
		output[i,j] = conv.σ(bias + sum(kernel .* input[x:x+kx-1,y:y+ky-1,:,1]))
	end
	return output
end

X = rand(Float32, 60, 60, 1, 1)
conv(X)


all(conv(X) .≈ convolute(conv, X))
