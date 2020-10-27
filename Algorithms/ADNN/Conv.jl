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



function size_after_conv(input_size::T, stride::T, size::T) where T <: Tuple{Int,Int}
	function n(inps, size, stride)
		return Int(floor((inps - size) / stride) + 1)
	end
	return n(input_size[1], size[1], stride[1]), n(input_size[2], size[2], stride[2])
end

function convolute(conv, input)
	kx, ky, kd1, kd2 = size(conv.weight)
	# flip weights
	kernel = [conv.weight[kx-i+1,kx-j+1,k,l] for i in 1:ky, j in 1:ky, k in 1:kd1, l in 1:kd2]
	stride = conv.stride
	inx, iny, inz = size(input)
	m, n = size_after_conv((inx, iny), stride, (kx, ky))
	output = Array{Float32, 3}(undef, m, n, kd2)
	for i in 1:m, j in 1:n
		x = 1+(i-1)*stride[1]
		y = 1+(j-1)*stride[2]
		w = sum(kernel .* input[x:x+kx-1,y:y+ky-1,:,1], dims=(1,2,3))
		w = reshape(w, :)
		output[i,j,:] = conv.σ.(conv.bias .+ w)
	end
	return output
end

conv = Conv((10, 10), (2=>3), relu, stride=4)
conv.weight
conv.bias

X = rand(Float32, 60, 60, 2, 1)
C1 = conv(X)
C2 = convolute(conv, X)

all(conv(X) .≈ convolute(conv, X))

sum(abs.(conv(X) .- convolute(conv, X)))


x = rand(Float32, 10, 10, 2)
k = conv.weight
k .* x

sum(k .* x, dims=(1,2,3))

C1[:,:,3]


kx, ky, kd1, kd2 = size(conv.weight)
# flip weights
kernel = [conv.weight[kx-i+1,kx-j+1,kd1-k+1,kd2-l+1] for i in 1:ky, j in 1:ky, k in 1:kd1, l in 1:kd2]

s = conv.stride
i = 10
j = 10
x = 1+(i-1)*s[1]
y = 1+(j-1)*s[2]
w = sum(kernel .* X[x:x+kx-1,y:y+ky-1,:,1], dims=(1,2,3))
w = reshape(w, :)
conv.σ.(conv.bias .+ w)

sum(kernel .* X[x:x+kx-1,y:y+ky-1,:,1], dims=(1,2,3))

conv.bias

x = [1., 2., 3.]
y = [2., -1, 3.]

w = DVec([1.,1.,1.])
r = x⋅w + y⋅w
backward(r)
w.∇

w = DVec([1.,1.,1.])

r = x⋅w
backward(r)
r = y⋅w
backward(r)

w.∇
