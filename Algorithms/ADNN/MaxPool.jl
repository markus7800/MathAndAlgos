struct MaxPool
    size::Tuple{Int,Int}
    stride::Tuple{Int,Int}
end

function MaxPool(size; stride=size)
	strides = (0,0) .+ stride
	MaxPool(size, strides)
end

# 3 dimensional array
function (mp::MaxPool)(A::Union{AbstractVector, DTensor})
	maxpool(mp.size, mp.stride, A)
end


# 3 dimensional array
function maxpool(ksize::Tuple{Int,Int}, stride::Tuple{Int,Int}, A::DTensor)
	kx, ky = ksize
	inx, iny, inz = size(A)
	m, n = size_after_conv((inx, iny), stride, (kx, ky))
	output = Array{DVal, 3}(undef, m, n, inz)
	for i in 1:m, j in 1:n, k in 1:inz
		x = 1+(i-1)*stride[1]
		y = 1+(j-1)*stride[2]
		output[i,j,k] = maximum(A[x:x+kx-1,y:y+ky-1,k])
	end
	return DTensor(output)
end

A = DTensor(Float64.(X[:,:,:,1]))
maxpool((2,2), (1,1), A)
