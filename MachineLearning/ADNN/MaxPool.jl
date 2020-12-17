struct MaxPool
    size::Tuple{Int,Int}
    stride::Tuple{Int,Int}
end

function MaxPool(size; stride=size)
	strides = (0,0) .+ stride
	MaxPool(size, strides)
end

# 3 dimensional array
function (mp::MaxPool)(A::DTensor)
	# maxpool(mp.size, mp.stride, A)
	out = maxpool(mp.size, mp.stride, A.s)
	res = DTensor(out, prev=[A], op="maxpool")
	res.backward = function bw(∇)
		∇A = ∇maxpool(mp.size, mp.stride, A.s, out, ∇)
		A.∇ += ∇A
	end
	res
end

# 3 dimensional array
function maxpool(ksize::Tuple{Int,Int}, stride::Tuple{Int,Int}, A::DTensor)
	kx, ky = ksize
	inx, iny, inz = size(A)
	m, n = size_after_conv((inx, iny), stride, (0,0), (kx, ky))
	output = Array{DVal, 3}(undef, m, n, inz)
	for i in 1:m, j in 1:n, k in 1:inz
		x = 1+(i-1)*stride[1]
		y = 1+(j-1)*stride[2]
		output[i,j,k] = maximum(A[x:x+kx-1,y:y+ky-1,k])
	end
	return DTensor(output)
end


# 3 dimensional array
function maxpool(ksize::Tuple{Int,Int}, stride::Tuple{Int,Int}, A::AbstractArray)
	kx, ky = ksize
	inx, iny, inz = size(A)
	m, n = size_after_conv((inx, iny), stride, (0,0), (kx, ky))
	output = Array{Float64, 3}(undef, m, n, inz)
	@inbounds for i in 1:m, j in 1:n, k in 1:inz
		x = 1+(i-1)*stride[1]
		y = 1+(j-1)*stride[2]
		m = -Inf
		for x´ in x:x+kx-1, y´ in y:y+ky-1
			m = max(m, A[x´,y´,k])
		end
		output[i,j,k] = m
	end
	return output
end

# 3 dimensional array
function ∇maxpool(ksize::Tuple{Int,Int}, stride::Tuple{Int,Int}, A::AbstractArray, out::AbstractArray, ∇::AbstractArray)
	kx, ky = ksize
	inx, iny, inz = size(A)
	∇A = zeros(inx, iny, inz)
	m, n = size_after_conv((inx, iny), stride, (0,0), (kx, ky))
	@assert (m,n,inz) == size(∇)
	@inbounds for i in 1:m, j in 1:n, k in 1:inz
		x = 1+(i-1)*stride[1]
		y = 1+(j-1)*stride[2]
		for x´ in x:x+kx-1, y´ in y:y+ky-1
			if A[x´,y´,k] ≥ out[i,j,k] - 1e-9
				∇A[x´,y´,k] +=  ∇[i,j,k]
			end
		end
	end
	∇A
end
