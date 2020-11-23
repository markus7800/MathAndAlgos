

# ≈ 20μs
function random_permute(X::Array{Float32, 3}, pad=4, crop=32, flip=true)
    nx, ny, nc = size(X)
    Y = Array{Float32, 3}(undef, nx+2*pad, ny+2*pad, nc)
    Y[pad+1:nx+pad, pad+1:ny+pad, :] .= X

    # extend X by reflecting edges
    Y[1:pad, pad+1:ny+pad, :] .= X[pad+1:-1:2, :, :]
    Y[nx+pad+1:end, pad+1:ny+pad, :] .= X[nx-1:-1:nx-pad, :, :]
    Y[:, 1:pad, :] .= Y[:, 2*pad+1:-1:pad+2, :]
    Y[:, ny+pad+1:end, :] .= Y[:,ny+pad-1:-1:ny,:]

    # random crop
    i = rand(1:2*pad)
    j = rand(1:2*pad)
    Y = Y[i:i+crop-1, j:j+crop-1,:]

    # random flip
    if flip && rand() ≤ 0.5
        Y = Y[:,end:-1:1,:] # horizontal flip
    end

    return Y
end

# ≈ 2s
function random_permute_set(train_set; shuffle=true, kw...)
    count = 0
    permuted_set = similar(train_set)
    N = length(train_set)
    # at least shuffle order of batches
    indexes = shuffle ? sample(1:N, N, replace=false) : collect(1:N)

    v,t = @timed for (k,batch) in enumerate(train_set)
        permuted_images = similar(batch[1])
        for i in 1:size(batch[1],4)
            permuted_images[:,:,:,i] = random_permute(batch[1][:,:,:,i]; kw...)
            count += 1
        end
        permuted_set[indexes[k]] = (permuted_images, batch[2])
    end
    return permuted_set, count, t
end
