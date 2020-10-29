include("NN.jl")

using Flux.Data.MNIST
using Base.Iterators: partition
using Flux: onehotbatch, onecold
using Statistics

# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

function get_processed_data(batch_size)
    # Load labels and images from Flux.Data.MNIST
    train_labels = MNIST.labels()
    train_imgs = MNIST.images()
    mb_idxs = partition(1:length(train_imgs), batch_size)
    train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

    # Prepare test set as one giant minibatch:
    test_imgs = MNIST.images(:test)
    test_labels = MNIST.labels(:test)
    test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))

    return train_set, test_set

end


accuracy(m, test_set) = accuracy(m, test_set...)
function accuracy(model::Model, x, y)
    d = size(x)
    ranges = fill(:, length(d)-1)
    mean(map(i -> onecold(model(x[ranges...,i]).s) == onecold(y[:,i]), 1:size(x,2)))
end
# accuracy(model::Chain, x, y) = mean(map(i -> onecold(model(x[:,i])) == onecold(y[:,i]), 1:size(x,2)))


using ProgressMeter
function learn!(m::Model, train_set; η=0.01)
    cols = fill(:, length(size(train_set[1][1]))-1)

    @showprogress for batch in train_set
        imgs, labels = batch
        batch_size = size(imgs,2)
        r = DVal(0.)
        for i in 1:batch_size
            img = imgs[cols...,i]
            label = labels[:,i]
            r += logitcrossentropy(m(img), label)
        end
        backward(r)
        update_GDS!(m, η=η/batch_size)
    end
end


flatten(x) = reshape(x,:,size(x)[end])


# MLP


train_set, test_set = get_processed_data(128)

train_set = map(batch -> flatten.(batch), train_set)
test_set = flatten.(test_set)

n_in = size(test_set[1], 1)

Random.seed!(1)
m = Model(
  Dense(n_in, 30, sigmoid, init=:glorot),
  Dense(30, 10, sigmoid, init=:glorot)
)

accuracy(m, test_set)
learn!(m, train_set, η=1.)
accuracy(m, test_set)


Random.seed!(1)
m = Model(
  Dense(n_in, 30, sigma, init=:normal),
  Dense(30, 10, sigma, init=:normal)
)

accuracy(m, test_set)
learn!(m, train_set, η=1.)
accuracy(m, test_set)


Random.seed!(1)
m = Model(
  Dense(n_in, 30, sigmoid, init=:uniform),
  Dense(30, 10, sigmoid, init=:uniform)
)

accuracy(m, test_set)
learn!(m, train_set, η=1.)
accuracy(m, test_set)





# CONV
Random.seed!(1)
m = Model(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), 1=>16, relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 13x13 image
    Conv((3, 3), 16=>32, relu),
    MaxPool((2,2)),

    # Third convolution, operating upon a 5x5 image
    Conv((3, 3), 32=>32, relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (1, 1, 32, N)
    flatten,
    Dense(32, 10)
)



train_set, test_set = get_processed_data(128)

train_set = map(batch -> Array.(batch), train_set)
test_set = Array.(test_set)


accuracy(m, test_set)
learn!(m, train_set, η=0.01)

train_set[1][1]

img = test_set[1][:,:,:,1]
lab = test_set[2][:,1]

r = logitcrossentropy(m(img), lab)
backward(r)
