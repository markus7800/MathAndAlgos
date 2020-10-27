include("NN.jl")

using Flux.Data.MNIST
using Base.Iterators: partition
using Flux: onehotbatch, flatten

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


# MLP


train_set, test_set = get_processed_data(128)

train_set = map(batch -> flatten.(batch), train_set)
test_set = flatten.(test_set)

n_in = size(test_set[1], 1)

Random.seed!(1)
m = Model(
  Dense(n_in, 30, sigma),
  Dense(30, 10, sigma)
)

m(test_set[1][:,1])


function update_GDS!(d::DVal; η=0.01)
    # topological order all of the children in the graph
    topo = DType[]
    visited = Set{DType}()
    function build_topo(v)
        if !(v in visited)
            push!(visited, v)
            for child in v.prev
                build_topo(child)
            end
            push!(topo,v)
        end
    end

    for v in reverse(topo)
        v.s .-= η * v.∇
        v.∇ .= 0
    end
end

function learn!(m::Model, train_set)
    for batch in train_set
        imgs, labels = batch
        batch_size = size(imgs,2)
        for i in 1:batch_size
            img = imgs[:,i]
            label = labels[:,i]
            r = logitcrossentropy(m(img), label)
            backward(r)
        end
    end
end

imgs, labels = train_set[1]

i = 10
img = imgs[:,i]
label = labels[:,i]

v = m(img)
backward(sum(v), v=true)

@time r = logitcrossentropy(v, label)

backward(r, v=true)

v = DVec(ones(10))
@time r = logitcrossentropy(v, label)
backward(r, v=true)


v = DVec(ones(10))
r = sum(sigma(v))
backward(r)

v.∇
