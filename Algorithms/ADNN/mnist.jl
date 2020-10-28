include("NN.jl")

using Flux.Data.MNIST
using Base.Iterators: partition
using Flux: onehotbatch, onecold, flatten
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

    build_topo(d)

    for v in reverse(topo)
        v.s -= η * v.∇
        if v isa DVal
            v.∇ = 0
        else
            v.∇ .= 0
        end
    end
end
accuracy(m, test_set) = accuracy(m, test_set...)
accuracy(model::Model, x, y) = mean(map(i -> onecold(model(x[:,i]).s) == onecold(y[:,i]), 1:size(x,2)))
accuracy(model::Chain, x, y) = mean(map(i -> onecold(model(x[:,i])) == onecold(y[:,i]), 1:size(x,2)))


using ProgressMeter
function learn!(m::Model, train_set; η=0.01)
    @showprogress for batch in train_set
        imgs, labels = batch
        batch_size = size(imgs,2)
        r = DVal(0.)
        for i in 1:batch_size
            img = imgs[:,i]
            label = labels[:,i]
            r += logitcrossentropy(m(img), label)
        end
        backward(r)
        update_GDS!(r, η=η/batch_size)
    end
end


# MLP


train_set, test_set = get_processed_data(128)

train_set = map(batch -> flatten.(batch), train_set)
test_set = flatten.(test_set)

n_in = size(test_set[1], 1)

Random.seed!(1)
m = Model(
  Dense(n_in, 30, sigma, init=:glorot),
  Dense(30, 10, sigma, init=:glorot)
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
  Dense(n_in, 30, sigma, init=:uniform),
  Dense(30, 10, sigma, init=:uniform)
)

accuracy(m, test_set)
learn!(m, train_set, η=1.)
accuracy(m, test_set)









m(test_set[1][:,1])
m.layers[1].W.s

i = 10
img = imgs[:,i]
label = labels[:,i]

r = logitcrossentropy(m(img), label)
print_tree(r)
backward(r, v=true)

m.layers[2].b.∇

update_GDS!(r, η=0.01)


accuracy(test_imgs, test_labels, m)

m(img).s

learn!(m, train_set)
accuracy(imgs, labels, m)

import Flux
Random.seed!(1)
m2 = Chain(
  Flux.Dense(n_in, 30, sigma),
  Flux.Dense(30, 10, sigma)
)

m2.layers[1].W .= m.layers[1].W.s
m2.layers[1].b .= m.layers[1].b.s
m2.layers[2].W .= m.layers[2].W.s
m2.layers[2].b .= m.layers[2].b.s


m(test_set[1][:,1]).s


m2(test_set[1][:,1])

g = gradient(params(m)) do
    Flux.logitcrossentropy(m2(img), label)
end

Flux.logitcrossentropy(m2(img), label)

a = Flux.logitcrossentropy(m2(test_imgs), test_labels)
b = sum(Flux.logitcrossentropy(m2(test_imgs[:,i]), test_labels[:,i]) for i in 1:size(test_imgs,2)) / size(test_imgs,2)
a .≈ b

k = collect(keys(g.grads))[3]

g.grads[k][1][1][2]

augment(x) = x .+ 0.1f0*randn(eltype(x), size(x))
loss(x,y) = Flux.logitcrossentropy(m2(x), y)

accuracy(test_imgs, test_labels, m2)

Flux.Optimise.train!(loss, params(m2), train_set, Descent(1))

accuracy(test_imgs, test_labels, m2)
