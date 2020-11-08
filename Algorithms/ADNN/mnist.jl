include("NN.jl")

using Flux.Data.MNIST
using Base.Iterators: partition
using Flux: onehotbatch, onecold
using Statistics
using StatsBase
using Random
import JLD

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
    mean(map(i -> onecold(model(x[ranges...,i]).s) == onecold(y[:,i]), 1:size(y,2)))
end
# accuracy(model::Chain, x, y) = mean(map(i -> onecold(model(x[:,i])) == onecold(y[:,i]), 1:size(x,2)))


augment(x) = x .+ 0.1*randn(eltype(x), size(x))

import ProgressMeter
function learn!(m::Model, train_set, opt; aug=false)
    zero_∇!(m)

    cols = fill(:, length(size(train_set[1][1]))-1)

    ProgressMeter.@showprogress for batch in train_set
        imgs, labels = batch
        batch_size = size(imgs,2)
        r = DVal(0.)
        for i in 1:batch_size
            img = imgs[cols...,i]
            label = labels[:,i]
            if aug
                img = augment(img)
            end
            r += logitcrossentropy(m(img), label)
        end
        backward(r*(1/batch_size))
        update_GDS!(m, opt)
    end
end

function train!(m::Model, train_set, test_set, n_epoch, opt=Descent(0.01); kw...)
    n_batches = length(train_set)
    @info "Start training..."
    acc = accuracy(m, test_set)
    @info "Start accuracy: $acc"
    for n in 1:n_epoch
        is = sample(1:n_batches, n_batches, replace=true)
        learn!(m, train_set[is], opt; kw...)
        acc = accuracy(m, test_set)
        @info "Epoch $n/$n_epoch done. Accuracy: $acc"
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
learn!(m, train_set, Descent(1.))
accuracy(m, test_set)


Random.seed!(1)
m = Model(
  Dense(n_in, 30, sigma, init=:normal),
  Dense(30, 10, sigma, init=:normal)
)

accuracy(m, test_set)
learn!(m, train_set, Descent(1.))
accuracy(m, test_set)


Random.seed!(1)
m = Model(
  Dense(n_in, 30, sigmoid, init=:uniform),
  Dense(30, 10, sigmoid, init=:uniform)
)

accuracy(m, test_set)
learn!(m, train_set, Descent(1.0))
accuracy(m, test_set)


Random.seed!(1)
m = Model(
  Dense(n_in, 30, sigmoid, init=:glorot),
  Dense(30, 10, sigmoid, init=:glorot)
)

Random.seed!(1)
train!(m, train_set, test_set, 10, Descent(1.)) # 92.8 %

Random.seed!(1)
train!(m, train_set, test_set, 10, ADAM()) # 93 %


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

    # Reshape 3d tensor into a 2d one using flatten, at this point it should be (1, 1, 32, N)
    flatten,
    Dense(32, 10)
)



train_set, test_set = get_processed_data(128)

train_set = map(batch -> Array.(batch), train_set)
test_set = Array.(test_set)


train_set[1][1]

img = test_set[1][:,:,:,1]
lab = test_set[2][:,1]

using BenchmarkTools
@btime r = logitcrossentropy(m(img), lab) # 15.970 ms vs prev 59ms
r = logitcrossentropy(m(img), lab)
@btime backward(r) # 10.407 ms vs prev 29ms

(60000 * (59+29) + 10000 * 59) / 1000 / 60

test_set100 = (test_set[1][:,:,:,1:100], test_set[2][:,1:100])

Random.seed!(1)
# only 7 times slower now instead of 30 times slower :)
# now 7 mins instead of 25-30 mins
train!(m, train_set, test_set100, 5, ADAM(),aug=true) # 95.77 % after 5, 97.9 % after 10

JLD.save("convmodel.jld",
    "W1", m.layers[1].W.s, "b1", m.layers[1].b.s,
    "W2", m.layers[3].W.s, "b2", m.layers[3].b.s,
    "W3", m.layers[5].W.s, "b3", m.layers[5].b.s,
    "W4", m.layers[8].W.s, "b4", m.layers[8].b.s
)


@timed accuracy(m, test_set)

est = map(i -> m(test_set[1][:,:,:,i]).s, 1:size(test_set[1],4))
softmax(v) = exp.(v) / sum(exp.(v))
ps = softmax.(est)
sum(onecold.(est) .== onecold(test_set[2]))
wronga = findall(onecold.(est) .!= onecold(test_set[2]))

m(test_set[1])
using Plots
function plot_img_core(img, lab, ps)
    n = size(img, 1)
    img = [img[n-i+1,j] for i in 1:n, j in 1:n]

    p1 = heatmap(img, legend=false)
    p2 = bar(0:9, ps, legend=false)
    bar!([lab], [ps[lab+1]], fc=:green)
    f, i = findmax(ps)
    bar!([i-1], [f], fc=:red)
    xticks!(0:9)
    ylims!((0,1))
    plot(p1,p2)
end
function plot_img(set, ps, i)
    # println(size(set[1][:,:,1,i]), ", ", findfirst(set[2][:,i])-1, ", ", ps[i])
    plot_img_core(set[1][:,:,1,i], findfirst(set[2][:,i])-1, ps[i])
end

function make_wronga_anim(wronga)
    anim = Animation()
    @progress for i in wronga
        p = plot_img(test_set, ps, i)
        frame(anim, p)
    end
    return anim
end


anim = make_wronga_anim(wronga)

gif(anim, "wronga.gif", fps=1)
plot_img(test_set, ps, 44)

import Flux
Random.seed!(1)
m2 = Flux.Chain(
    # First convolution, operating upon a 28x28 image
    Flux.Conv((3, 3), 1=>16, Flux.relu),
    Flux.MaxPool((2,2)),

    # Second convolution, operating upon a 13x13 image
    Flux.Conv((3, 3), 16=>32, Flux.relu),
    Flux.MaxPool((2,2)),

    # Third convolution, operating upon a 5x5 image
    Flux.Conv((3, 3), 32=>32, Flux.relu),
    Flux.MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using flatten, at this point it should be (1, 1, 32, N)
    Flux.flatten,
    Flux.Dense(32, 10)
)

@btime Flux.logitcrossentropy(m2(test_set[1][:,:,:,1:1]), lab) # 162.408 μs

@time Flux.logitcrossentropy(m2(test_set[1]), test_set[2]) # 2.45 s

m3 = Model(
    Conv(m2.layers[1], relu),
    MaxPool((2,2)),
    Conv(m2.layers[3], relu),
    MaxPool((2,2)),
    Conv(m2.layers[5], relu),
    MaxPool((2,2)),
    flatten,
    Dense(m2.layers[8])
)


zero_∇!(m3)
r = logitcrossentropy(m3(img), lab)
r.s
backward(r)

img_ = reshape(img, size(img)..., 1)
Flux.logitcrossentropy(m2(img_),lab)

ps = Flux.params(m2)
gs = Flux.gradient(ps) do
    Flux.logitcrossentropy(m2(img_), lab)
end
gs[m2.layers[1].weight]
∇W = m3.layers[1].W.∇

sum(abs.(gs[m2.layers[1].weight] .- flip(m3.layers[1].W.∇)))
∇W

sum(abs.(∇W .- ∇W_))
