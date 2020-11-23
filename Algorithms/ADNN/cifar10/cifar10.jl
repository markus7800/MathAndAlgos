using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, flatten
import Functors: @functor
using Metalhead
using Metalhead: trainimgs, valimgs
using Images: channelview
using Random
using Base.Iterators: partition
using Statistics
using BSON
using StatsBase
using Dates
using Printf

# include("lr_schedular.jl")
# include("random_permute.jl")
# include("show.jl")


# no need - just to check
using CUDA
CUDA.has_cuda()
CUDA.version()
CUDA.device()


# Function to convert the RGB image to Float32 Arrays
function getarray(X)
    Float32.(permutedims(channelview(X), (2, 3, 1)))
end

# statistics
const CIFAR10_MEANS  = reshape(Float32[0.4914, 0.4822, 0.4465],1,1,3)
const CIFAR10_SDS = reshape(Float32[0.2023, 0.1994, 0.2010],1,1,3)

function get_processed_data(;batchsize=128, splitr_ = 0.1, N=40_000)
    # Fetching the train and validation data and getting them into proper shape
    X = trainimgs(CIFAR10)
    imgs = [getarray(X[i].img) for i in 1:N] # 50_000 available
    #onehot encode labels of batch

    labels = onehotbatch([X[i].ground_truth.class for i in 1:N],1:10)

    train_pop = Int((1-splitr_) * N)
    train = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, batchsize)]
    valset = collect(train_pop+1:N)
    valX = cat(imgs[valset]..., dims = 4)
    valY = labels[:, valset]

    val = (valX,valY)
    return train, val
end

function get_test_data(N=1_000)
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)

    # CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
    testimgs = [getarray(test[i].img) for i in 1:N] # 10_000 available
    testY = onehotbatch([test[i].ground_truth.class for i in 1:N], 1:10)
    testX = cat(testimgs..., dims = 4)

    test = (testX,testY)
    return test
end



struct ConvBlock
    conv::Conv
    bn::BatchNorm
    mp::MaxPool
    pool::Bool
    σ
end

function ConvBlock(ch::Pair{Int,Int}, size=(3,3), pad=(1,1); pool=false)
    conv = Conv(size, ch, pad=pad)
    bn = BatchNorm(ch[2])
    mp = MaxPool((2,2))
    return ConvBlock(conv,bn,mp,pool,relu)
end

function (CB::ConvBlock)(x)
    x = CB.conv(x)
    x = CB.bn(x)
    x = CB.σ.(x)
    if CB.pool
        x = CB.mp(x)
    end
    return x
end

@functor ConvBlock

struct ResBlock
    conv_block_1::ConvBlock
    conv_block_2::ConvBlock
end

function ResBlock(ch::Pair{Int,Pair{Int,Int}})
    inc, ch2 = ch
    midc, out = ch2
    ResBlock(ConvBlock(inc=>midc), ConvBlock(midc=>out))
end

function (RB::ResBlock)(x)
    y = RB.conv_block_1(x)
    y = RB.conv_block_2(y)
    return y + x
end

@functor ResBlock

function ResNet9(in_channels, n_class)
    prep = ConvBlock(in_channels => 64)
    conv1 = ConvBlock(64=>128, pool=true)
    res1 = ResBlock(128=>128=>128)
    conv2 = ConvBlock(128=>256, pool=true)
    conv3 = ConvBlock(256=>512, pool=true)
    res2 = ResBlock(512=>512=>512)
    classifier = MaxPool((4,4)), flatten, Dense(512, n_class)

    return Chain(prep, conv1, res1, conv2, conv3, res2, classifier...)
end



# x is one big batch
function accuracy(x, y, m; batchsize=1000, enable_gpu=false)
    N = size(x,4)
    s = 0
    _pu = enable_gpu ? gpu : cpu
    # batch to decrease ram demand
    for is in partition(1:N, batchsize)
        s += sum(onecold(cpu(m(_pu(x[:,:,:,is]))), 1:10) .== onecold(cpu(y[:,is]), 1:10))
    end
    return s/N
end

function train(; epochs=8, normalize=false, batchsize=400, permute=true, schedule_lr=true, enable_gpu=false)
    Random.seed!(1)

    @info("Load training data.")
    train_set, val_set = get_processed_data(batchsize=batchsize)
    test_set = get_test_data()

    if normalize
        @info("Normalizing training data.")
        map!(x -> ((x[1] .- CIFAR10_MEANS) ./ CIFAR10_SDS, x[2]), train_set, train_set)
        val_set = (val_set[1] .- CIFAR10_MEANS) ./ CIFAR10_SDS, val_set[2]
        test_set = (test_set[1] .- CIFAR10_MEANS) ./ CIFAR10_SDS, test_set[2]
    end


    @info("Constructing Model.")

    Random.seed!(1)
    m = ResNet9(3,10)
    if enable_gpu
        m = gpu(m)
        train_set = gpu.(train_set)
        val_set = gpu(val_set)
        test_set = gpu(test_set)
    end

    sqnorm(x) = sum(abs2, x)
    L2_penalty(m) = sum(sqnorm, params(m))
    λ = 1f-4
    loss(x, y) = logitcrossentropy(m(x), y) + λ * L2_penalty(m)

    # batches = length(train_set)
    # @info("Number of epochs: $epochs.")
    # @info("Number of batches: $batches.")
    # ocs = OneCycleSchedular(max_lr=0.01, epochs=epochs, batches=batches)
    # if schedule_lr
    #     opt = ADAM(get_lr(ocs))
    #     function cb()
    #         step!(ocs)
    #         opt.eta = get_lr(ocs)
    #     end
    # else
    #     opt = ADAM(0.001)
    #     cb = () -> ()
    # end

    opt = ADAM(0.001)

    @info "Started training at: " * Dates.format(now(), "yyyy/m/d HH:MM")
    for epoch in 1:epochs
        @info "Epoch $(epoch):"
        v,t, = @timed Flux.train!(loss, params(m), train_set, opt)
        @info "Finished in $(Int(round(t/60))) minutes."
        v,t, = @timed (@info "Accuracy on validation: $(accuracy(test_set..., m))")
        @info "Validated in $t seconds."

        # if permute
        #     # do preprocessing on CPU
        #     epoch_set, count, t = random_permute_set(train_set)
        #     @info @sprintf "\tRandomly permuted %d images in %.2f seconds." count t
        # else
        #     epoch_set = train_set
        # end
        #
        # if enable_gpu
        #     epoch_set = gpu.(epoch_set)
        # end
        #
        # v,t_train, = @timed Flux.train!(loss, params(m), epoch_set, opt)
        # @info "\tEpoch finished in $(Int(round(t_train/60))) minutes."
        #
        # v,t_val, = @timed if enable_gpu
        #     acc = accuracy(val_set..., m, batchsize=1000)
        # else
        #     acc = accuracy(val_set..., m)
        # end
        #
        # @info "\tAccuracy on validation: $acc"
        # @info "\tValidated in $(Int(round(t_val))) seconds."
        #
        # if schedule_lr
        #     @info @sprintf "\tLearning rate is at %.4f" opt.eta
        # end
        #
        # if epoch < epochs
        #     t = t_train + t_val
        #     ETA = now() + Second(Int(round(t*(epochs-epoch))))
        #     @info "\tExpected finishing time: " * Dates.format(ETA, "yyyy/m/d HH:MM")
        # end
    end
    @info "Finished training at: " * Dates.format(now(), "yyyy/m/d HH:MM")

    return m
end


function test(m; normalize=false, enable_gpu=false, N=1_000)
    test_data = get_test_data(N)
    @info("Test data loaded.")

    if normalize
        test_data = (test_data[1] .- CIFAR10_MEANS) ./ CIFAR10_SDS, test_data[2]
    end

    # Print the final accuracy
    acc,t = @timed accuracy(test_data..., m, enable_gpu=enable_gpu)
    @info("Accuracy $acc evaluated in $t seconds.")
end

m = train(normalize=true, batchsize=400, epochs=15,
    enable_gpu=true, permute=false, schedule_lr=false)

@time acc = test(cpu(m),normalize=true, enable_gpu=false) # 14s
@time acc = test(m,normalize=true, enable_gpu=true) # 1.3

using CUDA

d = gpu(Conv((10,10), 2=>3))
d = fmap(CUDA.cu, Conv((10,10), 2=>3))

d = gpu(ConvBlock(4=>4))
typeof(d.conv.weight

d = gpu(ResBlock(4=>4=>4))
typeof(d.conv_block_1.conv.weight)

ps = params(m)
n_params = sum(map(p->prod(size(p)), ps))
BSON.@save joinpath("cifar_conv2.bson") params=ps
@info "Saved model."

# MACBOOK
# 40 min pro epoch
# 18s for acc on test -> 1000 imgs









using Plots

test_data = get_test_data()
img = test_data[1][:,:,:,1]

function img_from_float(X)
    n, m, = size(X)
    img = Array{RGB}(undef, n, m)
    for i in 1:n, j in 1:m
        img[i,j] = RGB((X[i,j,:])...)
    end
    img
end


pussy = train_set[1][1][:,:,:,10]
using BenchmarkTools
@btime random_permute(pussy)
r_pussy = random_permute(pussy)
img_from_float(pussy)
img_from_float(r_pussy)

k = 0
permuted_set = random_permute_set(train_set)

μ, σ = reshape([0.4914, 0.4822, 0.4465],1,1,3), reshape([0.2023, 0.1994, 0.2010],1,1,3)
img_from_float((img .- μ) ./ σ)

lr0 = 0.001
lr_max = 0.01
lr_end =
plot(t->(lr_max-lr0)*t / cos(π*t) + lr0, 0, 1)


train_set, val_set = get_processed_data(batchsize=128)

for i in 1:10
    display(img_from_float(permuted_set[1][1][:,:,:,i]))
end

trainimgs(CIFAR10)

Random.seed!(1)
m = ResNet9(3,10)

ps = params(m)
sum(map(p->prod(size(p)), ps))

@time accuracy(val_set..., m)
