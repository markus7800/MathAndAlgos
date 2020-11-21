using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, flatten
import Functors: @functor
using Metalhead
using Metalhead: trainimgs
using Images: channelview
using Random
using Base.Iterators: partition
using Statistics



# Function to convert the RGB image to Float32 Arrays
function getarray(X)
    Float32.(permutedims(channelview(X), (2, 3, 1)))
end

function get_processed_data(;batchsize=128, splitr_ = 0.1)
    # Fetching the train and validation data and getting them into proper shape
    X = trainimgs(CIFAR10)
    imgs = [getarray(X[i].img) for i in 1:40000]
    #onehot encode labels of batch

    labels = onehotbatch([X[i].ground_truth.class for i in 1:40000],1:10)

    train_pop = Int((1-splitr_)* 40000)
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, batchsize)])
    valset = collect(train_pop+1:40000)
    valX = cat(imgs[valset]..., dims = 4) |> gpu
    valY = labels[:, valset] |> gpu

    val = (valX,valY)
    return train, val
end

function get_test_data()
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)

    # CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
    testimgs = [getarray(test[i].img) for i in 1:1000]
    testY = onehotbatch([test[i].ground_truth.class for i in 1:1000], 1:10) |> gpu
    testX = cat(testimgs..., dims = 4) |> gpu

    test = (testX,testY)
    return test
end



struct ConvBlock
    conv::Conv
    bn::BatchNorm
    mp::MaxPool
    pool::Bool
    σ
    function ConvBlock(ch::Pair{Int,Int}, size=(3,3), pad=(1,1); pool=false)
        conv = Conv(size, ch, pad=pad)
        bn = BatchNorm(ch[2])
        mp = MaxPool((2,2))
        return new(conv,bn,mp,pool,relu)
    end
end

function (CB::ConvBlock)(x)
    x = CB.conv(x)
    x = CB.bn(x)
    if CB.pool
        x = CB.mp(x)
    end
    return CB.σ.(x)
end

@functor ConvBlock

struct ResBlock
    conv_block_1::ConvBlock
    conv_block_2::ConvBlock
    function ResBlock(ch::Pair{Int,Pair{Int,Int}})
        inc, ch2 = ch
        midc, out = ch2
        new(ConvBlock(inc=>midc), ConvBlock(midc=>out))
    end
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





accuracy(x, y, m) = mean(onecold(cpu(m(x)), 1:10) .== onecold(cpu(y), 1:10))

function train(; epochs=8, normalize=false, batchsize=400)

    @info("Load training data")
    train_set, val_set = get_processed_data(batchsize=batchsize)
    if normalize
        @info("Normalizing training data.")
        μ  = reshape(Float32[0.4914, 0.4822, 0.4465],1,1,3)
        σ = reshape(Float32[0.2023, 0.1994, 0.2010],1,1,3)
        map!(x -> ((x[1] .- μ) ./ σ, x[2]), train_set, train_set)
        val_set = (val_set[1] .- μ) ./ σ, val_set[2]
    end
    display(typeof(train_set[1][1]))
    display(typeof(val_set[1]))

    @info("Constructing Model")
    Random.seed!(1)
    m = ResNet9(3,10)

    sqnorm(x) = sum(abs2, x)
    L2_penalty(m) = sum(sqnorm, params(m))
    λ = 1f-4
    loss(x, y) = logitcrossentropy(m(x), y) + λ * L2_penalty(m)

    #throttle_s = 10
    #evalcb = throttle(() -> @show(loss(val...)), throttle_s)

    opt = ADAM(0.001)
    Flux.@epochs epochs begin
        v,t, = @timed Flux.train!(loss, params(m), train_set, opt)
        @info "Finished in $(Int(round(t/60))) minutes."
        v,t, = @timed show(accuracy(val_set..., m))
        @info "Validated in $(Int(round(t))) seconds."
    end

    return m
end


function test(m)
    test_data = get_test_data()

    # Print the final accuracy
    @show(accuracy(test_data..., m))
end


m = train(normalize=true, batchsize=128)
@time test(m)

# MACBOOK
# 18 min pro epoch
# 15s for acc on test -> 1000 imgs


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

img_from_float(img)

μ, σ = reshape([0.4914, 0.4822, 0.4465],1,1,3), reshape([0.2023, 0.1994, 0.2010],1,1,3)
img_from_float((img .- μ) ./ σ)

lr0 = 0.001
lr_max = 0.01
lr_end =
plot(t->(lr_max-lr0)*t / cos(π*t) + lr0, 0, 1)


train_set, val_set = get_processed_data(batchsize=128)

trainimgs(CIFAR10)
