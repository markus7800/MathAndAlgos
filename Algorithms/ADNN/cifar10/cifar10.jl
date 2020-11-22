using Flux
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, flatten
import Functors: @functor
using Metalhead
using Metalhead: trainimgs
using Images: channelview
using Random
using Base.Iterators: partition
using Statistics
using BSON
using Plots
using StatsBase


# Function to convert the RGB image to Float32 Arrays
function getarray(X)
    Float32.(permutedims(channelview(X), (2, 3, 1)))
end

# statistics
const μ  = reshape(Float32[0.4914, 0.4822, 0.4465],1,1,3)
const σ = reshape(Float32[0.2023, 0.1994, 0.2010],1,1,3)

function get_processed_data(;batchsize=128, splitr_ = 0.1, N=40_000)
    # Fetching the train and validation data and getting them into proper shape
    X = trainimgs(CIFAR10)
    imgs = [getarray(X[i].img) for i in 1:N] # 50_000 available
    #onehot encode labels of batch

    labels = onehotbatch([X[i].ground_truth.class for i in 1:N],1:10)

    train_pop = Int((1-splitr_)* N)
    train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, batchsize)])
    valset = collect(train_pop+1:N)
    valX = cat(imgs[valset]..., dims = 4) |> gpu
    valY = labels[:, valset] |> gpu

    val = (valX,valY)
    return train, val
end

function get_test_data(N=1000)
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)

    # CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
    testimgs = [getarray(test[i].img) for i in 1:N] # 10_000 available
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


function cos_annealing(f0, f1, pct)
    # pct from 0 to 1
    cos_out = cos(π * pct) + 1 # = 2 if pct = 0; 0 if pct = 1
    return f1 + (f0 - f1) * cos_out / 2
end

mutable struct OneCycleSchedular
    # first phase increase to max_lr
    # second phase decrease to ≈ 0
    max_lr # learning rate at pct_start * total_steps
    start_lr
    end_lr
    pct_start # start of second phase
    pct_start_step
    total_steps
    base_momentum # momentum at pct_start * total_steps
    max_momentum # momentum at start and end
    current_step
    function OneCycleSchedular(;max_lr, epochs, batches,
        start_lr=max_lr/25, end_lr=max_lr/10e4, pct_start=0.3,
        base_momentum=0.85, max_momentum=0.95)

        this  = new()

        this.total_steps = epochs * batches
        this.pct_start = pct_start
        this.pct_start_step = this.total_steps * pct_start

        this.start_lr = start_lr
        this.max_lr = max_lr
        this.end_lr = end_lr

        this.base_momentum = base_momentum
        this.max_momentum = max_momentum

        this.current_step = 0

        return this
    end
end

function step!(ocs::OneCycleSchedular)
    ocs.current_step += 1
end

function get_lr(ocs::OneCycleSchedular)
    cs = ocs.current_step
    if cs ≤ ocs.pct_start_step
        # first phase: increasing
        pct = cs / ocs.pct_start_step # ∈ [0,1]
        lr0 = ocs.start_lr
        lr1 = ocs.max_lr

        m0 = ocs.base_momentum
        m1 = ocs.max_momentum
    else
        # second phase: decreasing
        pct = (cs - ocs.pct_start_step) / (ocs.total_steps - ocs.pct_start_step) # ∈ [0,1]
        lr0 = ocs.max_lr
        lr1 = ocs.end_lr

        m0 = ocs.max_momentum
        m1 = ocs.base_momentum
    end

    return cos_annealing(lr0, lr1, pct)# , cos_annealing(m0, m1, pct)
end

function plot_schedule(;kw...)
    scheduler = OneCycleSchedular(;kw...)
    lrs = []
    for i in 0:scheduler.total_steps
        push!(lrs, get_lr(scheduler))
        step!(scheduler)
    end
    plot(lrs)
end

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
    N = length(trainset)
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

# x is one big batch
function accuracy(x, y, m; batchsize=100)
    N = size(x,4)
    s = 0
    # batch to decrease ram pressure on macbook
    for is in partition(1:N, batchsize)
        s += sum(onecold(cpu(m(x[:,:,:,is])), 1:10) .== onecold(cpu(y[is]), 1:10))
    end
    return s/N
end

function train(; epochs=8, normalize=false, batchsize=400, permute=true, schedule_lr=true)

    @info("Load training data.")
    train_set, val_set = get_processed_data(batchsize=batchsize)
    if normalize
        @info("Normalizing training data.")
        map!(x -> ((x[1] .- μ) ./ σ, x[2]), train_set, train_set)
        val_set = (val_set[1] .- μ) ./ σ, val_set[2]
    end
    @info("Constructing Model.")
    Random.seed!(1)
    m = ResNet9(3,10)

    sqnorm(x) = sum(abs2, x)
    L2_penalty(m) = sum(sqnorm, params(m))
    λ = 1f-4
    loss(x, y) = logitcrossentropy(m(x), y) + λ * L2_penalty(m)

    batches = length(train_set)
    @info("Number of epochs: $epochs.")
    @info("Number of batches: $batches.")
    ocs = OneCycleSchedular(max_lr=0.01, epochs=epochs, batches=batches)
    if schedule_lr
        opt = ADAM(get_lr(ocs))
        function cb()
            step!(ocs)
            opt.eta = get_lr(ocs)
        end
    else
        opt = ADAM(0.001)
        cb = () -> ()
    end

    Flux.@epochs epochs begin
        if permute
            epoch_set, count, t = random_permute_set(train_set)
            @info "Randomly permuted $count images in $t seconds."
        else
            epoch_set = train_set
        end

        v,t, = @timed Flux.train!(loss, params(m), epoch_set, opt, cb=cb)
        @info "Finished in $(Int(round(t/60))) minutes."

        v,t, = @timed (@info "Accuracy on validation: $(accuracy(val_set..., m))")
        @info "Validated in $(Int(round(t))) seconds."

        if schedule_lr
            @info "Learning rate is at $(opt.eta)"
        end
    end

    return m
end


function test(m; normalize=false)
    test_data = get_test_data()

    if normalize
        test_data = (test_data[1] .- μ) ./ σ, test_data[2]
    end
    # Print the final accuracy
    @show(accuracy(test_data..., m))
end

plot_schedule(max_lr=0.01, epochs=, batches=282)

m = train(normalize=true, batchsize=128, epochs=8)
@time acc = test(m,normalize=true)

ps = params(m)
n_params = sum(map(p->prod(size(p)), ps))
using BSON
BSON.@save joinpath("Algorithms/ADNN/cifar10/cifar_conv.bson") params=ps

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
