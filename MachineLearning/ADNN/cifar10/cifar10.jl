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

include("lr_schedular.jl")
include("random_permute.jl")


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

function get_processed_data(;batchsize=128, splitr_ = 0.1, N=50_000)
    # Fetching the train and validation data and getting them into proper shape
    X = trainimgs(CIFAR10)
    imgs = [getarray(X[i].img) for i in 1:N] # 50_000 available
    #onehot encode labels of batch

    labels = onehotbatch([X[i].ground_truth.class for i in 1:N],1:10)

    train_pop = Int((1-splitr_) * N)
    train = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:train_pop, batchsize)]
    val = [(cat(imgs[i]..., dims=4), labels[:,i]) for i in partition(train_pop+1:N, batchsize)]

    return train, val
end

function get_test_data(;N=10_000, batchsize=400)
    # Fetch the test data from Metalhead and get it into proper shape.
    test = valimgs(CIFAR10)

    # CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs
    testimgs = [getarray(test[i].img) for i in 1:N] # 10_000 available
    labels = onehotbatch([test[i].ground_truth.class for i in 1:N], 1:10)
    test = [(cat(testimgs[i]..., dims = 4), labels[:,i]) for i in partition(1:N, batchsize)]

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

include("show.jl")

# X is batched (list of tuples), ram friendly
function accuracy(X, m)
    N = sum(map(b -> size(b[1],4), X))
    s = 0
    @progress for batch in X
        s += sum(onecold(cpu(m(batch[1])), 1:10) .== onecold(cpu(batch[2]), 1:10))
    end
    return s/N
end

function train(; epochs=8, normalize=false, batchsize=400, λ = 1f-4,
                permute=true, schedule_lr=true, enable_gpu=false)

    Random.seed!(1)

    @info("Load training data.")
    train_set, val_set = get_processed_data(batchsize=batchsize)

    if normalize
        @info("Normalizing training data.")
        map!(x -> ((x[1] .- CIFAR10_MEANS) ./ CIFAR10_SDS, x[2]), train_set, train_set)
        map!(x -> ((x[1] .- CIFAR10_MEANS) ./ CIFAR10_SDS, x[2]), val_set, val_set)
    end


    @info("Constructing Model.")

    Random.seed!(1)
    m = ResNet9(3,10)
    if enable_gpu
        m = gpu(m)
        # leave train set on cpu for permuting
        val_set = gpu.(val_set)
    end

    sqnorm(x) = sum(abs2, x)
    L2_penalty(m) = sum(sqnorm, params(m))
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

    best_acc_val = -Inf
    acc_vals = []
    acc_trains = []

    @info "Started training at: " * Dates.format(now(), "yyyy/m/d HH:MM")
    Flux.@epochs epochs begin
        if permute
            # do preprocessing on CPU
            epoch_set, count, t = random_permute_set(train_set)
            @info @sprintf "\tRandomly permuted %d images in %.2f seconds." count t
        else
            epoch_set = train_set
        end

        if enable_gpu
            epoch_set = gpu.(epoch_set)
        end

        v, t_epoch, = @timed Flux.train!(loss, params(m), epoch_set, opt, cb=cb)
        @info "\tEpoch finished in $(Int(round(t_epoch/60))) minutes."

        acc_train, t_train, = @timed accuracy(epoch_set, m)
        @info @sprintf "\tAccuracy on train: %.4f (%.2f seconds)." acc_train t_train
        push!(acc_trains, acc_train)

        acc_val, t_val, = @timed accuracy(val_set, m)
        @info @sprintf "\tAccuracy on validation: %.4f (%.2f seconds)." acc_val t_val
        push!(acc_vals, acc_val)
        if acc_val > best_acc_val
            best_acc_val = acc_val
            BSON.@save "Algorithms/ADNN/cifar10/temp.bson" model=cpu(m)
            @info "\tBest accuracy so far. Model saved."
        end

        if schedule_lr
            @info @sprintf "\tLearning rate is at %.4f" opt.eta
        end
    end
    @info "Finished training at: " * Dates.format(now(), "yyyy/m/d HH:MM")

    return m, acc_trains, acc_vals
end


function test(m; normalize=false, enable_gpu=false, N=10_000, batchsize=128)
    test_data = get_test_data(N=N, batchsize=batchsize)
    @info("Test data loaded.")

    if normalize
        map!(x -> ((x[1] .- CIFAR10_MEANS) ./ CIFAR10_SDS, x[2]), test_data, test_data)
    end

    if enable_gpu
        test_data = gpu(test_data)
    end

    # Print the final accuracy
    acc,t = @timed accuracy(test_data, m)
    @info("Accuracy $acc evaluated in $t seconds.")
end

m, acc_trains, acc_vals = train(normalize=true, batchsize=400, epochs=30,
    enable_gpu=true, permute=true, schedule_lr=true)

@time acc = test(m, normalize=true, enable_gpu=true)

m2 = BSON.load("Algorithms/ADNN/cifar10/cifar10_resnet9.bson")[:model]
m2 = gpu(m2)
@time test(m2, normalize=true, enable_gpu=true)

using Plots
plot(acc_trains, label="Training", ylabel="Accuracy", xlabel="Epoch", title="Cifar10 - Resnet9");
plot!(acc_vals, label="Validation", legend=:topleft)
savefig("Algorithms/ADNN/cifar10/accuracy_resnet_9.pdf")
savefig("Algorithms/ADNN/cifar10/accuracy_resnet_9.svg")

# using Images
function img_from_float(X)
    n, m, = size(X)
    img = Array{RGB}(undef, n, m)
    for i in 1:n, j in 1:m
        img[i,j] = RGB((X[i,j,:])...)
    end
    img
end

test_set = get_test_data(N=10_000, batchsize=100)
test_set_normed = map(x -> ((x[1] .- CIFAR10_MEANS) ./ CIFAR10_SDS, x[2]), test_set)
pred = map(b -> m2(b[1]), test_set_normed)
pred_ps = cat(softmax.(pred)..., dims=2)
pred_class = onecold(pred_ps, 1:10)
true_class = vcat(map(b -> onecold(b[2], 1:10), test_set_normed)...)

sum(pred_class .== true_class)
wronga = findall(pred_class .!= true_class)
righta = findall(pred_class .== true_class)

test_set_complete = Array{Float32}(undef, 32, 32, 3, 10_000)
for (i,is) in enumerate(partition(1:10_000, 100))
    global test_set_complete[:,:,:,is] = test_set[i][1]
end

classes = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]

function plot_prediction(i)
    img = img_from_float(test_set_complete[:,:,:,i])
    p = plot(img, xaxis=nothing, yaxis=nothing, border=:none)
    xlabel!(classes[true_class[i]])
    c = pred_class[i]
    annotate!(16, -2, @sprintf "%s (%.2f %%)" classes[c] (pred_ps[c,i]*100))
    p2 = bar(reverse(pred_ps[:,i]), orientation=:horizontal, legend=false)
    xlims!((0,1.))
    yticks!((1:10, reverse(classes)))
    plot(p,p2)
end

function plot_classification(set)
    size = 32
    pad = 4
    max_count = maximum(StatsBase.counts(pred_class[set]))
    p = plot(
        xlims=(-100, max_count*(size+pad)),
        size=((max_count*(size+pad)+100)*2, (10*(size+pad) + 100)*2),
        xaxis=false, yaxis=false, border=:none)

    counts = zeros(Int, 10)

    for c in 1:10
        y = (c-1)*(size+pad)
        annotate!(-75, y+size/2, classes[c])
    end

    @progress for j in set
        img = img_from_float(test_set_complete[:,:,:,j])
        c = pred_class[j]
        y = (c-1)*(size+pad)

        i = counts[c]
        x = (i-1)*(size+pad)
        plot!(x:x+size, y:y+size, img)

        counts[c] += 1
    end
    p
end

function make_anim(set)
    anim = Animation()
    @progress for i in set
        frame(anim, plot_prediction(i))
    end
    return anim
end



wronga_anim = make_anim(sample(wronga,200))
#gif(wronga_anim, "wrong_cifar10.gif", fps=1)
mp4(wronga_anim, "wrong_cifar10.mp4", fps=1)


righta_anim = make_anim(sample(righta, 200))
#gif(righta_anim, "wrong_cifar10.gif", fps=1)
mp4(righta_anim, "right_cifar10.mp4", fps=1)

plot_prediction(1)

img = img_from_float(test_set_complete[:,:,:,i])


Random.seed!(1)
wronga_sub = vcat([sample(wronga[pred_class[wronga] .== i], 10, replace=false) for i in 1:10]...)
plot_classification(wronga_sub)
savefig("wronga100.png")

Random.seed!(1)
righta_sub = vcat([sample(righta[pred_class[righta] .== i], 10, replace=false) for i in 1:10]...)
plot_classification(righta_sub)
savefig("righta100.png")
