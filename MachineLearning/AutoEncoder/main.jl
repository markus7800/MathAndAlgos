using Flux
using Flux.Data.MNIST
using Base.Iterators: partition
using Flux.Losses: mse
using Statistics
using Plots
using Random

function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, prod(size(X[1])), length(idxs))
    for i in 1:length(idxs)
        X_batch[:, i] = reshape(Float32.(X[idxs[i]]),:)
    end
    # Y_batch = Y[idxs]
    return (X_batch, X_batch)
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

train_set, test_set = get_processed_data(64)
train_labels =  MNIST.labels(:train)
test_labels =  MNIST.labels(:test)

mutable struct AutoEncoder
    encoder::Chain
    decoder::Chain

    function AutoEncoder(dim::Int)
        this = new()
        this.encoder = Chain(
            Dense(784, 200, tanh),
            Dense(200, 100, tanh),
            Dense(100, 50, tanh),
            Dense(50, 5*dim, tanh),
            Dense(5*dim, dim, sigmoid)
        )
        this.decoder = Chain(
            Dense(dim, 5*dim, tanh),
            Dense(5*dim, 50, tanh),
            Dense(50, 100, tanh),
            Dense(100, 200, tanh),
            Dense(200, 784, sigmoid)
        )

        return this
    end
end



function loss(x, y)
    ŷ = ae.decoder(ae.encoder(x))
    return mean(sum((y .- ŷ).^2, dims=1))
end




Random.seed!(1)
ae = AutoEncoder(2)
opt = ADAM(0.001)

loss(test_set...)

for epoch in 1:20
    Flux.train!(loss, params(ae.encoder, ae.decoder), train_set, opt)

    l_train = mean(loss(x[1], x[2]) for x in train_set)
    l_test = loss(test_set...)

    @info("Epoch $epoch: $l_train $l_test")
end

losses = sum((test_set[1] .- ae.decoder(ae.encoder(test_set[1]))) .^2, dims=1)[1,:]

best_encodings = Dict()
for label in 0:9
    ixs = findall(test_labels .== label)
    m, i = findmin(losses[test_labels .== label])
    println(label, ": ", ixs[i], " ", m)
    best_encodings[label] = ixs[i]
end


function plot_number(img; color=:grays)
    heatmap(reshape(img, 28, 28), legend=false, aspect_ratio = 1,
        color=color, grid=false, axis=false, yflip = true)
end


for label in 0:9
    img = test_set[1][:,best_encodings[label]]
    p1 = plot_number(img)
    p2 = plot_number(ae.decoder(ae.encoder(img)))
    p = plot(p1, p2)
    display(p)
end

img1 = test_set[1][:,best_encodings[0]]
encoding1 = ae.encoder(img1)
img2 = test_set[1][:,best_encodings[9]]
encoding2 = ae.encoder(img2)


nframes = 100
anim = Animation()
for t in LinRange(0, 1, nframes)
    latent = encoding1 + t * (encoding2 - encoding1)
    real = ae.decoder(latent)
    p = plot_number(real, color=cgrad([:white, :black]))
    frame(anim, p)
end
gif(anim, "tmp.gif")

latent = reduce(hcat, [ae.encoder(x[1]) for x in train_set])

function plot_latent(latent, best_encodings, point=nothing)
    p = scatter(latent[1,:], latent[2,:], legend=false, aspect_ratio=1,
        markercolor=train_labels, markerstrokecolor=train_labels,
        grid=false, xlims=(0,1), ylims=(0,1), axis=false, palette=palette(:tab10));
    for label in 0:9
        img = test_set[1][:,best_encodings[label]]
        x,y = ae.encoder(img)
        #scatter!([x], [y], color=:black, markershape=:x)
        annotate!(x,y,text(string(label)))
    end
    if !isnothing(point)
        scatter!([point[1]], [point[2]], markercolor=:black, markersize=10, alpha=0.5)
    end

    return p
end

img1 = test_set[1][:,best_encodings[1]]

p1 = plot_latent(latent, best_encodings, (0.5,0.5))
p2 = plot_number(img1, color=cgrad([:white, :black]))
plot(p1, p2)

M = [(a + b) % 10 for a in 0:9, b in 0:9]

seq = []
ixs = ones(Int, 10)

a = 1
while true
    ixs[a] += 1
    b = ixs[a]
    b > 10 && break

    push!(seq, M[a,1]=>M[a,b])
    println(seq[end])
    a = M[a,b] + 1
end

nframes = 100
freeze_frames = 10
anim = Animation()
normal_color = cgrad([:white, :black])
freeze_color = cgrad([:white, :black])

@progress for (a,b) in seq
    img1 = test_set[1][:,best_encodings[a]]
    encoding1 = ae.encoder(img1)
    img2 = test_set[1][:,best_encodings[b]]
    encoding2 = ae.encoder(img2)

    # for t in 1:freeze_frames
    #     p = plot_number(img1, color=freeze_color)
    #     frame(anim, p)
    # end

    for t in LinRange(0, 1, nframes)
        latent_point = encoding1 + t * (encoding2 - encoding1)
        real = ae.decoder(latent_point)
        p1 = plot_latent(latent, best_encodings, latent_point)
        p2 = plot_number(real, color=normal_color)
        p = plot(p1,p2)
        frame(anim, p)
    end
end

gif(anim, "tmp.gif")
