
using Gen
using Flux
using LinearAlgebra
include("mvbernoulli.jl")

mutable struct Encoder
    img_size::Int
    z_dim::Int
    hidden_dim::Int

    function Encoder(img_size::Int, z_dim::Int, hidden_dim::Int)
        enc = new()
        enc.img_size = img_size
        enc.z_dim = z_dim
        enc.hidden_dim = hidden_dim
        return enc
    end
end

function init_encoder_params(enc::Encoder, model)
    fc1 = Dense(enc.img_size, enc.hidden_dim, softplus)
    fc2_loc = Dense(enc.hidden_dim, enc.z_dim, identity)
    fc2_scale = Dense(enc.hidden_dim, enc.z_dim, exp)

    init_param!(model, :enc_fc1_W, fc1.weight)
    init_param!(model, :enc_fc1_b, fc1.bias)

    init_param!(model, :enc_fc2_loc_W, fc2_loc.weight)
    init_param!(model, :enc_fc2_loc_b, fc2_loc.bias)

    init_param!(model, :enc_fc2_scale_W, fc2_scale.weight)
    init_param!(model, :enc_fc2_scale_b, fc2_scale.bias)
end

# X img batches (img_size, n_imgs)
# q(z|x,θ)
@gen function guide(X::Array{Float32})
    # reconstruct Encoder from parameter store
    @param enc_fc1_W::Matrix{Float32}
    @param enc_fc1_b::Vector{Float32}
    fc1 = Dense(enc_fc1_W, enc_fc1_b, softplus)

    @param enc_fc2_loc_W::Matrix{Float32}
    @param enc_fc2_loc_b::Vector{Float32}
    fc2_loc = Dense(enc_fc2_loc_W, enc_fc2_loc_b, identity)

    @param enc_fc2_scale_W::Matrix{Float32}
    @param enc_fc2_scale_b::Vector{Float32}
    fc2_scale = Dense(enc_fc2_scale_W, enc_fc2_scale_b, exp)

    encoder_loc = Chain(fc1, fc2_loc)
    encoder_scale = Chain(fc1, fc2_scale)

    img_size, n_imgs = size(X)

    for i in 1:n_imgs
        # evaluate encoder
        z_loc = encoder_loc(X[:,i])
        z_scale = encoder_scale(X[:,i])

        if !isposdef(Diagonal(z_scale .+ 1e-3))
            println(minimum(z_scale), ", ", maximum(z_scale))
            println(minimum(Diagonal(z_scale .+ 1e-3)), ", ", maximum(Diagonal(z_scale .+ 1e-3)))
        end

        # sample latent variables
        z = @trace(mvnormal(z_loc, Diagonal(z_scale .+ 1e-3)), (:latent, i))
    end
end


mutable struct Decoder
    img_size::Int
    z_dim::Int
    hidden_dim::Int

    function Decoder(img_size::Int, z_dim::Int, hidden_dim::Int)
        dec = new()
        dec.img_size = img_size
        dec.z_dim = z_dim
        dec.hidden_dim = hidden_dim
        return dec
    end
end

function init_decoder_params(dec::Decoder, model)
    fc1 = Dense(dec.z_dim, dec.hidden_dim, softplus)
    fc2 = Dense(dec.hidden_dim, dec.img_size, sigmoid)

    init_param!(model, :dec_fc1_W, fc1.weight)
    init_param!(model, :dec_fc1_b, fc1.bias)

    init_param!(model, :dec_fc2_W, fc2.weight)
    init_param!(model, :dec_fc2_b, fc2.bias)
end

# p(x|z,θ)p(z)
@gen function model(X::Array{Float32})
    # reconstruct Decoder from parameter store
    @param dec_fc1_W::Matrix{Float32}
    @param dec_fc1_b::Vector{Float32}
    fc1 = Dense(dec_fc1_W, dec_fc1_b, softplus)

    @param dec_fc2_W::Matrix{Float32}
    @param dec_fc2_b::Vector{Float32}
    fc2 = Dense(dec_fc2_W, dec_fc2_b, sigmoid)

    decoder = Chain(fc1, fc2)

    img_size, n_imgs = size(X)
    z_dim = size(dec_fc1_W, 2)

    for i in 1:n_imgs
        z_loc = zeros(Float32, z_dim)
        z_scale = ones(Float32, z_dim)

        z = @trace(mvnormal(z_loc, Diagonal(z_scale)), (:latent, i))

        img_loc = decoder(z)
        @trace(mvbernoulli(img_loc), (:observation, i))
    end
end


import MLDatasets: MNIST
import MLDataUtils: shuffleobs, stratifiedobs
using Random

train_x, train_y = MNIST.traindata()

img_size = 28*28
train_x = Float32.(reshape(train_x, img_size, :))

n_imgs = 100

Random.seed!(0)
(X, y), = stratifiedobs((train_x, train_y), p=(n_imgs+1)/size(train_x)[2]);
X = Matrix(X)
y = Vector(y)

enc = Encoder(img_size, 50, 400)
dec = Decoder(img_size, 50, 400)

init_encoder_params(enc, guide)
init_decoder_params(dec, model)

observations = choicemap()
for i in 1:n_imgs
    observations[(:observation, i)] = X[:,i] .> 0.5
end

# guide_update = ParamUpdate(Gen.ADAM(1e-3, 0.9, 0.999, 1e-8), guide)
guide_update = ParamUpdate(Gen.FixedStepGradientDescent(1e-3), guide)
model_update = ParamUpdate(Gen.FixedStepGradientDescent(1e-3), model)
black_box_vi!(
    model, (X,), model_update,
    observations,
    guide, (X,), guide_update;
    iters=100, samples_per_iter=10, verbose=true)

mvnormal
