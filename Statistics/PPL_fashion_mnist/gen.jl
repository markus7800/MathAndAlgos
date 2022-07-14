using Plots, Random, LinearAlgebra
import Distributions
# Set a random seed.
Random.seed!(3)

# Construct 30 data points for each cluster.
N = 30

# Parameters for each cluster, we assume that each cluster is Gaussian distributed in the example.
μs = [-3.5, 0.0]

# Construct the data points.
# x = mapreduce(c -> rand(Distributions.MvNormal([μs[c], μs[c]], 1. * I(2)), N), hcat, 1:2)
x = Matrix{Float64}(undef, 2, 2*N)
for i in 1:2*N
    k = (i > 30) + 1
    x[:,i] = mvnormal([μs[k], μs[k]], 1. * I(2))
end

# Visualization.
scatter(x[1,:], x[2,:], legend = false, title = "Synthetic Dataset")

using Gen

@gen function my_model(xs::Vector{Float64})
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 10), :intercept)
    for (i, x) in enumerate(xs)
        @trace(normal(slope * x + intercept, 1), "y-$i")
    end
end

function my_inference_program(xs::Vector{Float64}, ys::Vector{Float64}, num_iters::Int)
    # Create a set of constraints fixing the
    # y coordinates to the observed y values
    constraints = choicemap()
    for (i, y) in enumerate(ys)
        constraints["y-$i"] = y
    end

    # Run the model, constrained by `constraints`,
    # to get an initial execution trace
    (trace, _) = generate(my_model, (xs,), constraints)

    # Iteratively update the slope then the intercept,
    # using Gen's metropolis_hastings operator.
    for iter=1:num_iters
        (trace, _) = metropolis_hastings(trace, select(:slope))
        (trace, _) = metropolis_hastings(trace, select(:intercept))
    end

    # From the final trace, read out the slope and
    # the intercept.
    choices = get_choices(trace)
    return (choices[:slope], choices[:intercept])
end

xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]
(slope, intercept) = my_inference_program(xs, ys, 1000)
println("slope: $slope, intercept: $intercept")

@gen function gausian_mixture_model(x::Array{Float64})
    μ1 = @trace(normal(0,1), :mu1)
    μ2 = @trace(normal(0,1), :mu2)

    μ = [μ1, μ2]

    N = size(x, 2)
    for i in 1:N
        k = @trace(categorical([0.5, 0.5]), "k-$i")
        @trace(mvnormal([μ[k], μ[k]], 1. * I(2)), "x-$i")
    end
end

import ProgressMeter
function gmm_inference(x::Array{Float64}, num_iters::Int)
    N = size(x, 2)

    constraints = choicemap()
    for i in 1:N
        constraints["x-$i"] = x[:,i]
    end


    (trace, _) = generate(gausian_mixture_model, (x,), constraints)

    mu1s = Float64[]
    mu2s = Float64[]

    ProgressMeter.@showprogress for iter in 1:num_iters
        (trace, _) = metropolis_hastings(trace, select(:mu1))
        push!(mu1s, trace[:mu1])
        (trace, _) = metropolis_hastings(trace, select(:mu2))
        push!(mu2s, trace[:mu2])

        for i in 1:N
            (trace, _) = metropolis_hastings(trace, select("k-$i"))
        end
    end

    choices = get_choices(trace)
    return choices, mu1s, mu2s
end

choices, mu1s, mu2s = gmm_inference(x, 10_000)

plot(mu1s, label="μ1")
plot(mu2s, label="μ2")

choices[:mu1], choices[:mu2]
[choices["k-$i"] for i in 1:size(x,2)]

μs

categorical([0.5,0.5])


using Gen
@gen function disc_model(x::Float64)
    @param a::Float64
    @param b::Float64
    @trace(normal(a * x + b, 1.), :y)
end

init_param!(disc_model, :a, 1.)
init_param!(disc_model, :b, 0.)

disc_model(1.)

@gen function disc_model_2(x::Float64)
    @param w::Vector{Float64}
    @trace(normal(w[1] * x + w[2], 1.), :y)
end

init_param!(disc_model_2, :w, [1.,0.])
disc_model_2(1.)

using Flux
using LinearAlgebra

@gen function disc_model_3(x::Vector{Float32})
    @param W::Matrix{Float32}
    @param b::Vector{Float32}
    n = length(b)
    d = Dense(W, b, sigmoid)
    display(W)
    display(b)

    @trace(mvnormal(d(x), 1. * I(n)), :y)
end

using Random
Random.seed!(0)
n = 5
d = Dense(n, n)

init_param!(disc_model_3, :W, d.weight)
init_param!(disc_model_3, :b, d.bias)

disc_model_3(ones(Float32, n))

d = Dense(5, 5, sigmoid)
d.weight
d.
Dense()



model = Chain(Dense(5, 10, sigmoid), Dense(10, 5, sigmoid))

ps = params(model)

typeof(ps)

ps.params.dict
