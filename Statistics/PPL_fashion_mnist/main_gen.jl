
import MLDatasets: FashionMNIST
import MLDataUtils: shuffleobs, stratifiedobs
using Random

train_x, train_y = FashionMNIST.traindata()
train_x = Float64.(reshape(train_x, 28*28, :))


Random.seed!(0)
(small_train_x, small_train_y), = stratifiedobs((train_x, train_y), p=0.01)


# test_x,  test_y  = FashionMNIST.testdata()

using ImageCore
FashionMNIST.convert2image(train_x[:,1])

names = FashionMNIST.classnames()

tshirt = 0
sneaker = 7
names[tshirt+1]
names[sneaker+1]

two_labels = (train_y .== tshirt) .| (train_y .== sneaker)

train_x_2 = train_x[:,two_labels]
train_y_2 = train_y[two_labels]

FashionMNIST.convert2image(train_x_2[:,4])


Random.seed!(0)
(small_train_x, small_train_y), = stratifiedobs((train_x_2, train_y_2), p=0.01)

using LinearAlgebra

using Gen

@gen function gausian_mixture_model_1(x::Array{Float64}, y::Vector{Int})
    d, N = size(x)
    μ1 = @trace(mvnormal(fill(0.5, d), 0.5 * I(d)), "mu-1")
    μ2 = @trace(mvnormal(fill(0.5, d), 0.5 * I(d)), "mu-2")

    μ = [μ1, μ2]

    for i in 1:N
        #@trace(categorical([0.5, 0.5]), "k-$i")
        k = (y[i] == sneaker) + 1
        @trace(mvnormal(μ[k], 0.5 * I(d)), "x-$i")
    end
end

import ProgressMeter
function gmm_inference_1(x::Array{Float64}, y::Vector{Int}, num_iters::Int)
    d, N = size(x)

    constraints = choicemap()
    for i in 1:N
        constraints["x-$i"] = x[:,i]
    end

    (trace, _) = generate(gausian_mixture_model_1, (x,y), constraints)

    # mu1s = Float64[]
    # mu2s = Float64[]

    ProgressMeter.@showprogress for iter in 1:num_iters
        (trace, _) = metropolis_hastings(trace, select("mu-1"))
        (trace, _) = metropolis_hastings(trace, select("mu-2"))

        # for i in 1:N
        #     (trace, _) = metropolis_hastings(trace, select("k-$i"))
        # end
    end

    choices = get_choices(trace)
    return choices
end

x = Array(small_train_x)
y = Vector(small_train_y)
choices = gmm_inference_1(x, y, 1000)

mu_1 = clamp.(choices["mu-1"], 0, 1)
FashionMNIST.convert2image(mu_1)
mu_2 = clamp.(choices["mu-2"], 0, 1)
FashionMNIST.convert2image(mu_2)
