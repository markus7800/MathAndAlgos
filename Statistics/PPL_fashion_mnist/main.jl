


import MLDatasets: FashionMNIST
import MLDataUtils: shuffleobs, stratifiedobs
using Random

train_x, train_y = FashionMNIST.traindata()
Random.seed!(0)
(small_train_x, small_train_y), = stratifiedobs((train_x, train_y), p=0.01)


# test_x,  test_y  = FashionMNIST.testdata()

using ImageCore
FashionMNIST.convert2image(train_x[:,:,1])


using Turing
using StatsPlots
using LinearAlgebra

@model function GaussianMixtureModel_1(x, num_classes::Int)

    D, N = size(x)

    # means for classes
    # μs = zeros(Float64, D, num_classes)

    # for i in 1:num_classes
    #     μs[:, i] ~ MvNormal(zeros(D), 1.) # isonormal Σ = I
    # end

    μ ~ MvNormal(zeros(num_classes*D), 1.)

    # Equal probability for all classes
    w = ones(num_classes) ./ num_classes

    # Draw assignments for each image and generate it from a multivariate normal
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ Categorical(w) # ∈ 1..num_classes
        x[:,i] ~ MvNormal(μ[ (k[i]-1)*D+1 : k[i]*D ], 1.)
    end

    return μ
end

# x has shape 28 x 28 x n_obs
function transform_tensor(x)
    x_new = Float64.(reshape(x, 28*28, :))

    # function invphi(v)
    #     return quantile(Normal(), v)
    # end
    #
    # x_new[x_new .≤ 0.001] .= 0.001
    # x_new[x_new .≥ 0.999] .= 0.999
    #
    # x_new = invphi.(x_new)

    return x_new
end

function inv_transform_tensor(x)
    x_new = Float64.(reshape(x, 28, 28, :))

    function phi(v)
        return cdf(Normal(), v)
    end

    x_new = phi.(x_new)

    x_new[x_new .≤ 0.001] .= 0
    x_new[x_new .≥ 0.999] .= 1

    return x_new
end


x = transform_tensor(small_train_x)

gmm_model_1 = GaussianMixtureModel_1(x, 10)

gmm_sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ))
tchain = sample(gmm_model_1, gmm_sampler, 10)


# num_classes = 10
# D = 5
# a = collect(1 : D * num_classes)
# for k in 1:num_classes
#     println((k-1)*D+1, ":", (k)*D, " = ", a[(k-1)*D+1:k*D])
# end
