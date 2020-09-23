using LinearAlgebra

abstract type Kernel end

struct FunctionKernel <: Kernel
    k::Function # inputs are vectors
end

function cov(fk::FunctionKernel, x::Vector{Float64}, x´::Vector{Float64})
    return fk.k(x,x´)
end

function cov(fk::FunctionKernel, x::Float64, x´::Float64)
    return fk.k([x],[x´])
end

# Squared Exponential
struct SE <: Kernel
    c::Float64
    l::Float64
end


function cov(k::Kernel, xs::AbstractMatrix{Float64}, xs´::AbstractMatrix{Float64})
    l = size(xs,2)
    l´ = size(xs´,2)
    K = zeros(l, l´)
    for i in 1:l, j in 1:l´
        K[i,j] = cov(k, xs[:,i], xs´[:,j])
        # K[j,i]
    end
    return K
end

function cov(se::SE, x::Vector{Float64}, x´::Vector{Float64})
    d = x - x´
    return cov(se, d'd)
end

function cov(se::SE, x::Float64, x´::Float64)
    return cov(se, (x - x´)^2)
end

function cov(se::SE, r::Float64)
    return se.c * exp(-0.5 * r / se.l)
end




# xs = collect(LinRange(-1,1,100))
#
# import GaussianProcesses
#
# se2 = GaussianProcesses.SE(0.,0.)
# me2 = GaussianProcesses.MeanZero()
# gp2 = GaussianProcesses.GP(Float64[-10.],Float64[0], me2, se2)
#
# GaussianProcesses.cov(se2, xs, xs)
#
# plot(gp2, xlims=(-3,3), obsv=false)
#
# Random.seed!(1)
# s = rand(gp2, xs, 10)
# plot!(xs, s)
#
#
# K2 = GaussianProcesses.cov(se2, xs', xs')
# isposdef(K2)
# for i in 1:100
#     K2[i,i] -= 1e-6 * 12
# end
# isposdef(K2)
#
# Σ, chol = GaussianProcesses.make_posdef!(deepcopy(K2))
#
# K2 - Σ
# isposdef(Σ)
# tr(K2) / size(K2, 1)
#
#
# K = cov(gp.kernel, xs', xs')
#
# xs' isa AbstractMatrix{Float64}
#
# GaussianProcesses.cov(se2, xs[1], xs[1])
#
# @which GaussianProcesses.cov(se2, xs', xs')
# GaussianProcesses.cov(se2, [xs[1]], [xs[1]])
#
# se2.σ2 * exp(-0.5*abs(r)^2/se2.ℓ2)
#
# se2.σ2
# se2.ℓ2
#
# r = xs[1] - xs[3]
#
# cov(se, xs[1], xs[3])
#
# sum(K .- K2)
#
# gp2.nobs
#
# using Random
# import PDMats
# xs = collect(LinRange(-1,1,100))
# K = cov(SE(1.,1.), xs', xs')
# make_posdef!(K)
# U = cholesky(K).U
#
# Kpd = PDMats.PDMat(K)
#
# Random.seed!(1)
# Z = randn(100,10)
#
# uw = PDMats.unwhiten(Kpd, Z)
#
# sum(uw .- U'Z)
