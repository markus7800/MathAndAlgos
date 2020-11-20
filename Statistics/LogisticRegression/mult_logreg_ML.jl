
using LinearAlgebra
include("../../utils/posdef.jl")

struct MULTI_LOGR_ML
    X::Matrix
    Y::Matrix
    W::Vector
    basis::Function
    λ::Float64

    function MULTI_LOGR_ML(X::Matrix, Y::Matrix;
        basis::Function=x->vcat(1.,x), λ::Float64=0.)

        Phi = calc_ϕ(X, basis)

        W = GD(Phi, Y, λ)
        return new(X, Y, W, basis, λ)
    end
end


function calc_ϕ(X::Matrix, basis::Function)
    d = length(basis(X[1,:]))
    N = size(X,1)

    Phi = Array{Float64,2}(undef, N, d)
    for n in 1:N
        Phi[n,:] = basis(X[n,:])
    end

    return Phi
end

function softmax(A::Matrix)
    B = exp.(A)
    B ./ sum(B, dims=2)
end

function predict(Phi::Matrix, W::Matrix)
    p0 = softmax(Phi * W)
    ŷ = zeros(size(Phi,1))
    return ŷ
end

function predict(lg::BIN_LOGR_ML, X_pred)
    Phi = calc_ϕ(X_pred, lg.basis)
    p0 = σ.(Phi * lg.w)
    ŷ = zeros(size(Phi,1))
    ŷ[p0 .< 0.5] .= 1
    return ŷ
end

function NewtonRaphson(Phi::Matrix, y::Vector, λ; max_iter=10^3)
    p = size(Phi,2)
    w = ones(p)

    for i in 1:max_iter
        ŷ = σ.(Phi * w)
        ∇E = (Phi')*(ŷ - y) + λ*w
        R = Diagonal(ŷ .* (1 .- ŷ))
        ∇∇E = Phi'R*Phi + λ*I(p)
        make_symmetric!(∇∇E)
        make_posdef!(∇∇E)
        w -= cholesky(∇∇E) \ ∇E

        # z = Phi*w - R \ (ŷ - y)
        # w == ∇∇E \ Phi'R*z

        missclass = sum(y .!= predict(Phi, w))
        d∇ = norm(∇E)
        if d∇ < 1e-12
            @info "Iter $i: missclass = $missclass, ||∇|| = $d∇"
            break
        end
    end
    return w
end
