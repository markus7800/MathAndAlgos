
using LinearAlgebra
include("../../utils/posdef.jl")

struct MULTI_LOGR_ML
    X::Matrix
    Y::Matrix
    W::Matrix
    basis::Function
    λ::Float64

    function MULTI_LOGR_ML(X::Matrix, Y::Matrix;
        basis::Function=x->vcat(1.,x), λ::Float64=0., alg=:NR)

        Phi = calc_ϕ(X, basis)

        if alg == :GD
            W = GD(Phi, Y, λ)
        elseif alg == :NR
            W = NewtonRaphson(Phi, Y, λ)
        end
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
    P = softmax(Phi * W)
    return Int.(P .== maxs)
end

function predict(lg::MULTI_LOGR_ML, X_pred)
    Phi = calc_ϕ(X_pred, lg.basis)
    return predict(Phi, lg.W)
end

function predict_probs(lg::MULTI_LOGR_ML, X_pred)
    Phi = calc_ϕ(X_pred, lg.basis)
    P = softmax(Phi * lg.W)
    vec(maximum(P, dims=2))
end

function GD(Phi::Matrix, Y::Matrix, λ; η=0.01, max_iter=10^5)
    p = size(Phi,2)
    n_class = size(Y,2)
    W = ones(p, n_class)

    @progress for i in 1:max_iter
        Ŷ = softmax(Phi * W)
        ∇E = (Phi')*(Ŷ - Y) + λ*W
        W -= η * ∇E

        missclass = sum(Y .!= predict(Phi, W)) / n_class
        d∇ = norm(∇E)
        if d∇ < 1e-12 || missclass == 0
            @info "Iter $i: missclass = $missclass, ||∇|| = $d∇"
            break
        elseif i == max_iter
            @warn "Max iter reached $i: missclass = $missclass, ||∇|| = $d∇"
        end
    end
    return W
end

function NewtonRaphson(Phi::Matrix, Y::Matrix, λ; max_iter=10^3)
    p = size(Phi,2)
    n_class = size(Y,2)
    N = p*n_class

    W = ones(p, n_class)
    ∇∇E = zeros(N, N)
    for i in 1:max_iter
        Ŷ = softmax(Phi * W)
        ∇E = (Phi')*(Ŷ - Y) + λ*W
        ∇E = reshape(∇E, N)

        for i in 1:n_class, j in 1:n_class
            R = Diagonal(Ŷ[:,i] .* (Int(i==j) .- Ŷ[:,j]))
            ∇∇E[p*(i-1)+1 : p*i, p*(j-1)+1 : p*j] .= Phi'R*Phi + λ*I(p)
        end

        make_symmetric!(∇∇E)
        make_posdef!(∇∇E)
        W -= reshape(cholesky(∇∇E) \ ∇E, p, n_class)

        missclass = sum(Y .!= predict(Phi, W)) / n_class
        d∇ = norm(∇E)
        if d∇ < 1e-12 || missclass == 0
            @info "Iter $i: missclass = $missclass, ||∇|| = $d∇"
            break
        end
    end
    return W
end
