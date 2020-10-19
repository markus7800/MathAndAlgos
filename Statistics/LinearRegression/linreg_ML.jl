using LinearAlgebra
using Plots

struct LM_ML
    X::Matrix
    y::Vector
    w::Vector
    basis::Function
    λ::Float64

    function LM_ML(X::Matrix, y::Vector;
        basis::Function=x->vcat(1.,x), λ::Float64=0.)

        Phi = calc_ϕ(X, basis)
        d = size(Phi,2)

        A = λ*I(d) + Phi'Phi
        b = Phi'y

        w = A \ b

        return new(X, y, w, basis, λ)
    end
end

function LM_ML(X::Vector, y::Vector; kw...)
    LM_ML(reshape(X,:,1), y; kw...)
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

function predict(lm::LM_ML, X_pred::Matrix)
    Φ = calc_ϕ(X_pred, lm.basis)
    return vec(Φ*lm.w)
end


function predict(lm::LM_ML, X_pred::Vector)
    predict(lm, reshape(X_pred,:,1))
end

function plot_lm(lm::LM_ML; kw...)
    @assert size(lm.X,2) == 1
    x_vec = vec(lm.X)
    scatter(x_vec, lm.y, label="data"; kw...)
    ts = collect(LinRange(minimum(x_vec), maximum(x_vec), 500))
    plot!(ts, predict(lm, ts), label="maximum likelihood prediction")
end
