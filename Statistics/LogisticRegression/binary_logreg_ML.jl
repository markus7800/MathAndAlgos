
using LinearAlgebra
include("../../utils/posdef.jl")


struct BIN_LOGR_ML
    X::Matrix
    y::Vector
    w::Vector
    basis::Function
    λ::Float64

    function BIN_LOGR_ML(X::Matrix, y::Vector;
        basis::Function=x->vcat(1.,x), λ::Float64=0.)

        Phi = calc_ϕ(X, basis)

        w = NewtonRaphson(Phi, y, λ)
        return new(X, y, w, basis, λ)
    end
end

σ(x) = 1/(1 + exp(-x))

function calc_ϕ(X::Matrix, basis::Function)
    d = length(basis(X[1,:]))
    N = size(X,1)

    Phi = Array{Float64,2}(undef, N, d)
    for n in 1:N
        Phi[n,:] = basis(X[n,:])
    end

    return Phi
end



function predict(Phi::Matrix, w::Vector)
    p1 = σ.(Phi * w)
    ŷ = zeros(Int, size(Phi,1))
    ŷ[p1 .≥ 0.5] .= 1
    return ŷ
end

function predict(lg::BIN_LOGR_ML, X_pred)
    Phi = calc_ϕ(X_pred, lg.basis)
    return predict(Phi, lg.w)
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

        missclass = sum(y != predict(Phi, w))
        d∇ = norm(∇E)
        if d∇ < 1e-12 || missclass == 0
            @info "Iter $i: missclass = $missclass, ||∇|| = $d∇"
            break
        end
    end
    return w
end
