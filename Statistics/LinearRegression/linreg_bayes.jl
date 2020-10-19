using LinearAlgebra
using Plots
using Distributions

struct LM_Bayes
    X::Matrix
    y::Vector
    w::Vector # MAP
    basis::Function
    α::Float64 # mean-zero isometric prior
    β::Union{Float64,Nothing}
    m::Vector
    S::Matrix
    S_inv::Matrix

    function LM_Bayes(X::Matrix, y::Vector;
        basis::Function=x->vcat(1.,x), α::Float64=0., β=nothing)

        Phi = calc_ϕ(X, basis)
        d = size(Phi,2)


        if β != nothing
            S_inv = α*I(d) + β*Phi'Phi
            S = inv(S_inv)
            b = Phi'y
            m = β*S*b
        else
            error("PANIC")
        end

        return new(X, y, m, basis, α, β, m, S, S_inv)
    end
end


function LM_Bayes(X::Vector, y::Vector; kw...)
    LM_Bayes(reshape(X,:,1), y; kw...)
end

function plot_lm(lm::LM_Bayes)
    @assert size(lm.X,2) == 1
    plot(t -> lm.w[1] + lm.w[2]*t, ribbon=t -> 1.96 * √variance(lm, t),
        label="maximum likelihood with 95% confidence", xlims=(minimum(vec(lm.X)), maximum(vec(lm.X))))
    scatter!(vec(lm.X), lm.y, label="data")
    # plot!(t -> lm.w[1] + lm.w[2]*t + 1.96 * √variance(lm, t), lc=1, label="")
    # plot!(t -> lm.w[1] + lm.w[2]*t - 1.96 * √variance(lm, t), lc=1, label="")
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

function variance(lm::LM_Bayes, x)
    return 1/lm.β + lm.basis(x)'lm.S*lm.basis(x)
end

function sample(lm::LM_Bayes, n=1)
    w = rand(MvNormal(lm.m,lm.S), n)
end
