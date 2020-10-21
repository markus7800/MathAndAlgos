using LinearAlgebra
using Plots
using Distributions
include("../../utils/posdef.jl")


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
            make_posdef!(S_inv)
            U = inv(cholesky(S_inv).U)
            S = U*U' # inv(S_inv)
            make_posdef!(S)
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

function predict(lm::LM_Bayes, X_pred::Matrix, w=lm.w)
    Φ = calc_ϕ(X_pred, lm.basis)
    return vec(Φ*w)
end

function predict(lm::LM_Bayes, X_pred::Vector, w=lm.w)
    predict(lm, reshape(X_pred,:,1), w)
end

function sample(lm::LM_Bayes, n=1)
    w = rand(MvNormal(lm.m,lm.S), n)
end

function plot_lm(lm::LM_Bayes; xlims=nothing, kw...)
    @assert size(lm.X,2) == 1
    x_vec = vec(lm.X)
    if xlims == nothing
        ts = collect(LinRange(minimum(x_vec), maximum(x_vec), 500))
    else
        x0, x1 = xlims
        ts = collect(LinRange(x0, x1, 500))
    end
    σs = map(t->1.96 * √variance(lm, t), ts)
    plot(ts, predict(lm, ts), ribbon=σs,
        label="MAP with 95% confidence", lc=:black)
    scatter!(vec(lm.X), lm.y, label="data"; kw...)
end

function plot_w(lm::LM_Bayes; xlims=nothing, ylims=nothing)
    if length(lm.w) == 2
        D = MvNormal(lm.m,lm.S)
        if xlims == nothing
            xlims = (lm.m[1] - 2.56 * lm.S[1,1], lm.m[1] + 2.56 * lm.S[1,1])
        end
        if ylims == nothing
            ylims = (lm.m[2] - 2.56 * lm.S[2,2], lm.m[2] + 2.56 * lm.S[2,2])
        end

        xs = LinRange(xlims..., 500)
        ys = LinRange(ylims..., 500)
        heatmap(xs, ys, (x,y)->pdf(D, [x,y]))
        scatter!([(lm.m[1], lm.m[2])], markershape=:x, label="", mc=:black)
    end
end
