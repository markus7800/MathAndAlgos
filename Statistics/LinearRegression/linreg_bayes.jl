using LinearAlgebra
using Plots
using Distributions
include("../../utils/posdef.jl")


struct LM_Bayes
    X::Matrix
    y::Vector
    N::Int
    w::Vector # MAP
    basis::Function
    α::Float64 # mean-zero isometric prior
    β::Union{Float64,Nothing}
    m::Vector
    S::Matrix
    S_inv::Matrix
    a0::Union{Float64,Nothing}
    b0::Union{Float64,Nothing}
    a::Union{Float64,Nothing}
    b::Union{Float64,Nothing}
    known_var::Bool

    #=
        prior over w is N(0, α^{-1}I)
        prior over (w, β) is N(0, β^{-1}I)Γ(β, a_0, b_0)
        TODO: maybe more general prior
    =#
    function LM_Bayes(X::Matrix, y::Vector;
        basis::Function=x->vcat(1.,x), α::Float64=0., β=nothing, a0=nothing, b0=nothing)

        N = length(y)

        Phi = calc_ϕ(X, basis)
        n, d = size(Phi)
        @assert n == N

        # m_0 = 0
        # S_0^{-1} = α I(d)
        known_var = β != nothing

        c = known_var ? β : 1.0

        S_inv = α*I(d) + c*Phi'Phi
        make_posdef!(S_inv)
        U = inv(cholesky(S_inv).U)
        S = U*U' # inv(S_inv)
        make_posdef!(S)
        b = Phi'y
        m = c*S*b

        if known_var
            a = nothing
            b = nothing
        else
            @assert a0 != nothing && b0 != nothing
            a = a0 + N/2
            b = b0 + 0.5 * y'y
        end

        return new(X, y, N, m, basis, α, β, m, S, S_inv, a0, b0, a, b, known_var)
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

function student_t_params(lm::LM_Bayes, x)
    λ = (1 + lm.basis(x)'lm.S*lm.basis(x)) * lm.a / lm.b
    μ = lm.basis(x)'lm.m
    ν = 2 * lm.a
    return ν, λ, ν
end

function variance(lm::LM_Bayes, x)
    if lm.known_var
        return 1/lm.β + lm.basis(x)'lm.S*lm.basis(x)
    else
        ν, λ, ν = student_t_params(lm, x)
        return ν/(ν-2) * 1/λ
    end
end

function predict(lm::LM_Bayes, X_pred::Matrix, w=lm.w)
    Φ = calc_ϕ(X_pred, lm.basis)
    return vec(Φ*w)
end

function predict(lm::LM_Bayes, X_pred::Vector, w=lm.w)
    predict(lm, reshape(X_pred,:,1), w)
end

function sample(lm::LM_Bayes, n=1)
    if lm.known_var
        w = rand(MvNormal(lm.m,lm.S), n)
        return w
    end
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
    σs = map(t->√variance(lm, t), ts)
    plot(ts, predict(lm, ts), ribbon=σs,
        label="MAP with 95% confidence", lc=:black)
    scatter!(vec(lm.X), lm.y, label="data"; kw...)
end

function plot_w_prior_2d(lm::LM_Bayes; xlims=nothing, ylims=nothing)
    if length(lm.w) == 2
        D = MvNormal(zeros(2),1/lm.α * I(2))
        if xlims == nothing
            xlims = (- 2.56 / lm.α, 2.56 / lm.α)
        end
        if ylims == nothing
            ylims = (- 2.56 / lm.α, 2.56 / lm.α)
        end

        xs = LinRange(xlims..., 500)
        ys = LinRange(ylims..., 500)
        heatmap(xs, ys, (x,y)->pdf(D, [x,y]))
        scatter!([(0, 0)], markershape=:x, label="", mc=:black)
    end
end

function plot_w_posterior_2d(lm::LM_Bayes; xlims=nothing, ylims=nothing)
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

function plot_β_prior(lm::LM_Bayes)
    G = Gamma(lm.a0, lm.b0)
    plot(x -> pdf(G, x))
end

function plot_β_prior(lm::LM_Bayes; kw...)
    G = Gamma(lm.a0, 1/lm.b0)
    plot(x -> pdf(G, x);kw...)
end


function plot_β_posterior(lm::LM_Bayes; kw...)
    G = Gamma(lm.a, 1/lm.b)
    plot(x -> pdf(G, x);kw...)
end
