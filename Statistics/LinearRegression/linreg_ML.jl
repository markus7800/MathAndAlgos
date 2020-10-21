using LinearAlgebra
using Plots
using Printf
using StringLiterals

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


function summary_stats(lm::LM_ML)
    Phi = calc_ϕ(lm.X, lm.basis)
    n, p = size(Phi)
    A = lm.λ * I(p) + Phi'Phi
    make_posdef!(A)
    A_inv = inv(cholesky(A))

    d = sqrt.(Diagonal(A_inv))
    ŷ = Phi*lm.w
    σ = sqrt(1/(n-p) * sum((lm.y .- ŷ).^2))


    siglevels = [1, 0.1, 0.05, 0.01, 0.001]
    symbols = ["", ".", "*", "**", "***"]

    T = TDist(n-p)
    println()
    println("Coefficients:")
    pr"""    Estimate  \%11s("Std")   \%12s("t value")   \%16s("P(>|t|)")\n"""
    for (i,w) in enumerate(lm.w)
        sd = σ * d[i,i]
        z = w/sd
        # pval = 1 - cdf(T, abs(z)) + cdf(T,-abs(z))
        pval = 2 * (1 - cdf(T, abs(z))) # symmetric
        i = sum(pval .< siglevels)
        s = symbols[i]
        pr"\%12.4f(w) \%12.4f(sd)   \%12.2f(z)   \%s(pval) \t \%s(s)\n"
    end
    println()

    tss = sum((lm.y .- mean(lm.y)).^2)
    rss = sum((lm.y .- ŷ).^2)
    reg_ss = sum((ŷ .- mean(lm.y)).^2)

    df = p+1
    println(@sprintf "Residual standard error: %.5f on %d degress of freedom" sqrt(rss/df) df)


    R_squared = 1 - rss/tss
    R_squared_adj = 1 - (rss/(n-p)) / (tss/(n-1))
    println(@sprintf "Multiple R-squared: %.4f, \t Adjusted R-squared: %.4f" R_squared R_squared_adj)

    f = (reg_ss/(p-1)) / (rss/(n-p))
    F = FDist(p-1, n-p)
    pval = 1 - cdf(F, abs(f)) + cdf(F,-abs(f))
    println(@sprintf "F-statistic: %f on %d and %d DF, p-value: %s" f (p-1) (n-p) pval)
    println()
end

# X = [1., 2., 3., 4. , 5., 6.]
# y = 3 * X .- 1
#
# ϵ = [0, 0.1, -0.1, 0., 0.1, 0.1]
# y += ϵ
#
# lm = LM_ML(X, y)
#
# plot_lm(lm)
#
# summary_stats(lm)
#
