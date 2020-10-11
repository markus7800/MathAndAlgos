using Optim
include("../../utils/posdef.jl")
using CVXOPT

function svm_predict(β0, β, X)
    y = sign.(β0 .+ X'β)
    return y
end

function svm_predict(α, X_train, y_train, k::Function, X; fast=true)
    if fast
        # only use significant coeffs
        is = α .> 1e-3
        α = α[is]
        X_train = X_train[:,is]
        y_train = y_train[is]
    end

    n = size(X_train,2)

    βX = [sum(α[i]*y_train[i]*k(X_train[:,i], X_train[:,j]) for i in 1:n) for j in 1:n]
    β0 = -(maximum(βX[y_train .== -1]) + minimum(βX[y_train .== 1]))/2

    Z = [sum(α[i]*y_train[i]*k(X_train[:,i], X[:,j]) for i in 1:n) + β0 for j in 1:size(X,2)]
    y = sign.(Z)
    return y
end

function naive_svm(X, y)
    d = size(X,1)
    Random.seed!(1)
    B0 = rand(d+1)

    function F(B)
        β0 = B[1]
        β = B[2:end]

        # predict
        H = β0 .+ X'β
        y_pred = (H .> 0) .* 2 .- 1 #  ∈ {-1,1}

        S = y .* H
        return -sum(S[y .!= y_pred])
    end

    function ∇F!(∇, B)
        β0 = B[1]
        β = B[2:end]

        # predict
        H = β0 .+ X'β
        y_pred = (H .> 0) .* 2 .- 1 #  ∈ {-1,1}

        false_is = y .!= y_pred

        ∇[1] = -sum(y[false_is])

        yX = reshape(y,1,:) .* X
        ∇[2:end] = -sum(yX[:,false_is], dims=2)
    end

    res = optimize(F, ∇F!, B0)

    if res.minimum ≈ 0
        println("Succesfully separated points.")
    end
    β0 = res.minimizer[1]
    β = res.minimizer[2:end]
    return β0, β
end

dot(x,y) = x'y


function svm_coordinate_descent(X, y, N, k::Function=dot)
    n = size(X,2)

    coeffs = [y[i] * y[j] * k(X[:,i], X[:,j]) for i in 1:n, j in 1:n]

    function F(α)
        s = sum(α)
        for i in 1:n, j in 1:n
            s -= 0.5 * α[i] * α[j] * coeffs[i,j]
        end
        return s
    end

    function ∇F(α, k)
        return 1 - sum(α[j] * coeffs[k,j] for j in 1:n)
    end

    η = 0.001

    Random.seed!(1)
    α = rand(n)

    F_max = F(α)
    α_max = α

    @progress for i in 1:N
        g_max = 0
        for k in 1:n
            ∇ = ∇F(α, k)
            η_ = η; α_k = α[k]
            for i in 1:10
                α[k] = α_k + η_ * ∇
                if F(α) > F_max
                    break
                end
                η_ /= 2
            end
            α -= y * y'α/y'y # project onto y'α = 0
            # clamp!(α, 0, Inf) # clamp to α ≥ 0

            F_current = F(α)
            if F_current > F_max
                F_max = F_current; α_max = α
                # println("New max: $F_max")
            end

            g_max = max(abs(∇), g_max)
        end
        println("g_max: $g_max")
    end

    β = reshape(sum((y.*α_max)' .* X, dims=2),:)

    βX = vec(β'X)
    β0 = -(maximum(βX[y .== -1]) + minimum(βX[y .== 1]))/2

    return β0, β, α
end


function svm(X, y, k::Function=dot)
    n = size(X,2)

    #= Solve
        max_α ∑_i α_i - 0.5  ∑_i ∑_j α_i α_J y_i y_j k(x_i, x_j)
        s.t. α ≥ 0 and y'a = 0

      Bring in form for cvxopt
        min_x 0.5 x'Px + q'x
        s.t Gh ≤ 0 and Ax = b
    =#

    Z = [k(X[:,i], X[:,j]) for i in 1:n, j in 1:n]
    P = y*y' .* Z
    make_posdef!(P)

    q = fill(-1., size(P,1))

    G = -Matrix{Float64}(I(n))
    h = zeros(n)

    A = Matrix{Float64}(y')
    b = [0.]

    res = CVXOPT.qp(P, q, G, h, A=A, b=b)

    α = vec(res["x"])

    β = vec(sum((y.*α)' .* X, dims=2))

    # https://stats.stackexchange.com/questions/91269/deriving-the-optimal-value-for-the-intercept-term-in-svm
    βX = vec(β'X)
    β0 = -(maximum(βX[y .== -1]) + minimum(βX[y .== 1]))/2

    return β0, β, α
end
