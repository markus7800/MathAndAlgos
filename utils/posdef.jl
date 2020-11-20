using LinearAlgebra

function make_posdef!(K::AbstractMatrix; chances=10)
    if isposdef(K)
        return
    end
    for _ in 1:chances
        ϵ = 1e-6 * tr(K) / size(K,1)
        for i in 1:size(K,1)
            K[i,i] += ϵ
        end
        if isposdef(K)
            return
        end
    end
    throw(ArgumentError("K is not positive definite."))
end

function make_symmetric!(K::AbstractMatrix, tol=1e-9)
    D = LowerTriangular(K - K') / 2
    if maximum(abs.(D)) > tol
        error("K is not symmetric within tolerance.")
    end
    K .-= D
    K .+= D'
end
