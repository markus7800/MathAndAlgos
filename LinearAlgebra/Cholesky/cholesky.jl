

function cholesky_decomp(A::Matrix)
    n,m = size(A)
    L = zeros(n,m)
    for i in 1:n
        for j in 1:i-1
            L[i,j] = 1/L[j,j] * (A[i,j] - (L[i,1:j-1])'*(L[j,1:j-1]))
        end
        L[i,i] = sqrt(A[i,i] - sum(L[i,1:i-1].^2))
    end
    U = UpperTriangular(L')
    return LowerTriangular(L), U
end
