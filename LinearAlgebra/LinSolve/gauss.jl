using LinearAlgebra

function gauss(L::LowerTriangular, b::Vector)
    n, = size(L)
    x = zeros(n)
    for i in 1:n
        x[i] = b[i]
        for j in 1:i-1
            x[i] -= L[i,j] * x[j]
        end
        x[i] /= L[i,i]
    end
    return x
end

function gauss(U::UpperTriangular, b::Vector)
    n, = size(U)
    x = zeros(n)
    for i in n:-1:1
        x[i] = b[i]
        for j in i+1:1:n
            x[i] -= U[i,j] * x[j]
        end
        x[i] /= U[i,i]
    end
    return x
end

function gauss(A::Matrix, b::Vector)
    m, n = size(A)
    @assert m == n

    U = zeros(n,n+1)
    U[1:n,1:n] .= A
    U[:,n+1] .= b

    for j in 1:n # cols
        v, pivot = findmax(abs.(U[j:n,j]))
        pivot += j - 1

        v == 0 && continue

        if j != pivot
            tmp = U[j,:]
            U[j,:] = U[pivot,:] / U[pivot,j]
            U[pivot,:] = tmp
        else
            U[j,:] /= U[j,j]
        end

        for i in j+1:n
            v = U[i, j]
            if v != 0
                U[i,:] -= U[j,:] * v
            end
        end
    end
    return gauss(UpperTriangular(U[1:n, 1:n]), U[:,n+1])
end
