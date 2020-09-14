
using LinearAlgebra

#=
 Implementation of LU decomposition.
 A regular matrix A can be decomposed in a lower triangular matrix L and
 an upper triangular matrix U, such that PA = LU,
 where P is permutation matrix.

 The method used is Gaussian Elimination without pivoting,
 this means that P = I, the identity matrix.
=#

# implementation like in the lecture notes
function LU_without_pivot(A::Matrix)
    n = size(A,2)
    # assert square matrix
    @assert size(A,1) == n

    U = copy(A)
    L = Matrix{Float64}(I(n))

    for j in 1:n
        L_j = Matrix{Float64}(I(n))
        for i in j+1:n
            # record changes
            L_j[i,j] = - U[i,j] / U[j,j]
            # substract j-th row scaled by a_ij / a_jj from i-th row
            U[i,:] .-= U[j,:] .* (U[i,j] / U[j,j])
        end
        L = L_j * L # multiply changes together (slow)
    end

    return Matrix{Float64}(I(n)), LowerTriangular(inv(L)), UpperTriangular(U) # (inv -> slow)
end

# faster implementation without matrix multiplication or inversion
function LU_without_pivot_fast(A::Matrix)
    n = size(A,2)
    # assert square matrix
    @assert size(A,1) == n

    U = copy(A)
    L = Matrix{Float64}(I(n))

    for j in 1:n
        for i in j+1:n
            # one can show that the resulting L matrix looks like this
            L[i,j] = U[i,j] / U[j,j]
            # substract j-th row scaled by a_ij / a_jj from i-th row
            U[i,:] .-= U[j,:] .* (U[i,j] / U[j,j])
        end
    end

    return Matrix{Float64}(I(n)), LowerTriangular(L), UpperTriangular(U)
end

#=
 Implementation of LU decomposition.
 A regular matrix A can be decomposed in a lower triangular matrix L and
 an upper triangular matrix U, such that PA = LU,
 where P is permutation matrix.

 The method used is Gaussian Elimination with pivoting,
 this means that at each step j we find the row with largest absolute value at column j.
 We then bring this row to index j and record the changes in the permuatinon matrix P.
=#

# implementation like in the lecture notes
function LU_with_pivot(A::Matrix)
    n = size(A,2)
    # assert square matrix
    @assert size(A,1) == n

    U = copy(A)
    L = Array{Matrix{Float64}}(undef, n)
    P = Array{Matrix{Float64}}(undef, n)

    for j in 1:n
        # find row with largest absolute value
        i = argmax(abs.(U[j:n,j])) + (j-1) # have to shift here

        # check if singular
        if abs(U[i,j]) < eps(Float64)
            @warn "Singular Matrix can not be LU decomposed."
            return nothing, nothing
        end

        P_j = Matrix{Float64}(I(n))
        if i != j
            # pivot
            tmp = U[i,:]
            U[i,:] = U[j,:]
            U[j,:] = tmp
            # record changes in permutation matrix
            tmp = P_j[i,:]
            P_j[i,:] = P_j[j,:]
            P_j[j,:] = tmp
        end

        L_j = Matrix{Float64}(I(n))
        for i in j+1:n
            # record changes
            L_j[i,j] = - U[i,j] / U[j,j]
            # substract j-th row scaled by a_ij / a_jj from i-th row
            U[i,:] .-= U[j,:] .* (U[i,j] / U[j,j])
        end

        L[j] = L_j; P[j] = P_j
    end

    # one can show that
    # K_j = P_{n} P_{n-1} ... P_{j+1} L_{j} P_{j+1}' ... P_{n-1}' P_{n}'
    # such that
    # K_{n} K_{n-1} ... K_1 P_{n} P_{n-1} ... P_1 A = U
    P_tot = Matrix{Float64}(I(n))
    K = Matrix{Float64}(I(n))
    for j in n:-1:1
        K_j  = P_tot * L[j] * P_tot'
        K = K * K_j
        P_tot = P_tot * P[j]
    end # many matrix multiplications -> slow
    L = inv(K) # slow

    return P_tot, LowerTriangular(L), UpperTriangular(U)
end

# faster implementation, do not have to store each P_j and L_j
# and multiple afterwards. Can be computed directly.
function LU_with_pivot_fast(A::Matrix)
    n = size(A,2)
    # assert square matrix
    @assert size(A,1) == n

    U = copy(A)
    L = Matrix{Float64}(I(n))
    P = Matrix(I(n))

    for j in 1:n
        # find row with largest absolute value
        i = argmax(abs.(U[j:n,j])) + (j-1) # have to shift here

        # check if singular
        if abs(U[i,j]) < eps(Float64)
            @warn "Singular Matrix can not be LU decomposed."
            return nothing, nothing
        end

        if i != j
            # pivot
            tmp = U[i,:]
            U[i,:] = U[j,:]
            U[j,:] = tmp

            tmp = L[i,:]    # have to permute L also
            L[i,:] = L[j,:]
            L[j,:] = tmp
            L[j,j] = 1; L[i,i] = 1 # diagonal could be swapped to 0

            # record changes in single permutation matrix
            tmp = P[i,:]
            P[i,:] = P[j,:]
            P[j,:] = tmp
        end

        for i in j+1:n
            # one can show that the resulting L matrix looks like this
            L[i,j] = U[i,j] / U[j,j]
            # substract j-th row scaled by a_ij / a_jj from i-th row
            U[i,:] .-= U[j,:] .* (U[i,j] / U[j,j])
        end
    end

    return P, LowerTriangular(L), UpperTriangular(U)
end

function LU(A::Matrix; pivot=true)
    if pivot
        return LU_with_pivot_fast(A)
    else
        return LU_without_pivot_fast(A)
    end
end
