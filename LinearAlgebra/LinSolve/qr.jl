# 7.11
using LinearAlgebra

#=
 Implementation of QR factorization.
 Every matrix A ∈ F^{n x m} can be factored into a orthogonal/unitary matrix Q
 and an upper triangular matrix R, such that A = QR.
 To achieve this factorization Householder reflections are of essence.
=#

# Real Case
function HouseholderReflection(x::Vector{Float64})
    n = length(x)
    α = -sign(x[1]) * norm(x)
    αe1 = zeros(Float64, n); αe1[1] = α
    u = x - αe1
    if u == 0
        return 1.0 * I(n)
    else
        return 1.0 * I(n) - 2/u'u * (u * u')
    end
end

# Complex case
function HouseholderReflection(x::Vector{ComplexF64})
    n = length(x)
    α = -x[1]/abs(x[1]) * norm(x) # equivalent to -exp(im*angle(x[1]))
    αe1 = zeros(ComplexF64, n); αe1[1] = α
    u = x - αe1
    v = u / norm(u)
    if u == 0
        return 1.0 * I(n)
    else
        return 1.0 * I(n) - (1 + x'v/v'x) * (v * v')
    end
end

function QR(A::Matrix)
    n, m = size(A)

    R = copy(A)
    Q = Matrix{eltype(A)}(I(n))
    j = 0

    while j < min(n-1, m)
        j += 1
        x = R[j:n,j]
        Q_j = Matrix{eltype(A)}(I(n))
        P = HouseholderReflection(x)

        # place Householder reflection at lower right position
        Q_j[j:n,j:n] = P

        R = Q_j * R
        Q = Q * Q_j'
    end

    # Set elements below diagonal to 0
    for j in 1:m, i in j+1:n
        @assert abs(R[i,j]) < 1e-9
        R[i,j] = 0.0
    end

    return Q, UpperTriangular(R)
end
