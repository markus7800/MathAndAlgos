
using LinearAlgebra

include("gauss.jl")
include("qr.jl")
include("lu.jl")
include("L2.jl")



function solve(A::Matrix, b::Vector; method=:gauss, kw...)
    if method == :gauss
        return gauss(A, b)
    elseif method == :lu
        P, L, U = LU(A; kw...)
        # Ax = P'LUx = b
        # LUx = Pb
        # Ly = Pb
        # Ux = y
        y = gauss(L, P*b)
        x = gauss(U, y)
        return x
    elseif method == :qr
        Q, R = QR(A)
        # Ax = QRx = b
        # Rx = Q'b
        x = gauss(R, Q'b)
        return x
    elseif method == :L2
        x = solve_L2(A, b)
        return x
    end
end
