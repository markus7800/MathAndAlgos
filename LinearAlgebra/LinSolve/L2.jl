function WolfeLS(f::Function, ∇f::Function; x::Array{Float64,1},
    p::Array{Float64,1}, c1=10^-4, c2 = 0.9, n_max=10^3)

    α = 0; β = Inf; t = 1

    for _ in 1:n_max
        if f(x + t * p) > f(x) + c1 * ∇f(x)'p
            β = t
        elseif ∇f(x + t * p)'p < c2 * ∇f(x)'p
            α = t
        else
            return t
        end
        if β < Inf
            t = (α + β) / 2
        else
            t = 2*α
        end
    end

    return t
end

# norm(x::Array{Float64}) = √sum(x.^2)

function estimate_L(f::Function, ∇f::Function,
    lower::Array{Float64,1}, upper::Array{Float64,1}, N = 100)

    xs = rand_position(lower, upper, N)
    L = 0
    for i in 1:N, j in i:N
        x = xs[:,i]; y = xs[:,j]
        d = norm(x .- y)
        if d > 0
            L = max(L, norm(∇f(x) .- ∇f(y)) / d)
        end
    end
    return L
end

function AcceleratedGD(f::Function, ∇f::Function; x0::Array{Float64, 1},
    stepsize=:fixed, L=10, tol = 1e-12, n_max=10^5)

    λ = 1.0
    x = x0
    xs = [x, x] # to be able to calculate differences
    x_new = fill(Inf, length(x))
    n = 2
    while maximum(abs.(x - x_new)) > tol && n ≤ n_max
        λ_old = λ
        λ = (1 + √(4λ + 1)) / 2
        γ = (λ_old - 1) / λ


        d = γ * (xs[n] - xs[n-1])
        y = x + d

        if stepsize == :fixed
            α = 1 / L
        elseif stepsize == :Wolfe
            α = WolfeLS(f, ∇f, x=y, p=-∇f(y))
        end
        g = -α * ∇f(y)

        x = xs[n]
        x_new = y + g

        push!(xs, x_new)
        n += 1
    end

    return x_new, xs
end


function solve_L2(A::Matrix, b::Vector)
    # min_x ||Ax - b)||_2^2
    x = zeros(size(A,2))
    f(x) = 0.5 * sum((A*x - b).^2)
    AA = adjoint(A)*A
    Ab = adjoint(A)*b
    ∇f(x) = AA*x - Ab

    x, xs = AcceleratedGD(f, ∇f, x0=x, stepsize=:Wolfe)
    n = length(xs)
    println("Number of steps: $n")
    return x
end
