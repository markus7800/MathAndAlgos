

# computes coeffs of polynomial given by
# ∑ coeffs[i] * x^i * (x - d)
function mult_poly_mono!(coeffs::Vector{Float64}, new_coeffs::Vector{Float64}, d::Float64)
    n = length(coeffs)

    new_coeffs[n+1] = coeffs[n]
    new_coeffs[1] = -coeffs[1]*d

    for i in 2:n
        new_coeffs[i] = coeffs[i-1] - coeffs[i]*d
    end
end

# n is degree of coeffs + 1, [1,2,0,0,0,0] -> 2
function mult_poly_mono_inplace!(coeffs::Vector{Float64}, d::Float64, n::Int)
    coeffs_i_minus_1 = coeffs[1]

    coeffs[n+1] = coeffs[n]
    coeffs[1] = -coeffs[1]*d

    for i in 2:n
         # store for next iteration as we overwrite it
        tmp = coeffs[i]
        coeffs[i] = coeffs_i_minus_1 - coeffs[i]*d
        coeffs_i_minus_1 = tmp
    end
end

function mult_poly_mono(coeffs::Vector{Float64}, d::Float64)::Vector{Float64}
    n = length(coeffs)
    new_coeffs = Vector{Float64}(undef,n+1)
    mult_poly_mono!(coeffs, new_coeffs, d)
    return new_coeffs
end

# computes coeffs of polynomial given by
# ∑ coeffs[i] * x^i / (x - d)
# if d is root of polynom
function div_poly_mono!(coeffs::Vector{Float64}, new_coeffs::Vector{Float64}, d::Float64)
    n = length(coeffs)

    new_coeffs[1] = -coeffs[1] / d
    new_coeffs[n-1] = coeffs[n]

    for i in 2:n-2
        new_coeffs[i] = (new_coeffs[i-1] - coeffs[i]) / d
    end
end

# n is degree of coeffs - 1, [1,2,0,0,0,0] -> 0
function div_poly_mono_inplace!(coeffs::Vector{Float64}, d::Float64, n::Int)
    coeffs[1] = -coeffs[1] / d
    coeffs[n-1] = coeffs[n]
    coeffs[n] = 0
    for i in 2:n-2
         # store for next iteration as we overwrite it
        coeffs[i] = (coeffs[i-1] - coeffs[i]) / d
    end
end

function div_poly_mono(coeffs::Vector{Float64}, d::Float64)::Vector{Float64}
    n = length(coeffs)
    new_coeffs = Vector{Float64}(undef,n-1)
    div_poly_mono!(coeffs, new_coeffs, d)
    return new_coeffs
end


p = mult_poly_mono([1., 2., 3.], 4.)
div_poly_mono(p, 4.)

p = mult_poly_mono([7., 2., 0., 5.], 3.)
div_poly_mono(p, 3.)

begin
    println("Normal:\n")
    p = zeros(1)
    p[1] = 1
    for (i, x) in enumerate([2., -2., 3.])
        p = mult_poly_mono(p, x)
        println(i, ": ", p)
    end
    println()
    for (i, x) in enumerate([2., -2., 3.])
        p = div_poly_mono(p, x)
        println(i, ": ", p)
    end
    println()
end

begin
    println("Inplace:\n")
    g = 4
    p = zeros(g)
    p[1] = 1
    for (i, x) in enumerate([2., -2., 3.])
        mult_poly_mono_inplace!(p, x, i)
        println(i, ": ", p)
    end

    println()
    for (i, x) in enumerate([2., -2., 3.])
        div_poly_mono_inplace!(p, x, g - i + 1)
        println(g - i - 1, ": ", p)
    end
    println()
end

function get_polynom_function(coeffs::Vector{Float64})
    return function poly(x::Float64)::Float64
        powers = [x^p for p in 0:length(coeffs)-1]
        return coeffs'powers
    end
end

function polynom_interpolation_vandermond(xs::Vector{Float64}, ys::Vector{Float64})::Vector{Float64}
    n = length(xs)
    V = zeros(n, n)
    x_power = ones(n)
    for i in 1:n
        V[:,i] = x_power
        x_power .*= xs
    end
    return V \ ys
end

function polynom_interpolation_lagrange_mult(xs::Vector{Float64}, ys::Vector{Float64})::Vector{Float64}
    n = length(xs)
    p = zeros(n)
    for i in 1:n
        lagrange = [1.]
        for j in 1:n
            i == j && continue
            lagrange = mult_poly_mono(lagrange, xs[j]) / (xs[i] - xs[j])
        end
        lagrange *= ys[i]
        p += lagrange
    end
    return p
end

function polynom_interpolation_lagrange_mult_inplace(xs::Vector{Float64}, ys::Vector{Float64})::Vector{Float64}
    n = length(xs)
    p = zeros(n)
    lagrange = zeros(n)
    for i in 1:n
        lagrange .= 0
        lagrange[1] = 1

        c = 1
        g = 1
        for j in 1:n
            i == j && continue
            mult_poly_mono_inplace!(lagrange, xs[j], g)
            lagrange /= (xs[i] - xs[j])
            g += 1
        end
        lagrange *= ys[i]
        p += lagrange
    end
    return p
end

function polynom_interpolation_lagrange_div(xs::Vector{Float64}, ys::Vector{Float64})::Vector{Float64}
    n = length(xs)
    p = zeros(n)
    lagrange = zeros(n)

    full_lagrange = zeros(n+1)
    full_lagrange[1] = 1
    xm = mean(xs)
    c = 1
    for g in 1:n
        mult_poly_mono_inplace!(full_lagrange, xs[g], g)
        full_lagrange /= (xm - xs[g])
        c *= (xm - xs[g])
    end
    println(full_lagrange)
    println(c)

    for i in 1:n
        c = prod(xs[i] .- xs)
        div_poly_mono!(full_lagrange, lagrange, xs[i])
        lagrange *= ys[i] / c
        p += lagrange
    end

    return p
end

xs = [1., 2., 3.]
ys = [1., 2., -1.]

using Random
using StatsBase
Random.seed!(0)
n = 25
xs = Float64.(sample(-100:100, n, replace=false))
ys = Float64.(rand(-10:10, n))

p1 = polynom_interpolation_vandermond(xs, ys)

p2 = polynom_interpolation_lagrange_mult(xs, ys)

p2 = polynom_interpolation_lagrange_mult_inplace(xs, ys)

p2 = polynom_interpolation_lagrange_div(xs, ys)

sum(abs.(p2 .- p1))

pf1 = get_polynom_function(p1)
pf2 = get_polynom_function(p2)

sum(abs.(pf1.(xs) - ys))
sum(abs.(pf2.(xs) - ys))



Random.seed!(0)
n = 100
xs = Float64.(0:1/n:1)
coeffs = Float64.(rand(-10:10, n+1))
pf = get_polynom_function(coeffs)
ys = pf.(xs)



p1 = polynom_interpolation_vandermond(xs, ys)

p2 = polynom_interpolation_lagrange_mult(xs, ys)

p2 = polynom_interpolation_lagrange_mult_inplace(xs, ys)


for (f, t) in zip(p1, coeffs)
    println("$f $t")
end

pf1 = get_polynom_function(p1)
pf2 = get_polynom_function(p2)

sum(abs.(pf1.(xs) - ys))
sum(abs.(pf2.(xs) - ys))
