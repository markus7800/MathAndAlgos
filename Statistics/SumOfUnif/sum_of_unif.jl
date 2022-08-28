

using Combinatorics

#https://www.researchgate.net/publication/23962029_On_the_distribution_of_the_sum_of_independent_uniform_random_variables
# [0, a_i]
function pdf(s::T, a::Vector{T}) where T <: Real
    n = length(a)


    if s < 0 || s > sum(a)
        return 0.0
    end

    if n == 1
        return 1/a[1]
    end

    res = s^(n-1)

    for k in 1:n
        tmp = zero(typeof(s))
        Jk = combinations(1:n,k)
        for j in Jk
            tmp += max(s - sum(a[j[l]] for l in 1:k), zero(typeof(s)))^(n-1)
        end
        res += (-1)^k * tmp
    end

    A = prod(a)
    res = res / A / factorial(n-1)

    return res
end

# [-a_i, a_i]
function pdf2(s::T, a::Vector{T}) where T <: Real
    a2 = a .* 2
    return pdf(s + sum(a), a2)
end

# [a_i, b_i]
function pdf3(s::T, a::Vector{T}, b::Vector{T}) where T <: Real
    a2 = b .- a
    return pdf(s - sum(a), a2)
end

pdf(1//1, [1//1,1//1])

using Plots

plot(s -> pdf(s, [1.,2.,0.5]), xlims=(-1,5), legend=false)
plot(s -> pdf2(s, [1.,2.,0.5]), xlims=(-5,5), legend=false)
plot(s -> pdf3(s, [-1.,-2.,-0.5], [1.,2.,0.5]), xlims=(-5,5), legend=false)


begin
    p = plot(xlims=(-0.1,0.1), legend=false)
    for i in 1:2:15
        plot!(s -> pdf2(s, [1/k for k in 1:2:i]))
    end
    display(p)
end



begin
    for i in 1:2:15
        println(i, ": ", pdf2(BigFloat(0., precision=64), [BigFloat(1/k, precision=64) for k in 1:2:i]))
    end
end


a = -2.
b = 3.
c = -1.
d = 1.

plot(s -> pdf3(s, [a,c], [b,d]), xlims=(-5,5), legend=false)
vline!([a+d, b+c])
