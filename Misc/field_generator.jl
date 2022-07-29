
struct Polynomial
    coeffs::Vector{Int}
    P::Int

    function Polynomial(coeffs::Vector{Int}, P::Int)::Polynomial
        return new(coeffs, P)
    end

    function Polynomial(s::String, P::Int)::Polynomial
        xs = split(s, " + ")
        xs = [split(x, "x^") for x in xs]
        max_power = maximum(parse(Int, x[2]) for x in xs)

        coeffs = zeros(Int, max_power+1)
        for x in xs
            power = parse(Int, x[2])
            a = parse(Int, x[1])
            coeffs[power+1]=a
        end

        return new(coeffs, P)
    end
end

function degree(p::Polynomial)
    return length(p.coeffs)-1
end

function Base.show(io::IO, p::Polynomial)
    print(io, string(p))
end

function Base.string(p::Polynomial)
    if degree(p) == -1
        return "0"
    end

    s = ""

    add = false
    for (i, a) in enumerate(p.coeffs)
        i = i-1
        if a != 0
            if add
                s *= " + "
            end
            if i == 0
                s *= "$(a)"
            elseif i == 1 && a == 1
                s *= "x"
            elseif i == 1
                s *= "$(a)x"
            elseif a == 1
                s *= "x^$(i)"
            else
                s *= "$(a)x^$(i)"
            end
            add = true
        end
    end
    return s
end

function Base.getindex(p::Polynomial, i::Int)
    return i <= degree(p) ? p.coeffs[i+1] : 0
end

function Base.setindex!(p::Polynomial, a::Int, i::Int)
    p.coeffs[i+1] = a
end

function Base.isequal(p::Polynomial, q::Polynomial)
    P = p.P
    if q.P != P
        return false
    end

    d = degree(p)
    if degree(q) != d
        return false
    end

    for i in 0:d
        if (p[i] - q[i]) % P != 0
            return false
        end
    end

    return true
end

import Base.==
function ==(p::Polynomial, q::Polynomial)
    return isequal(p, q)
end

function make_canonical!(p::Polynomial)
    for (i, c) in enumerate(p.coeffs)
        a = (c % p.P)
        if a < 0
            a += p.P
        end
        p.coeffs[i] = a
    end
end

function Base.hash(p::Polynomial)
    h = UInt64(0)
    prune!(p)
    make_canonical!(p)
    for c in p.coeffs
        h = hash(c, h)
    end
    return h
end

function prune!(p::Polynomial)
    d = length(p.coeffs)
    for i in d:-1:1
        if p.coeffs[i] != 0
            resize!(p.coeffs, i)
            return p
        end
    end
    resize!(p.coeffs, 0)
    return p
end

function scale!(p::Polynomial, s::Int)::Polynomial
    d = 0
    for (i, a) in enumerate(p.coeffs)
        p.coeffs[i] = (s * a) % p.P
        if p.coeffs[i] != 0
            d = i
        end
    end
    resize!(p.coeffs, d)
    return p
end

function scale(p::Polynomial, s::Int)::Polynomial
    return scale!(deepcopy(p), s)
end

Base.:*(p::Polynomial, s::Int) = scale(p, s)
Base.:*(s::Int, p::Polynomial) = scale(p, s)


function multiply(p::Polynomial, q::Polynomial)::Polynomial
    @assert p.P == q.P
    if degree(p) == -1 || degree(q) == -1
        return Polynomial(Int[], p.P)
    end

    n = degree(p) + degree(q)
    pq = Polynomial(zeros(Int, n+1), p.P)

    d = 0
    for i in 0:n
        c = 0
        for k in 0:i
            c += p[k]*q[i-k]
            #@info ("i $i, k $k, r $(p[k]), s $(q[i-k]) c $c")
        end
        pq[i] = c % pq.P

        if pq[i] != 0
            d = i+1
        end
    end

    resize!(pq.coeffs, d)

    return pq
end

Base.:*(p::Polynomial, q::Polynomial) = multiply(p, q)

function add(p::Polynomial, q::Polynomial)::Polynomial
    @assert p.P == q.P
    n = max(degree(p), degree(q))
    if n == -1
        return Polynomial(Int[], p.P)
    end
    coeffs = zeros(Int, n+1)
    p_q = Polynomial(coeffs, p.P)
    d = 0
    for i in 0:n
        p_q[i] = (p[i] + q[i]) % p.P
        if p_q[i] != 0
            d = i+1
        end
    end
    resize!(p_q.coeffs, d)

    return p_q
end

Base.:+(p::Polynomial, q::Polynomial) = add(p, q)

# multiplicative invers
function Base.inv(x::Int, mod::Int)::Int
    for y in 1:(mod-1)
        if (x*y) % mod == 1
            return y
        end
    end
    error("Invalid input!")
end

function divisors(x::Int)
    d = Int[]
    for y in 1:(x-1)
        if x % y == 0
            push!(d, y)
        end
    end
    return d
end

function Base.rem(p::Polynomial, q::Polynomial)::Polynomial
    @assert p.P == q.P

    f = deepcopy(p)

    n = degree(f)
    m = degree(q)

    b = inv(q[m], q.P)

    while n >= m
        a = f[n]
        new_n = n
        uneq0 = false

        for i in n:-1:(n-m)
            # println("in: ", f[i], ", " ,-a*b*q[i-(n-m)])
            f[i] = (f[i] - a * b * q[i-(n-m)]) % q.P
            # println("out: ", f[i])

            if f[i] == 0 && !uneq0
                new_n -= 1
            else
                uneq0 = true
            end
            # println(new_n, ", ", uneq0)
        end
        # println()
        n = new_n
    end

    # resize!(f.coeffs, n+1)

    return prune!(f)
end

is_divisible_by(p::Polynomial, q::Polynomial) = degree(p % q) == -1



p = Polynomial([0,0,1,0,0], 3)
p = Polynomial([0], 3)
p.coeffs
prune!(p)
p.coeffs

p = Polynomial("1x^1", 3)
q = Polynomial("-2x^1", 3)
isequal(p, q)
p == q

d = Dict()
d[p] = 1
d[q]


p = Polynomial([1,0,2], 3)
p = Polynomial("1x^0 + 0x^1 + 2x^2", 3)
p[0]

[1,2,3] .% 3

scale(p, 2)
2 * p
scale(p, 3)
p * 3

p[1]=2
p

pq = multiply(p, p)
p * p

p + p

pq.coeffs

(p*p) % p

is_divisible_by(p*p, p)

Polynomial("1x^0 + 2x^4 + 1x^5", 3) % Polynomial("1x^0 + 2x^1 + 2x^2", 3)


mutable struct Field
    P::Int
    N::Int

    divisors::Vector{Int}
    normed_polynoms_of_degree::Dict{Int, Vector{Polynomial}}

    primitive_element::Polynomial
    elements::Vector{Polynomial} # normed polynomials of degree N

    symbols::Dict{Polynomial, Int}
    addition_table::Array{Int}
    multiplication_table::Array{Int}

    function Field(P::Int, N::Int)
        return new(
            P, N,
            Vector{Int}(),
            Dict{Int, Vector{Polynomial}}(),

            Polynomial(Int[], P),
            Vector{Polynomial}(),

            Dict{Polynomial, Int}(),
            zeros(Int, P^N, P^N),
            zeros(Int, P^N, P^N)
        )
    end
end

# polynomials of order < N
function find_elements(field::Field)
    hs = zeros(Int, field.N)
	h = Polynomial(hs, field.P)
    a = Vector{Polynomial}()

    # only modify coeefs up to N-1 (not normed anymore)
    permute_coeffs!(field, field.N-1, h, a)

    return a
end

function find_normed_polynoms_of_degree(field::Field, d::Int)

	hs = zeros(Int, d+1)
	hs[d+1] = 1
	h = Polynomial(hs, field.P)
    a = Vector{Polynomial}()

    permute_coeffs!(field, d-1, h, a)

    return a
end

# mutate f at d
function permute_coeffs!(field::Field, d::Int, f::Polynomial, a::Vector{Polynomial})
    if d == 0
        for k in 0:(field.P-1)
            f[d] = k
            push!(a, prune!(deepcopy(f)))
        end
    else
        for k in 0:(field.P-1)
            f[d] = k
            permute_coeffs!(field, d-1, f, a)
        end
    end
end

# no divisors in polynomial ring Z_P[N] ?
function is_irreducible(field::Field, f::Polynomial)
    for d in field.divisors # of N
        for g in field.normed_polynoms_of_degree[d] # if divisible then by these
            if is_divisible_by(f, g)
                return false
            end
        end
    end
    return true
end

# polynomial of degree N
function find_irreducible_polynomial(field::Field)
	q = Polynomial("-1x^1 + 1x^$(field.P^field.N)", field.P)

    coeffs = zeros(Int, field.N+1)
    coeffs[field.N+1] = 1
    f = Polynomial(coeffs, field.P) # x^N

    find_irreducible_polynomial!(field, field.N-1, q, f)
    return f
end

# primitive element f
# mutate f at index i with values 0...(P-1)
# has to divide x^(P^N) - x (q)
function find_irreducible_polynomial!(field::Field, i::Int, q::Polynomial, f::Polynomial)
    if i == 0
        for k in 0:(field.P-1)
            f[i] = k
            if is_divisible_by(q, f) # faster and necessary check
                if is_irreducible(field, f)
                    return true
                end
            end
        end
    else
        for k in 0:(field.P-1)
            f[i] = k
            if find_irreducible_polynomial!(field, i-1, q, f)
                return true
            end
        end
    end
    return false
end

function calculate_addition_table(field::Field)
    for (i, x) in enumerate(field.elements)
        for (j, y) in enumerate(field.elements)
            i > j && continue
            r = (x + y) % field.primitive_element
            k = field.symbols[r]
            field.addition_table[i, j] = k
            field.addition_table[j, i] = k
        end
    end
end

function calculate_multiplication_table(field::Field)
    for (i, x) in enumerate(field.elements)
        for (j, y) in enumerate(field.elements)
            i > j && continue
            r = (x * y) % field.primitive_element
            k = field.symbols[r]
            field.multiplication_table[i, j] = k
            field.multiplication_table[j, i] = k
        end
    end
end


function find_field(field::Field)

    field.divisors = divisors(field.N)
    field.elements = find_elements(field)

    for d in field.divisors
        field.normed_polynoms_of_degree[d] = find_normed_polynoms_of_degree(field, d)
    end

    for (i, elem) in enumerate(field.elements)
        field.symbols[elem] = i
    end

    field.primitive_element = find_irreducible_polynomial(field)

    #calculate_addition_table(field)
    #calculate_multiplication_table(field)
end

function Base.show(io::IO, field::Field)
    println(io, "GF($(field.P)^$(field.N))")
    println(io, "P: $(field.P), N: $(field.N)")
    print(io, "primitive element: $(field.primitive_element)")
end

field = Field(3,2)
find_elements(field)

find_irreducible_polynomial(field)

q = Polynomial("-1x^1 + 1x^$(field.P^field.N)", field.P)
p = Polynomial("1x^0 + 1x^2", field.P)
is_irreducible(field, p)

r = q % p

r.coeffs


function dot(ps::Vector{Polynomial}, qs::Vector{Polynomial})::Array{Int,3}
    P = ps[1].P

    @assert all(p.P == P for p in ps) && all(q.P == P for q in qs)

    max_deg_p = maximum(degree(p) for p in ps)
    max_deg_q = maximum(degree(q) for q in qs)

    if max_deg_p == -1 || max_deg_q == -1
        error("TODO")
    end

    D = max_deg_p + max_deg_q

    p_mat = zeros(Int, length(ps), D+1)
    q_mat = zeros(Int, length(qs), D+1)
    pq_mat = zeros(Int, length(ps), length(qs), D+1)

    for (i, p) in enumerate(ps), d in 0:degree(p)
        p_mat[i, d+1] = p[d]
    end

    for (i, q) in enumerate(qs), d in 0:degree(q)
        q_mat[i, d+1] = q[d]
    end

    for d in 0:D
        for i in 1:length(ps), j in 1:length(qs)
            c = 0
            for k in 0:d
                c += p_mat[i, k+1] * q_mat[j, d-k+1]
            end
            pq_mat[i, j, d+1] = c % P
        end
    end

    # compute remainder
    f = field.primitive_element
    m = degree(f)

    for i in 1:length(ps), j in 1:length(qs)
        n = 0
        for d in D:-1:0
            if pq_mat[i, j, d+1] != 0
                n = d
                break
            end
        end
        if n == 0
            continue
        end

        b = inv(f[m], f.P)

        while n >= m
            a = pq_mat[i,j,n+1]
            new_n = n
            uneq0 = false

            for k in n:-1:(n-m)
                # println("in: ", f[i], ", " ,-a*b*q[i-(n-m)])
                pq_mat[i,j,k+1] = (pq_mat[i,j,k+1] - a * b * f[k-(n-m)]) % q.P
                # println("out: ", f[i])

                if pq_mat[i,j,k+1] == 0 && !uneq0
                    new_n -= 1
                else
                    uneq0 = true
                end
                # println(new_n, ", ", uneq0)
            end
            # println()
            n = new_n
        end
    end
    pq_mat[pq_mat .< 0] .+= P

    return pq_mat
end



function get_symbols_from_coeffs(field::Field, coeffs::Array{Int,3})
    hash_to_polynomial = Dict(hash(p) => p for p in field.elements)
    #println(hash_to_polynomial)
    n, m, d = size(coeffs)
    symbols = zeros(Int, n, m)
    for i in 1:n, j in 1:m
        D = 0
        for k in d:-1:1
            if coeffs[i,j,k] != 0
                D = k
                break
            end
        end
        h = UInt64(0)
        for k in 1:D
            h = hash(coeffs[i,j,k], h)
        end
        #println("entry ", i, ", ", j, ",", coeffs[i,j,:])
        p = hash_to_polynomial[h]
        symbols[i, j] = field.symbols[p]
    end
    return symbols
end

using BenchmarkTools

field = Field(5,4)

find_field(field)

@btime begin
    calculate_multiplication_table(field)
end

using Plots

heatmap(field.multiplication_table, aspect_ratio=:equal, axis=nothing, border=nothing)

@btime begin
    coeffs = dot(field.elements, field.elements)
    mt = get_symbols_from_coeffs(field, coeffs)
end


coeffs = dot(field.elements, field.elements)
mt = get_symbols_from_coeffs(field, coeffs)
calculate_multiplication_table(field)

all(field.multiplication_table .== mt)

field.elements[3] * field.elements[3]

d = Dict()
d[field.elements[3]] = 0

d[hash(field.elements[3])]

field.sy
