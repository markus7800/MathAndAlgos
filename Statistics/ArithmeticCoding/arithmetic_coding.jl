using Plots

function encoder_infinite_prec(alphabet::Vector{T}, p::Vector{Float64}, xs::Vector{T})::BitVector where T
    d = cumsum(p)
    c = d - p
    charmap = Dict(c => i for (i,c) in enumerate(alphabet))

    a = 0
    b = 1

    for x in xs
        i = charmap[x]
        w = b - a
        b = a + w * d[i]
        a = a + w * c[i]
    end
    println("a: $a, b: $b")

    out = BitVector()
    s = 0
    while b < 0.5 || a > 0.5
        if b < 0.5
            push!(out, 0)
            a = 2*a
            b = 2*b
        elseif a > 0.5
            push!(out, 1)
            a = 2*(a-0.5)
            b = 2*(b-0.5)
        end
    end
    while a > 0.25 && b < 0.75
        s += 1
        a = 2*(a-0.25)
        b = 2*(b-0.25)
    end
    s += 1
    if a <= 0.25
        push!(out, 0)
        push!(out, fill(1, s)...)
    else
        push!(out, 1)
        push!(out, fill(0, s)...)
    end
    return out
end

function to_binary(f::Float64)::BitVector
    out = BitVector()
    if f == 0.
        push!(out, 0)
        return out
    end
    for i in 1:32
        if f == 0.
            break
        end
        if f >= 1/2^i
            f -= 1/2^i
            push!(out, 1)
        else
            push!(out, 0)
        end
    end
    return out
end

function draw_scale!(p, N, a, b, l, u)
    y = -N*3
    tick_length = 0.5
    plot!(p, [0,1], [y,y], color=:black);
    plot!(p, [0,0], [y-tick_length,y+tick_length], color=:black);
    plot!(p, [0.25,0.25], [y-tick_length/2,y+tick_length/2], color=:black);
    plot!(p, [0.5,0.5], [y-tick_length,y+tick_length], color=:black);
    plot!(p, [0.75,0.75], [y-tick_length/2,y+tick_length/2], color=:black);
    plot!(p, [1,1], [y-tick_length,y+tick_length], color=:black);

    q = (u-l)/4
    q0 = bitstring(to_binary(l))
    q1 = bitstring(to_binary(l+q))
    q2 = bitstring(to_binary(l+2*q))
    q3 = bitstring(to_binary(l+3*q))

    annotate!(p, [
        (0.,y+tick_length*2,text("$l", 5, :right, :hcenter, :black, rotation=0)),
        (0.125,y+tick_length/2,text(q0, 5, :hcenter, :bottom, :black)),
        (0.375,y+tick_length/2,text(q1, 5, :hcenter, :bottom, :black)),
        (0.625,y+tick_length/2,text(q2, 5, :hcenter, :bottom, :black)),
        (0.875,y+tick_length/2,text(q3, 5, :hcenter, :bottom, :black)),
        (1.,y+tick_length*2,text("$u", 5, :right, :hcenter, :black, rotation=0))
    ])
    scatter!(p, [(a,y), (b,y)], color=2);
end

function encoder_infinite_prec_rescale(alphabet::Vector{T}, p::Vector{Float64}, xs::Vector{T})::BitVector where T
    d = cumsum(p)
    c = d - p
    charmap = Dict(c => i for (i,c) in enumerate(alphabet))

    l = 0.
    u = 1.
    a = 0
    b = 1


    N = 0
    plt = plot(axis=nothing, legend=false, border=:none, xlims=(-0.1,1.1))
    draw_scale!(plt, N, a, b, l, u);N+=1;

    out = BitVector()
    s = 0
    for x in xs
        i = charmap[x]
        old_a = a
        old_b = b
        w = b - a
        b = a + w * d[i]
        a = a + w * c[i]
        println(x, ": $l [$a, $b) $u")

        # draw_scale!(plt, N, old_a, old_b, l, u);
        annotate!((-0.025,-(N-1+0.5)*3, text("$x", 5, :right, :vcenter, :black)))
        # N+=1;
        draw_scale!(plt, N, a, b, l, u);
        plot!([old_a, a], [-3*(N-1), -3*N], color=2, ls=:dash)
        plot!([old_b, b], [-3*(N-1), -3*N], color=2, ls=:dash)
        N+=1;

        println("rescale start")
        while b < 0.5 || a > 0.5
            if b < 0.5
                push!(out, 0)
                push!(out, fill(1, s)...)
                a = 2*a; b = 2*b

                scale = u-l
                u = l+scale*0.5; l = l+scale*0.;
                # l = 2*l; u = 2*u

                println("\tblow up left half $l $a $b $u emit ", "0"*"1"^s)
                draw_scale!(plt, N, a, b, l, u);
                plot!([0., 0.], [-3*(N-1), -3*N], color=:gray, ls=:dash)
                plot!([0.5, 1.], [-3*(N-1), -3*N], color=:gray, ls=:dash)
                annotate!((1.025,-(N-1+0.5)*3, text("0"*"1"^s, 5, :left, :vcenter, :black)))
                N+=1;

                s = 0
            elseif a > 0.5
                push!(out, 1)
                push!(out, fill(0, s)...)
                a = 2*(a-0.5); b = 2*(b-0.5)

                scale = u-l
                u = l + scale*1.; l = l + scale*0.5;
                # l = 2*(l-0.5); u = 2*(u-0.5)

                println("\tblow up right half $l $a $b $u emit ", "1"*"0"^s)
                draw_scale!(plt, N, a, b, l, u);
                plot!([0.5, 0.], [-3*(N-1), -3*N], color=:gray, ls=:dash)
                plot!([1., 1.], [-3*(N-1), -3*N], color=:gray, ls=:dash)
                annotate!((1.025,-(N-1+0.5)*3, text("1"*"0"^s, 5, :left, :vcenter, :black)))
                N+=1;

                s = 0
            end
        end
        # a <= 0.5 and 0.5 <= b
        while a > 0.25 && b < 3*0.25
            s += 1
            a = 2*(a-0.25); b = 2*(b-0.25)

            scale = u-l
            u = l + scale*0.75; l = l + scale*0.25;
            # l = 2*(l-0.25); u = 2*(u-0.25)
            println("\tblow up middle half $l $a $b $u")
            draw_scale!(plt, N, a, b, l, u);
            plot!([0.25, 0.], [-3*(N-1), -3*N], color=:gray, ls=:dash)
            plot!([0.75, 1.], [-3*(N-1), -3*N], color=:gray, ls=:dash)
            N+=1;
        end
        # a <= 0.25 or 0.75 <= b
        # [a, b) contains [0.25, 0.5) or [0.5, 0.75)
        println("rescale finished: $l [$a, $b) $u contains [0.25, 0.5) or [0.5, 0.75) s=$s")
        println("out: $out ", length(out) > 0 ? sum(y/2^i for (i, y) in enumerate(out)) : 0)
    end
    s += 1
    if a <= 0.25
        push!(out, 0)
        push!(out, fill(1, s)...)
        println("\temit ", "0"*"1"^s)
        annotate!((1.025,-(N-1+0.5)*3, text("0"*"1"^s, 5, :left, :vcenter, :black)))
    else
        push!(out, 1)
        push!(out, fill(0, s)...)
        println("\temit ", "1"*"0"^s)
        annotate!((1.025,-(N-1+0.5)*3, text("1"*"0"^s, 5, :left, :vcenter, :black)))
    end
    println("out: $out ", sum(y/2^i for (i, y) in enumerate(out)))

    ylims!(plt, (-N*3-1, 1))
    # plot!(plt, aspect_ratio=0.1, xlims=(0,1))
    # plot!(size=(-N*3-1, 1).*400)
    display(plt)
    # plot!(dpi=1000)
    savefig(plt, "Statistics/ArithmeticCoding/test.svg")

    return out
end

alphabet = ['I', 'K', 'W']
p = [18, 11, 11]
R = sum(p)
prec = 32
msg = ['K', 'I', 'W', 'I']

encoder_infinite_prec_rescale(alphabet, p./R, ['K', 'I', 'W'])
encoder_infinite_prec_rescale(alphabet, p./R, ['I', 'I', 'W'])
encoder_infinite_prec_rescale(alphabet, p./R, ['K', 'I', 'W', 'I'])
encoder_infinite_prec_rescale(alphabet, p./R, ['K', 'W', 'W', 'I'])
encoder_infinite_prec_rescale(alphabet, p./R, ['K', 'I', 'K', 'K'])
encoder_infinite_prec_rescale(alphabet, p./R, ['K', 'I', 'W', 'W'])
encoder_infinite_prec_rescale(alphabet, p./R, ['I', 'I', 'I', 'I'])

o = encoder_infinite_prec(alphabet, p./R, ['K', 'I', 'W', 'I'])
o = encoder_infinite_prec(alphabet, p./R, ['K', 'I', 'W'])
o = encoder_infinite_prec(alphabet, p./R, ['K', 'I'])

strs = String[]
d = Float64[]
for (i, x1) in enumerate(alphabet), (j,x2) in enumerate(alphabet), (k, x3) in enumerate(alphabet)
    println(x1,x2,x3)
    push!(strs, x1*x2*x3)
    c = length(d) > 0 ? d[end] : 0
    push!(d, c + p[i] * p[j] * p[k] / R^3)
end

# d = cumsum([1/3^3 for i in 1:3^3])
insert!(d, 1, 0.)
x = []
y = []
for i in 1:length(d)-1
    push!(x, i); push!(y, d[i])
    push!(x, i); push!(y, d[i+1])
end
push!(x, length(d)-0.5); push!(y, 1.)

using Plots.PlotMeasures
plot(x, y, legend=false, bottom_margin=10mm, xlims=(0,length(strs)+1))
xticks!((collect(1:length(strs)), strs), rotation=90)
# hline!([0.53971875, 0.57375], ls=:dash)
plot!([0, 12], [0.53971875, 0.53971875],ls=:dash,color=2)
plot!([0, 12], [0.57375, 0.57375],ls=:dash,color=2)

savefig("cdf.svg")

function decoder_infinite_prec(alphabet::Vector{T}, p::Vector{Float64}, ys::BitVector)::Vector{T} where T
    d = cumsum(p)
    c = d - p
    N = length(alphabet)

    a = 0
    b = 1

    z = sum(y/2^i for (i, y) in enumerate(ys))

    out = Vector{T}()
    while true
        for j in 1:N
            w = b - a
            b0 = a + w * d[j]
            a0 = a + w * c[j]
            if a0 <= z && z < b0
                push!(out, alphabet[j])
                a = a0
                b = b0
                if j == 1 # EOF
                    return out
                end
                break
            end
        end
    end
    return out
end

alphabet = [0, 1, 2]
p = [0.2,0.4,0.4]
msg = [2,1,0]

enc_msg = encoder_infinite_prec(alphabet, p, msg)
decoder_infinite_prec(alphabet, p, enc_msg)

function encoder_finite_prec(alphabet::Vector{T}, p::Vector{Int}, R::Int, precision::Int, xs::Vector{T})::BitVector where T
    d = cumsum(p)
    c = d - p
    charmap = Dict(c => i for (i,c) in enumerate(alphabet))

    ONE = 1 << precision
    HALF = ONE >> 1
    QUARTER = HALF >> 1

    a = 0
    b = ONE

    out = BitVector()
    s = 0
    for x in xs
        i = charmap[x]
        w = b - a
        b = a + (w * d[i]) ÷ R
        a = a + (w * c[i]) ÷ R
        @assert 0 <= a && a <= ONE "a: $a"
        @assert 0 <= b && b <= ONE "b: $b"

        while b < HALF || a > HALF
            if b < HALF
                push!(out, 0)
                push!(out, fill(1, s)...)
                a = 2*a
                b = 2*b
                s = 0
            elseif a > HALF
                push!(out, 1)
                push!(out, fill(0, s)...)
                a = 2*(a-HALF)
                b = 2*(b-HALF)
                s = 0
            end
        end
        while a > QUARTER && b < 3*QUARTER
            s += 1
            a = 2*(a-QUARTER)
            b = 2*(b-QUARTER)
        end
        @assert 0 <= a && a <= ONE "a: $a"
        @assert 0 <= b && b <= ONE "b: $b"
    end
    s += 1
    if a <= QUARTER
        push!(out, 0)
        push!(out, fill(1, s)...)
    else
        push!(out, 1)
        push!(out, fill(0, s)...)
    end

    return out
end

function decoder_finite_prec(alphabet::Vector{T}, p::Vector{Int}, R::Int, precision::Int, ys::BitVector)::Vector{T} where T
    d = cumsum(p)
    c = d - p
    N = length(alphabet)
    M = length(ys)

    ONE = 1 << precision
    HALF = ONE >> 1
    QUARTER = HALF >> 1

    a = 0
    b = ONE

    # read bits up to precision
    z = 0
    i = 1
    while i <= precision && i <= M
        if ys[i] == 1
            z += 1 << (precision-i)
        end
        i += 1
    end
    # z = y[1]...y[prec]

    out = Vector{T}()
    while true
        # decode symbol
        for j in 1:N
            w = b - a
            b0 = a + (w * d[j]) ÷ R
            a0 = a + (w * c[j]) ÷ R
            if a0 <= z && z < b0
                push!(out, alphabet[j])
                a = a0
                b = b0
                if j == 1 # EOF
                    return out
                end
                break
            end
        end
        # rescale at exactly same time as in encoding
        while b < HALF || a > HALF
            if b < HALF
                a = 2*a
                b = 2*b
                z = 2*z
            elseif a > HALF
                a = 2*(a-HALF)
                b = 2*(b-HALF)
                z = 2*(z-HALF)
            end
            # read next bit
            if i <= M && ys[i] == 1
                z += 1
            end
            i += 1
        end
        while a > QUARTER && b < 3*QUARTER
            a = 2*(a-QUARTER)
            b = 2*(b-QUARTER)
            z = 2*(z-QUARTER)
            # read next bit
            if i <= M && ys[i] == 1
                z += 1
            end
            i += 1
        end
    end
    return out
end

alphabet = [0, 1, 2]
p = [1,2,2]
R = sum(p)
prec = 32
msg = [2,1,0]

enc_msg = encoder_finite_prec(alphabet, p, R, prec, msg)
decoder_finite_prec(alphabet, p, R, prec, enc_msg)

alphabet = ['0', 'I', 'K', 'W']
p = [2, 20, 10, 10]
R = sum(p)
prec = 32
msg = ['K', 'I', 'W', 'I', '0']

enc_msg = encoder_finite_prec(alphabet, p, R, prec, msg)
decoder_finite_prec(alphabet, p, R, prec, enc_msg)

encoder_infinite_prec_rescale(alphabet, p./R, msg)


file = read("Statistics/ArithmeticCoding/arithmetic_coding.jl", String)
file *= '\x00'
char_count = Dict{Char,Int}()
for c in file
    char_count[c] = get(char_count, c, 0)+ 1
end
alphabet = sort([k for (k,v) in char_count])
R = sum(v for (k,v) in char_count)
p = [char_count[c] for c in alphabet]
prec = 32
msg = [c for c in file]
enc_msg = encoder_finite_prec(alphabet, p, R, prec, msg)
dec_msg = decoder_finite_prec(alphabet, p, R, prec, enc_msg)
reduce(*, dec_msg) == file

# bit
length(file) * 8
length(enc_msg)
length(enc_msg) / (length(file) * 8)

# byte
sizeof(file)
sizeof(enc_msg)

import Distributions: Categorical, Geometric, mean, entropy

function estimate_entropy(p::Vector{Int}, R::Int, n_iter::Int=10000)::Float64
    w = p ./ R
    char_dist = Categorical(w)
    est_entropy = 0.0
    for n in 1:n_iter
        e = 0.
        while true
            c = rand(char_dist)
            e -= log2(w[c])
            c == 1 && break
        end
        est_entropy = (n-1)/n * est_entropy + e/n
    end
    return est_entropy
end


function estimate_mean_enc_length(p::Vector{Int}, R::Int, prec::Int, n_iter::Int=10000)::Float64
    w = p ./ R
    mean_msg_length = mean(Geometric(w[1]))
    println("mean_msg_length: ", mean_msg_length)
    char_dist = Categorical(w)
    alphabet = collect(1:length(p))
    est_mean_length = 0.0
    for n in 1:n_iter
        msg = Int[]
        while true
            c = rand(char_dist)
            push!(msg, c)
            c == 1 && break
        end
        l = length(encoder_finite_prec(alphabet, p, R, prec, msg))
        est_mean_length = (n-1)/n * est_mean_length + l/n
    end
    return est_mean_length
end

p = [1,1,1,1,1]
R = sum(p)
estimate_entropy(p, R)
estimate_mean_enc_length(p, R, 32)

entropy(p./R, 2) / log2(2)
encoder_finite_prec(collect(1:5), p, R, 32, [5])
