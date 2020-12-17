import Flux
using BenchmarkTools
using Random
include("NN.jl")

function Conv(c::Flux.Conv, σ=identity)
    W = Float64.(flip(c.weight))
    b = Float64.(c.bias)
    stride = c.stride
	pad = c.pad[[1,2]]
    Conv(DTensor(W),DTensor(b),σ,stride,pad)
end
function Dense(m::Flux.Dense, σ=identity)
	Dense(DMat(Float64.(m.W)), DVec(Float64.(m.b)), σ)
end


Random.seed!(1)
conv_flux = Flux.Conv((5,5), 5=>10)
my_conv = Conv(conv_flux)
X = randn(100, 50, 5, 1)
X_ = reshape(X, 100, 50, 5)

sum(abs.(conv_flux(X) .- my_conv(X_).s))
sum(abs.(convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, X_) .-
	convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, X_)))

@btime conv_flux(X) # 5.526 ms (45 allocations: 692.05 KiB)
@btime my_conv(X_) # 5.547 ms (9 allocations: 690.45 KiB) vs prev 214.307 ms (1364570 allocations: 144.67 MiB)
@btime convolve_slice(my_conv.W.s, my_conv.b.s, my_conv.stride, my_conv.σ, X_) # 17.400 ms (70659 allocations: 50.67 MiB)
@btime convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, X_) # 12.331 ms (3 allocations: 345.11 KiB)
@btime convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, X_) # 5.515 ms (3 allocations: 345.11 KiB) (inbounds macro)
214.307 / 5.526 # faster than mine

Y = Float32.(X)
@btime conv_flux(Y) # 449.736 μs (48 allocations: 4.55 MiB)
5.526 / (0.449736) # faster than Float64
214.307 / (0.449736) # faster than mine

################################################################################

Random.seed!(1)
X = randn(100, 50, 5, 128)
@btime conv_flux(X) # 720.447 ms (45 allocations: 86.25 MiB)
5.526 * 128 # scales approximately as expected
(214.307 * 128) / 1000 # expected time for mine

Y = Float32.(X)
@btime conv_flux(Y) # 39.134 ms (302 allocations: 47.35 MiB)
(0.449736 * 128) / 39.134 # faster in batch
(214.307 * 128) / 39.134 # faster than mine iter

################################################################################


Random.seed!(1)
conv_flux = Flux.Conv((5,5), 5=>10)
X = randn(100, 50, 5, 1)
X_ = reshape(X, 100, 50, 5)
Y = Float32.(X)

ps = Flux.params(conv_flux, X)
gs = Flux.gradient(ps) do
	sum(conv_flux(X))
end
gs[ps[1]]

TX_ = DTensor(X_)
my_conv = Conv(conv_flux)
backward(sum(my_conv(TX_)))

sum(abs.(conv_flux(X) .- my_conv(TX_).s))
sum(abs.(gs[ps[3]] .- TX_.∇))
sum(abs.(gs[ps[2]] .- my_conv.b.∇))
sum(abs.(gs[ps[1]] .- flip(my_conv.W.∇)))


∇ = ones(size(my_conv(X))[1:3])

∇W, ∇b, ∇A = ∇convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, X_, ∇)


sum(abs.(gs[ps[3]] .- ∇A))
sum(abs.(gs[ps[2]] .- ∇b))
sum(abs.(gs[ps[1]] .- flip(∇W)))

sum(abs.(my_conv.W.∇ .- ∇W))



@btime begin
	TX_.∇ .= 0
	my_conv.W.∇ .= 0
	my_conv.b.∇ .= 0
	backward(sum(my_conv(TX_)))
end # 452.639 ms (1907797 allocations: 316.44 MiB)

@btime begin
	TX_.∇ .= 0
	my_conv.W.∇ .= 0
	my_conv.b.∇ .= 0
	A = DTensor(convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, TX_.s))
	backward(sum(A))
	∇W, ∇b, ∇TX_ = ∇convolve_loop(my_conv.W.s, my_conv.b.s, my_conv.stride, X_, A.∇)

	TX_.∇ .+= ∇TX_
	my_conv.W.∇ .+= ∇W
	my_conv.b.∇ .+= ∇b
end # 51.536 ms (39 allocations: 897.20 KiB)





Random.seed!(1)
mp_flux = Flux.MaxPool((5,5))
X = randn(100, 50, 5, 1)
X_ = reshape(X, 100, 50, 5)
Y = Float32.(X)

ps = Flux.params(mp_flux, X)
gs = Flux.gradient(ps) do
	sum(mp_flux(X))
end
gs[X]

TX_ = DTensor(X_)
my_mp = MaxPool((5,5))
r = sum(my_mp(TX_))
backward(r)

sum(abs.(mp_flux(X) .- my_mp(TX_).s))
sum(abs.(gs[ps[1]] .- TX_.∇))

∇ = ones(size(my_mp(TX_))[1:3])

out = maxpool(my_mp.size, my_mp.stride, TX_.s)
∇A = ∇maxpool(my_mp.size, my_mp.stride, TX_.s, out, ∇)


sum(abs.(gs[X] .- ∇A))


Random.seed!(1)
conv_flux = Flux.Conv((5,5), 5=>10, pad=(2,3))
my_conv = Conv(conv_flux)
X = randn(100, 50, 5, 1)
X_ = reshape(X, 100, 50, 5)

sum(abs.(conv_flux(X) .- my_conv(X_).s))

ps = Flux.params(conv_flux, X)
gs = Flux.gradient(ps) do
	sum(conv_flux(X))
end

TX_ = DTensor(X_)
my_conv = Conv(conv_flux)
backward(sum(my_conv(TX_)))

sum(abs.(conv_flux(X) .- my_conv(TX_).s))
sum(abs.(gs[ps[3]] .- TX_.∇))
sum(abs.(gs[ps[2]] .- my_conv.b.∇))
sum(abs.(gs[ps[1]] .- flip(my_conv.W.∇)))
