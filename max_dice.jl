

pdf(m::Int, n::Int, v::Int) = sum(binomial(m, k)*(1//n)^k * ((v-1)//n)^(m-k) for k in 1:m)
pdf2(m::Int, n::Int, v::Int) = (v//n)^m - ((v-1)//n)^m

m = 2
n = 6

ps = [pdf(m, n, v) for v in 1:n]
sum(ps)

E = ps'collect(1:n)

expectedvalue(m::Int, n::Int) = sum(v*((v-1)//n)^m * sum(binomial(m, k) * 1//(v-1)^k for k in 1:m) for v in 2:n) + 1//n^m
expectedvalue2(m::Int, n::Int) = 1//n^m * sum(v*(v^m-(v-1)^m) for v in 1:n)
expectedvalue3(m::Int, n::Int) = n - 1//n^m * sum(v^m for v in 1:(n-1))

zeta(m) = sum(v^m for v in 1:1_000_000)
zeta(m::Int) = zeta(Float64(m))

expectedvalue(m,n)
expectedvalue2(m,n)
expectedvalue3(m,n)

Float64(E) / n
m/(m+1)

[Float64(expectedvalue(m,n)) for n in 2:1000]

n - 1/n^m * zeta(-m)
