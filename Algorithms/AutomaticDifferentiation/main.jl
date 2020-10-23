include("AD.jl")
using Random

a = DVal(10.)
b = DVal(-5.)
c = DVal(2.)
d = DVal(-0.5)

e = c * (a + d * b)
backward(e)


a = DVal(10.)
r = a + a + a*a
backward(r)
a.∇

# w = [10, -5, 2., -0.5]
# gradient(w) do
#     a,b,c,d = w
#     c * (a + d * b)
# end

# ∂e/∂a = c
a.∇ == c.s
# ∂e/∂b = c * d
b.∇ == (c * d).s

e.∇

a = DVal(2.)
r = log(exp(a))
backward(r)
r.∇


Random.seed!(1)
_A = rand(10,10)
_B = rand(10,10)

A = DVal.(_A)
B = DVal.(_B)


C = A * B
_C = _A * _B
all(Float64.(C) .≈ _C)
