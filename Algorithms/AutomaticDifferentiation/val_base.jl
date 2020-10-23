
import Base.+
function +(self::DVal, other::DVal)
    res = DVal(self.s + other.s, prev=[self, other], op="+")
    res.backward = function bw()
        self.∇ += res.∇
        other.∇ += res.∇
    end
    return res
end

# TODO:
# @macroexpand @D (a,b) -> a*b (a,b) -> [b, a]
# @D (a,b) -> a*b (a,b) -> [b, a]
# macro D(X::Expr, Y::Expr)
#     quote
#         $(esc(X))
#         $Y
#     end
# end

import Base.*
function *(self::DVal, other::DVal)
    res = DVal(self.s * other.s, prev=[self, other], op="*")
    res.backward = function bw()
        self.∇ += other.s * res.∇
        other.∇ += self.s * res.∇
    end
    return res
end

# res = exp(v)
# ∂/∂v L(res) (∂/∂v L)(res) d/dv res  = res.∇ * d/dv exp(v)
# ∂L/∂v = ∂L/∂res ∂res/∂v = res.∇ * d/dv exp(v)
import Base.exp
function exp(v::DVal)
    res = DVal(exp(v.s), prev=[v], op="exp")
    res.backward = function bw()
        v.∇ += res.∇ * exp(v.s)
    end
    return res
end

import Base.sin
function sin(v::DVal)
    res = DVal(sin(v.s), prev=[v], op="sin")
    res.backward = function bw()
        v.∇ += res.∇ * cos(v.s)
    end
    return res
end

import Base.cos
function sin(v::DVal)
    res = DVal(sin(v.s), prev=[v], op="cos")
    res.backward = function bw()
        v.∇ += res.∇ * -sin(v.s)
    end
    return res
end

import Base.log
function log(v::DVal)
    res = DVal(log(v.s), prev=[v], op="log")
    res.backward = function bw()
        v.∇ += res.∇ * 1/v.s
    end
    return res
end



# TESTS

a = DVal(10.)
b = DVal(-5.)
c = DVal(2.)
d = DVal(-0.5)

e = c * (a + d * b)
backward(e)
@assert a.∇ == c.s && b.∇ == c.s*d.s

a = DVal(10.)
r = a + a + a*a
backward(r)
@assert a.∇ == 2 + 2*a.s


a = DVal(2.)
r = log(exp(a))
backward(r)
@assert r.∇ == 1

using Random
Random.seed!(1)
_A = rand(10,10)
_B = rand(10,10)

A = DVal.(_A)
B = DVal.(_B)


C = A * B
_C = _A * _B
@assert all(Float64.(C) .≈ _C)
