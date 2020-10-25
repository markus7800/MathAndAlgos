
import Base.+
function +(self::DVal, other::DVal)
    res = DVal(self.s + other.s, prev=[self, other], op="+")
    res.backward = function bw(∇)
        self.∇ += ∇
        other.∇ += ∇
    end
    return res
end

function +(self::DVal, r::Number)
    res = DVal(self.s + r, prev=[self], op="+")
    res.backward = function bw(∇)
        self.∇ += ∇
    end
    return res
end

+(r::Number, other::DVal) = other + r

import Base.-
-(self::DVal) = -1*self

-(self::DVal, other::DVal) = self + (-other)

-(self::DVal, r::Number) = self + (-r)

-(r::Number, other::DVal) = -(other + (-r))


import Base.*
function *(self::DVal, other::DVal)
    res = DVal(self.s * other.s, prev=[self, other], op="*")
    res.backward = function bw(∇)
        self.∇ += other.s * ∇
        other.∇ += self.s * ∇
    end
    return res
end

function *(self::DVal, r::Number)
    res = DVal(self.s * r, prev=[self], op="*")
    res.backward = function bw(∇)
        self.∇ += r * ∇
    end
    return res
end

*(self::Number, other::DVal) = other * self

import Base./
# d/da a/b = 1/b
# d/db a/b = -a/b^2
function /(self::DVal, other::DVal)
    res = DVal(self.s / other.s, prev=[self, other], op="*")
    res.backward = function bw(∇)
        self.∇ += 1/other.s * ∇
        other.∇ += -self.s/other.s^2 * ∇
    end
    return res
end

/(self::DVal, r::Number) = self * (1/r)
/(r::Number, other::DVal) = (1/r) * other


# res = exp(v)
# ∂/∂v L(res) (∂/∂v L)(res) d/dv res  = res.∇ * d/dv exp(v)
# ∂L/∂v = ∂L/∂res ∂res/∂v = res.∇ * d/dv exp(v)
import Base.exp
function exp(v::DVal)
    res = DVal(exp(v.s), prev=[v], op="exp")
    res.backward = function bw(∇)
        v.∇ += ∇ * exp(v.s)
    end
    return res
end

import Base.sin
function sin(v::DVal)
    res = DVal(sin(v.s), prev=[v], op="sin")
    res.backward = function bw(∇)
        v.∇ += ∇ * cos(v.s)
    end
    return res
end

import Base.cos
function sin(v::DVal)
    res = DVal(sin(v.s), prev=[v], op="cos")
    res.backward = function bw(∇)
        v.∇ += ∇ * -sin(v.s)
    end
    return res
end

import Base.log
function log(v::DVal)
    res = DVal(log(v.s), prev=[v], op="log")
    res.backward = function bw(∇)
        v.∇ += ∇ * 1/v.s
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


# constants

a = DVal(5.)
r = -3 * a
backward(r)
@assert a.∇ == -3

a = DVal(5.)
r = a + 3.
backward(r)
@assert a.∇ == 1
