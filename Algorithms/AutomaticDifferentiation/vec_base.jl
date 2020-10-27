# conversion
function DVec(v::Vector{DVal})
    res = DVec(map(d -> d.s, v), prev=v, op="vec<-[val]")
    res.backward = function bw(∇)
        for (d, g) in zip(res.prev, ∇)
            d.backward(g) # just pass down gradients
        end
    end
end


import Base.+
function +(self::DVec, other::DVec)
    res = DVec(self.s + other.s, prev=[self, other], op="+")
    res.backward = function bw(∇)
        self.∇ .+= ∇
        other.∇ .+= ∇
    end
    return res
end

function +(self::DVec, v::Union{Number,Vector{T}}) where T <: Number
    res = DVec(self.s .+ v, prev=[self], op="+ C")
    res.backward = function bw(∇)
        self.∇ .+= ∇
    end
    return res
end

+(v::Union{Number,Vector{T}}, other::DVec) where T <: Number = other + v

function +(self::DVal, other::DVec)
    res = DVec(self.s .+ other.s, prev=[self, other], op="+")
    res.backward = function bw(∇)
        self.∇ += sum(∇)
        other.∇ .+= ∇
    end
    return res
end
+(self::DVec, other::DVal) = other + self

import Base.-
-(self::DVec) = self * (-1)

-(self::DVec, other::DVec) = self + (-other)
-(self::Union{Number,Vector{T}, DVal}, other::DVec) where T <: Number = self + (-other)
-(self::DVec, other::Union{Number,Vector{T}, DVal}) where T <: Number = self + (-other)


# elementwise
import Base.*
function *(self::DVec, other::DVec)
    res = DVec(self.s .* other.s, prev=[self, other], op=".*")
    res.backward = function bw(∇)
        self.∇ .+= other.s .* ∇
        other.∇ .+= self.s .* ∇
    end
    return res
end

function *(self::DVec, v::Union{Number,Vector{T}}) where T <: Number
    res = DVec(self.s .* v, prev=[self], op=".* C")
    res.backward = function bw(∇)
        self.∇ .+= v .* ∇
    end
    return res
end

*(v::Union{Number,Vector{T}}, other::DVec) where T <: Number = other * v

import Base./
#elementwise

function /(r::Number, other::DVec)
    res = DVec(r ./ other.s, prev=[other], op="$r ./")
    res.backward = function bw(∇)
        other.∇ .+= -r .* (1 ./ other.s.^2) .* ∇
    end
    return res
end

/(self::DVec, other::DVec) = self * (1/other)
/(self::DVec, other::Union{Number,Vector{T}}) where T <: Number = self * (1/other)
/(self::Vector{T}, other::DVec)  where T <: Number = self * (1/other)


# dot

function ⋅(self::DVec, other::DVec)
    res = DVal(self.s'other.s, prev=[self, other], op="⋅")
    res.backward = function bw(∇)
        self.∇ .+= other.s * ∇
        other.∇ .+= self.s * ∇
    end
    return res
end


function ⋅(self::DVec, v::Vector{T}) where T <: Number
    res = DVal(self.s'v, prev=[self], op="⋅")
    res.backward = function bw(∇)
        self.∇ .+= v .* ∇
    end
    return res
end

⋅(v::Vector{T}, other::DVec) where T <: Number = other ⋅ v

import Base.sum
sum(self::DVec) = self ⋅ ones(length(self.s))


import Base.exp
function exp(v::DVec)
    res = DVec(exp.(v.s), prev=[v], op="exp.")
    res.backward = function bw(∇)
        v.∇ += ∇ .* exp.(v.s)
    end
    return res
end


v = DVec([1., 2., 3.])
w = DVec([2., -3., 1.])
x = DVec([2., -2., 5.])

u = x⋅(v + w)

backward(u)
@assert v.∇ == x.s

v = DVal.([1., 2., 3.])
w = DVal.([2., -3., 1.])
x = DVal.([2., -2., 5.]')

u = x*(v + w)

backward(u[1])
map(d -> d.∇, v)
@assert map(d -> d.∇, v) == [2., -2., 5.]


# constants

v = DVec([1., 2., 4.])
w = [-1., 2., -0.5]

r = w ⋅ v
backward(r)

@assert v.∇ == w

v = DVec([1., 2., 4.])
w = [-1., 2., -0.5]

c = 2*(v.s + w)

u = w + v
r = u ⋅ u

w = [1., 1., 1.]

backward(r)


@assert v.∇ == c

# other

v = DVec([1., 2., -4.])
r = sum(v*v)
backward(r)
v.∇

w = DVec([1., 2., -4.])
r = w ⋅ w
backward(r)
w.∇

@assert v.∇ == w.∇
