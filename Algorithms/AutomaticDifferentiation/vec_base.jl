
import Base.+
function +(self::DVec, other::DVec)
    res = DVec(self.s + other.s, prev=[self, other], op="+")
    res.backward = function bw(∇)
        self.∇ .+= ∇
        other.∇ .+= ∇
    end
    return demote(res)
end

function +(self::DVec, v::Union{Number,Vector{T}}) where T <: Number
    res = DVec(self.s .+ v, prev=[self], op="+")
    res.backward = function bw(∇)
        self.∇ .+= ∇
    end
    return demote(res)
end

+(v::Union{Number,Vector{T}}, other::DVec) where T <: Number = other + v

import Base.-
-(self::DVec) = self * (-1)

-(self::DVec, other::DVec) = self + (-other)
-(self::Union{Number,Vector{T}}, other::DVec) where T <: Number = self + (-other)
-(self::DVec, other::Union{Number,Vector{T}}) where T <: Number = self + (-other)

# elementwise
import Base.*
function *(self::DVec, other::DVec)
    res = DVec(self.s .* other.s, prev=[self, other], op="*")
    res.backward = function bw(∇)
        self.∇ .+= other.s .* ∇
        other.∇ .+= self.s .* ∇
    end
    return demote(res)
end

function *(self::DVec, v::Union{Number,Vector{T}}) where T <: Number
    res = DVec(self.s .* v, prev=[self], op="*")
    res.backward = function bw(∇)
        self.∇ .+= v .* ∇
    end
    return demote(res)
end

*(v::Union{Number,Vector{T}}, other::DVec) where T <: Number = other * v

import Base./
#elementwise

function /(r::Number, other::DVec)
    res = DVec(r ./ other.s, prev=[other], op="/")
    res.backward = function bw(∇)
        self.∇ .+= -v .* (1 ./ other.s.^2) .* ∇
    end
    return demote(res)
end

/(self::DVec, other::DVec) = self * (1/other)
/(self::DVec, other::Union{Number,Vector{T}}) where T <: Number = self * (1/other)
/(self::Union{Number,Vector{T}}, other::DVec)  where T <: Number = self * (1/other)


# dot

function ⋅(self::DVec, other::DVec)
    res = DVal(self.s'other.s, prev=[self, other], op="⋅")
    res.backward = function bw(∇)
        self.∇ .+= other.s * ∇
        other.∇ .+= self.s * ∇
    end
    return demote(res)
end


function ⋅(self::DVec, v::Vector{T}) where T <: Number
    res = DVal(self.s'v, prev=[self], op="⋅")
    res.backward = function bw(∇)
        self.∇ .+= v .* ∇
    end
    return demote(res)
end

⋅(v::Vector{T}, other::DVec) where T <: Number = other ⋅ v

import Base.sum
sum(self::DVec) = self ⋅ ones(length(self.s))

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

u = w + v
r = u ⋅ u
backward(r)

@assert v.∇ == 2*(v.s + w)

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
