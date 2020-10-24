
import Base.+
function +(self::DVec, other::DVec)
    res = DVec(self.s + other.s, prev=[self, other], op="+")
    res.backward = function bw(∇)
        self.∇ .+= ∇
        other.∇ .+= ∇
    end
    return demote(res)
end

function +(self::DVec, v::Vector{T}) where T <: Number
    res = DVec(self.s + v, prev=[self], op="+")
    res.backward = function bw(∇)
        self.∇ .+= ∇
    end
    return demote(res)
end

+(v::Vector{T}, other::DVec) where T <: Number = other + v

function ⋅(self::DVec, other::DVec)
    res = DVal(self.s'other.s, prev=[self, other], op="⋅")
    res.backward = function bw(∇)
        self.∇ .+= other.s .* ∇
        other.∇ .+= self.s .* ∇
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
