# conversion
# function DTensor(v::Array{DVal})
#     res = DTensor(map(d -> d.s, v), prev=v, op="vec<-[val]")
#     res.backward = function bw(∇)
#         for (d, g) in zip(res.prev, ∇)
#             d.backward(g) # just pass down gradients
#         end
#     end
# end


import Base.+
function +(self::DTensor, other::DTensor)
    res = DTensor(self.s .+ other.s, prev=[self, other], op=".+")
    res.backward = function bw(∇)
        self.∇ .+= ∇
        other.∇ .+= ∇
    end
    return res
end

function +(self::DTensor, v::Union{Number,Array{T}}) where T <: Number
    res = DTensor(self.s .+ v, prev=[self], op="+ C")
    res.backward = function bw(∇)
        self.∇ .+= ∇
    end
    return res
end

+(v::Union{Number,Array{T}}, other::DTensor) where T <: Number = other + v

# function +(self::DVal, other::DVec)
#     res = DVec(self.s .+ other.s, prev=[self, other], op="+")
#     res.backward = function bw(∇)
#         self.∇ += sum(∇)
#         other.∇ .+= ∇
#     end
#     return res
# end
# +(self::DVec, other::DVal) = other + self

# import Base.-
# -(self::DVec) = self * (-1)
#
# -(self::DVec, other::DVec) = self + (-other)
# -(self::Union{Number,Vector{T}, DVal}, other::DVec) where T <: Number = self + (-other)
# -(self::DVec, other::Union{Number,Vector{T}, DVal}) where T <: Number = self + (-other)


# elementwise
import Base.*
function *(self::DTensor, other::DTensor)
    res = DTensor(self.s .* other.s, prev=[self, other], op=".*")
    res.backward = function bw(∇)
        self.∇ .+= other.s .* ∇
        other.∇ .+= self.s .* ∇
    end
    return res
end

function *(self::DTensor, other::Array{T}) where T <: Number
    res = DTensor(self.s .* other, prev=[self], op=".* C")
    res.backward = function bw(∇)
        self.∇ .+= other .* ∇
    end
    return res
end

*(self::Array{T}, other::DTensor) where T <: Number = other * self


import Base.getindex
import Base.setindex!
function getindex(self::DTensor, I...)
    res = DTensor(getindex(self.s, I...), prev=[self], op="[$I]")
    res.backward = function bw(∇)
        setindex!(self.∇, self.∇ .+ ∇, I...)
    end
    return res
end

import Base.sum
function sum(self::DTensor)
    res = DVal(sum(self.s), prev=[self], op="sum")
    res.backward = function bw(∇)
        self.∇ .+= ∇ # ∇ is scalar
    end
    return res
end

_A = [1  2 6.; 3 4. 7.; 8. 9. 10.]

getindex(_A, 1:2, 2:3)
setindex!(_A, [0. 0.; 0. 0.], 1:2, 2:3)

A = DTensor(_A)

getindex(A, 1:2, 2:3)


v = DVec([1., 2., 3.])

backward(sum(v))

v.∇

backward(sum(A))

A.∇
