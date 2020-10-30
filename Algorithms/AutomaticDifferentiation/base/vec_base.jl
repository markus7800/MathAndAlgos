# conversion
function DVec(v::Vector{DVal})
    res = DVec(map(d -> d.s, v), prev=v, op="vec<-[val]")
    res.backward = function bw(∇)
        for (d, g) in zip(res.prev, ∇)
            d.∇ += g # just pass down gradients
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
