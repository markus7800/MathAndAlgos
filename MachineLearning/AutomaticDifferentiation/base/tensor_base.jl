# conversion
# function DTensor(v::Array{DVal})
#     res = DTensor(map(d -> d.s, v), prev=v, op="vec<-[val]")
#     res.backward = function bw(∇)
#         for (d, g) in zip(res.prev, ∇)
#             d.backward(g) # just pass down gradients
#         end
#     end
# end

function shape_diff(self, other)
    s1 = size(self)
    s2 = size(other)
    l = max(length(s1), length(s2))
    _s1 = vcat(s1..., zeros(l-length(s1)))
    _s2 = vcat(s2..., zeros(l-length(s2)))
    dims1 = findall(_s2 .> _s1)
    dims2 = findall(_s1 .> _s2)
    return s1, dims1, s2, dims2
end


import Base.+
function +(self::DTensor, other::DTensor)
    # determine shape differences
    s1, dims1, s2, dims2 = shape_diff(self, other)
    # if shape is same dims1 = dims2 = [] and sum does nothing

    res = DTensor(self.s .+ other.s, prev=[self, other], op=".+")
    res.backward = function bw(∇)
        r1 = sum(∇, dims=dims1) # sum not matching dims up to "reverse broadcasting"
        r2 = sum(∇, dims=dims2)
        self.∇ .+= reshape(r1, s1)
        other.∇ .+= reshape(r2, s2)
    end
    return res
end

function +(self::DTensor, v::Union{Number,Array{T}}) where T <: Number
    s1, dims1, = shape_diff(self, v)
    res = DTensor(self.s .+ v, prev=[self], op="+ C")
    res.backward = function bw(∇)
        self.∇ .+= reshape(sum(∇, dims=dims1), s1)
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
    # determine shape differences
    s1, dims1, s2, dims2 = shape_diff(self, other)
    # if shape is same dims1 = dims2 = [] and sum does nothing

    res = DTensor(self.s .* other.s, prev=[self, other], op=".*")
    res.backward = function bw(∇)
        r1 = sum(other.s .* ∇, dims=dims1) # sum not matching dims up to "reverse broadcasting"
        r2 = sum(self.s .* ∇, dims=dims2)
        self.∇ .+= reshape(r1, s1)
        other.∇ .+= reshape(r2, s2)
    end
    return res
end

function *(self::DTensor, other::Array{T}) where T <: Number
    # determine shape differences
    s1, dims1, s2, dims2 = shape_diff(self, other)
    # if shape is same dims1 = dims2 = [] and sum does nothing

    res = DTensor(self.s .* other, prev=[self], op=".* C")
    res.backward = function bw(∇)
        self.∇ .+= reshape(sum(other .* ∇, dims=dims1), s1)
    end
    return res
end

*(self::Array{T}, other::DTensor) where T <: Number = other * self


import Base.getindex
import Base.setindex!
function getindex(self::DTensor, I...)
    res = DTensor(getindex(self.s, I...), prev=[self], op="[$I]")
    res.backward = function bw(∇)
        S = getindex(self.∇, I...)
        setindex!(self.∇, S .+ ∇, I...)
    end
    return res
end

import Base.sum
# TODO: make prettier
function sum(self::DTensor; dims=nothing)
    if dims == nothing
        res = DVal(sum(self.s), prev=[self], op="sum")
        res.backward = function bw1(∇)
            self.∇ .+= ∇ # ∇ is scalar
        end
        return res
    else
        r = sum(self.s, dims=dims)
        res = DTensor(r, prev=[self], op="sum")
        res.backward = function bw2(∇)
            self.∇ .+= ∇
        end
    end
    return res
end

import Base.reshape
function reshape(self::DTensor, I...)
    shape = size(self.s)
    res = DTensor(reshape(self.s, I...), prev=[self], op="reshape")
    res.backward = function bw(∇)
        self.∇ .+= reshape(∇, shape...) # ∇ is scalar
    end
    return res
end

# tensors have to be vector or value and same shape
function DTensor(arr::Array{DTensor,2})
    m, n = size(arr)
    d = size(arr[1,1].s)
    ranges = map(i->1:i, d)
    T = Array{Float64}(undef, m, n, d...)
    for i in 1:m, j in 1:n
        T[i,j,ranges...] .= arr[i,j].s
    end
    res = DTensor(T, prev=vec(arr), op="conv")
    res.backward = function bw(∇)
        for i in 1:m, j in 1:n
            arr[i,j].∇ += reshape(∇[i,j,ranges...], d)
        end
    end
    return res
end

function DTensor(A::Array{DVal})
    res = DTensor(map(d -> d.s, A), prev=vec(A), op="tensor<-[val]")
    res.backward = function bw(∇)
        for (d, g) in zip(res.prev, vec(∇)) # prev stored in vectorised form
            d.∇ += g # just pass down gradients
        end
    end
    return res
end

function flatten(self::DTensor)
    shape = size(self)
    res = DVec(vec(self.s), prev=[self], op="flatten")
    res.backward = function bw(∇)
        self.∇ .+= reshape(∇, shape)
    end
    return res
end

import Base.max
function max(r::Number, self::DTensor)
    res = DTensor(max.(r, self.s), prev=[self], op="max.($r,.)")
    res.backward = function bw(∇)
        I = self.s .== res.s
        self.∇[I] .+= ∇[I]
    end
    return res
end

import Base.maximum
function maximum(self::DTensor)
    v, i = findmax(self.s)
    res = DVal(v, prev=[self], op="maximum")
    res.backward = function bw(∇)
        self.∇[i] += ∇
    end
    return res
end
