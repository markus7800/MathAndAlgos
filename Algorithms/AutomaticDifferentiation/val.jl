
mutable struct DVal <: DType
    s::Float64 # value
    ∇::Float64 # grad

    prev::Vector{DType}
    backward::Function

    op::String

    function DVal(s::Float64, ∇::Float64=0.; prev=DType[],
        op="", bw::Function=∇->nothing)

        this = new()
        this.s = s
        this.∇ = ∇
        this.backward = bw
        this.prev = prev # unique?
        this.op = op
        return this
    end
end

import Base.zero
zero(::DVal) = DVal(0.)
zero(::Type{DVal}) = DVal(0.)
Float64(dval::DVal) = dval.s
