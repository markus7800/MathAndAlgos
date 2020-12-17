mutable struct DTensor <: DType
    s::Array{Float64} # value
    ∇::Array{Float64} # grad

    prev::Vector{DType}
    backward::Function

    op::String

    function DTensor(s::Array{Float64}, ∇::Array{Float64}=zeros(size(s));
        prev=DType[], op="", bw::Function=∇->nothing)
        @assert size(s) == size(∇)

        this = new()
        this.s = s
        this.∇ = ∇
        this.backward = bw
        this.prev = prev # unique?
        this.op = op
        return this
    end
end

import Base.size
size(d::DTensor) = size(d.s)
