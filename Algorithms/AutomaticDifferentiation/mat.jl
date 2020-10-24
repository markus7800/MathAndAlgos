mutable struct DMat <: DType
    s::Matrix{Float64} # value
    ∇::Matrix{Float64} # grad

    prev::Vector{DType}
    backward::Function

    op::String

    function DMat(s::Matrix{Float64}, ∇::Matrix{Float64}=zeros(size(s));
        prev=DType[], op="", bw::Function=∇->nothing)

        this = new()
        this.s = s
        this.∇ = ∇
        this.backward = bw
        this.prev = prev # unique?
        this.op = op
        return this
    end
end

# import LinearAlgebra.adjoint
# function adjoint(d::DVec)
#     DMat(Matrix(d.s'), Matrix(d.∇'), prev=d.prev, op=d.op)
# end
#
# function adjoint(d::DMat)
#     d.s .= d.s'
#     d.∇ .= d.∇'
#     d
# end
