mutable struct DVec <: DType
    s::Vector{Float64} # value
    ∇::Vector{Float64} # grad

    prev::Vector{DType}
    backward::Function

    op::String

    function DVec(s::Vector{Float64}, ∇::Vector{Float64}=zeros(length(s));
            prev=DType[], op="")

        this = new()
        this.s = s
        this.∇ = ∇
        this.backward = () -> nothing
        this.prev = prev # unique?
        this.op = op
        return this
    end
end
