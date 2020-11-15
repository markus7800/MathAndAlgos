abstract type DType end

include("val.jl")
include("vec.jl")
include("mat.jl")
include("tensor.jl")
include("show.jl")


# Type Conversion
DVal(d::DMat) = DVal(d.s[1], d.∇[1], prev=d.prev, op="val<-mat . " * d.op, bw=d.backward)
DVal(d::DVec) = DVal(d.s[1], d.∇[1], prev=d.prev, op="val<-vec . " * d.op, bw=d.backward)

DVec(d::DMat) = DVec(vec(d.s), vec(d.∇), prev=d.prev, op="vec<-mat . " * d.op, bw=d.backward)
DVec(d::DVal) = DVec([d.s], [d.∇], prev=d.prev, op="vec<-val . " * d.op, bw=∇ -> d.backward(∇[1]))

DMat(d::DVec) = DMat(
    reshape(d.s,:,1), reshape(d.∇,:,1),
    prev=d.prev,op="mat<-vec . " * d.op, bw= ∇ -> d.backward(vec(∇))
    )
DMat(d::DVal) = DMat(
    reshape([d.s],1,1), reshape([d.∇],1,1),
    prev=d.prev, op="mat<-val . " * d.op, bw= ∇ -> d.backward(∇[1])
    )


# Demotions

function demote(d::DType)
    N = length(size(d.s))
    if N == 1 && length(d.s) == 1
        return DVal(d)
    elseif N == 2
        m, n = size(d.s)
        if n == 1 || m == 1
            if m * n == 1
                return DVal(d)
            else
                return DVec(d)
            end
        end
    end
    return d
end



function backward(d::DVal)
    # order the tree nodes by depth
    topo = DType[]
    visited = Set{DType}()
    function build_topo(v)
        if !(v in visited)
            push!(visited, v)
            for child in v.prev
                build_topo(child)
            end
            push!(topo,v)
        end
    end

    build_topo(d)

    d.∇ = 1
    # go one variable at a time and apply the chain rule to get its gradient
    for v in reverse(topo)
        v.backward(v.∇)
    end
end

include("base/val_base.jl")
include("base/vec_base.jl")
include("base/mat_base.jl")
include("base/tensor_base.jl")
