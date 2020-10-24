abstract type DType end

include("val.jl")
include("vec.jl")
include("mat.jl")

# Type Conversion

DVal(d::DMat) = DVal(d.s[1], d.∇[1], prev=d.prev, op=d.op, bw=d.backward)
DVal(d::DVec) = DVal(d.s[1], d.∇[1], prev=d.prev, op=d.op, bw=d.backward)

DVec(d::DMat) = DVec(vec(d.s), vec(d.∇), prev=d.prev, op=d.op, bw=d.backward)
DVec(d::DVal) = DVec([d.s], [d.∇], prev=d.prev, op=d.op, bw=d.backward)

DMat(d::DVec) = DMat(reshape(d.s,:,1), reshape(d.∇,:,1), prev=d.prev, op=d.op, bw=d.backward)
DMat(d::DVal) = DMat(DVec(d))


d = DVal(1π)
DVec(d)
DMat(d)

d = DVec([1π, 2π])
DVal(d)
DMat(d)

d = DMat([1π 2π; 3π 4π])
DVal(d)
DVec(d)

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


d = DMat([1. 2.; 3. 4.])
demote(d)

d = DVec([1., 2.])
demote(d)

d = DVal(1.)
demote(d)

d = DMat([1. 2.;])
demote(d)

d = DMat(Matrix(adjoint([1. 2.])))
demote(d)

d = DMat(reshape([1.], 1, 1))
demote(d)


function backward(d::DType)
    @assert size(d.s) == ()

    # topological order all of the children in the graph
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

    # go one variable at a time and apply the chain rule to get its gradient
    # dval.∇ .= 1
    if d isa DVal
        d.∇ = 1
    elseif d isa DVec
        d.∇ = ones(length(d.s))
    elseif d isa DMat
        d.∇ = ones(size(d.s))
    else
        error("PANIC")
    end

    for v in reverse(topo)
        v.backward(v.∇)
    end
end

include("val_base.jl")
include("vec_base.jl")
include("mat_base.jl")

# function gradient(f::Function, params::Vector)
#     map!(p -> DVal(p), params, params)
#     r = f()
#     r.backward()
#     r.s, map(p -> p.∇, params)
#     # map!(Float64(params))
# end
