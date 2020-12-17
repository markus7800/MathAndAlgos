
import Base.show
function Base.show(io::IO, l::Conv)
  print(io, "Conv(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
  print(io, ", stride: ", l.stride)
  print(io, ", pad: ", l.pad)
  l.σ == identity || print(io, ", ", l.σ)

  print(io, ")")
end

function Base.show(io::IO, l::BatchNorm)
  print(io, "BatchNorm($(join(size(l.β), ", "))")
  print(io, ", eps: ", l.ϵ)
  print(io, ", momentum: ", l.momentum)
  (l.λ == identity) || print(io, ", λ = $(l.λ)")
  print(io, ")")
end

function show(io::IO, CB::ConvBlock)
    println(io, "-Convolution Block")
    println(io, CB.conv)
    println(io, CB.bn)
    print(io, CB.σ)
    if CB.pool
        print(io, "\n")
        print(io, CB.mp)
    end
end

function show(io::IO, RB::ResBlock)
    println(io, "-Residual Block")
    println(io, RB.conv_block_1)
    print(io, RB.conv_block_1)
end

function pretty_print(c::Chain)
    for layer in c.layers
        println(layer)
    end
end
