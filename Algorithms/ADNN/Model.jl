using Base: tail

struct Model{T<:Tuple}
  layers::T
  Model(xs...) = new{typeof(xs)}(xs)
end

applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))

(c::Model)(x) = applychain(c.layers, x)

# update_GDS!(m::Model) = foreach(update_GDS!, m.layers)
