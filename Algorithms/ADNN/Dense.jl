
mutable struct Dense
    W::DMat
    b::DVec
    σ::Function
end

function (a::Dense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ(W*x .+ b)
end
