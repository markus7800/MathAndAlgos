
mutable struct Dense
    W::DMat
    b::DVec
    σ::Function
end

function (a::Dense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ(W*x + b)
end

Dense(in::Int, out::Int, σ::Function=identity) = Dense(DMat(rand(out,in)), DVec(rand(out)), σ)

D = Dense(2, 5, sigma)

D([1., 2.])
