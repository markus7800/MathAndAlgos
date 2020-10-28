
mutable struct Dense
    W::DMat
    b::DVec
    σ::Function
end

function (a::Dense)(x::Union{AbstractVector, DVec})
  W, b, σ = a.W, a.b, a.σ
  σ(W*x + b)
end

function Dense(in::Int, out::Int, σ::Function=identity; init=:glorot)
    if init == :glorot
        x = sqrt(6 / (in + out))
        W = x*(2*rand(out,in).-1) # uniform [-x,x]
    elseif init == :normal
        W = randn(out,in)
    else
        W = rand(out,in)
    end
    Dense(DMat(W), DVec(zeros(out)), σ)
end

function update_GDS!(m::Dense; η=0.01)
    m.W.s .-= η * m.W.∇
    m.b.s .-= η * m.b.∇

    m.W.∇ .= 0
    m.b.∇ .= 0
end
