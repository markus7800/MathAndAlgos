
mutable struct Dense
    W::DMat
    b::DVec
    σ::Function
end

function (a::Dense)(x::Union{AbstractVector, DVec})
  W, b, σ = a.W, a.b, a.σ
  σ(W*x + b)
end

function glorot(N...)
    x = sqrt(6 / sum(N))
    W = x*(2*rand(N...).-1) # uniform [-x,x]
end


function Dense(in::Int, out::Int, σ::Function=identity; init=:glorot)
    if init == :glorot
        W = glorot(out, in)
    elseif init == :normal
        W = randn(out,in)
    else
        @warn "Uniform init. u stupid?"
        W = rand(out,in)
    end
    Dense(DMat(W), DVec(zeros(out)), σ)
end

# function update_GDS!(m::Dense; η=0.01)
#     m.W.s .-= η * m.W.∇
#     m.b.s .-= η * m.b.∇
#
#     m.W.∇ .= 0
#     m.b.∇ .= 0
# end


function update_GDS!(m::Dense, opt)
    update!(opt, m.W.s, m.W.∇)
    update!(opt, m.b.s, m.b.∇)

    m.W.∇ .= 0
    m.b.∇ .= 0
end

function zero_∇!(m::Dense)
    m.W.∇ .= 0
    m.b.∇ .= 0
end
