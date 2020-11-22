

sigmoid(x::DType) = 1/(1+exp(-x))


# import Flux.logsoftmax
logsoftmax(v::DVec) = v - log(sum(exp(v)))
logsoftmax(v::Vector) = v .- log(sum(exp.(v)))

softmax(v::DVec) = exp(v) / sum(exp(v))
softmax(v::Vector) = exp.(v) ./ sum(exp.(v))


# import Flux.logitcrossentropy
#logitcrossentropy(ŷ::DVec, y::Vector) = -sum(y * logsoftmax(ŷ))
logitcrossentropy(ŷ::Vector, y::Vector) = -sum(y .* logsoftmax(ŷ))
∇logitcrossentropy(ŷ::Vector, y::Vector) = -y + sum(y) * softmax(ŷ)

function logitcrossentropy(ŷ::DVec, y::Vector)
    res = DVal(logitcrossentropy(ŷ.s, y), prev=[ŷ], op="logitcrossentropy")
    res.backward = function bw(∇)
        ŷ.∇ .+= ∇logitcrossentropy(ŷ.s, y) .* ∇
    end
    return res
end

relu(x::DType) = max(0,x)
