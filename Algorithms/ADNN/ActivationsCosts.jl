

sigmoid(x::DType) = 1/(1+exp(-x))


# import Flux.logsoftmax
logsoftmax(v::DVec) = v - log(sum(exp(v)))

# import Flux.logitcrossentropy
logitcrossentropy(ŷ::DVec, y::Vector) = -sum(y * logsoftmax(ŷ))

relu(x::DType) = max(0,x)
