
function sigma(x)
    1/(1+exp(-x))
end

# import Flux.logsoftmax
logsoftmax(v::DVec) = v - log(sum(exp(v)))

# import Flux.logitcrossentropy
logitcrossentropy(ŷ::DVec, y::Vector) = -sum(y * logsoftmax(ŷ))

v = DVec([1., 2., 3.])

logitcrossentropy(v, [true, false, false])
