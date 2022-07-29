
using Random
using BenchmarkTools

Random.seed!(0)
A = randn(512, 100_000)
x = rand(0:1, 100_000)

@btime A*x

mutable struct NNUE
    _x::Vector
    W1::Matrix
    _a1::Vector
    function NNUE(W1::Matrix)
        return new(zeros(size(W1,2)), W1, zeros(size(W1,1)))
    end
end

function (nnue::NNUE)(x::Vector)
    out = nnue._a1
    for j in 1:length(out)
        for i in 1:length(x)
            if x[i] == 0 && nnue._x[i] == 1
                out[j] -= nnue.W1[j,i]
            end
            if x[i] == 1 && nnue._x[i] == 0
                out[j] += nnue.W1[j,i]
            end
        end
    end
    return out
end

nnue = NNUE(A)

@btime nnue(x)
