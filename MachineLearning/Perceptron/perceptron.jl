
function classify(X, w)
    n, p = size(X)
    dX = hcat(ones(n), X)
    _classify(dX, w)
end

function _classify(dX, w)
    sign.(dX*w)
end

function perceptron(X, y; w_true=nothing)
    n, p = size(X)
    dX = hcat(ones(n), X)

    if w_true != nothing
        R2 = maximum(sum(X.^2,dims=2))
        γ2 = minimum(y .* (dX * w_true))^2
        @info "Theoretical maximum number of iterations: " ceil(R2/γ2)
    end


    w = zeros(p+1)
    ws = []

    k = 0
    miss_class = findall(_classify(dX, w) .!= y)
    while length(miss_class) != 0
        i = rand(miss_class)
        w = w + y[i]*dX[i,:]
        push!(ws, w)
        miss_class = findall(_classify(dX, w) .!= y)
        k += 1
    end

    @info "Finished in $k iterations."

    return w, ws
end
