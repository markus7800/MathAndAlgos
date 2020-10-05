using Optim

# type II maximum likelihood estimate
function optimize_params_1D(se::SE, xtrain::AbstractVector{Float64},
        ytrain::AbstractVector{Float64}; mean=:zero, σ=0.)

    n = length(ytrain)
    xs = xtrain'
    ys = ytrain

    if mean == :zero
        function F1(Q)
            c = Q[1]; l = Q[2]

            K = cov(SE(c,l), xs, xs) # n × n
            make_posdef!(K)
            A = inv(K + σ^2*I(n)) # n × n

            # (1 × n) * (n × n) * (n × 1)
            return ys' * A * ys - log(det(A)) # <- minimize log marginal likelihood
        end
        x0 = [se.c, se.l]
        res = optimize(F1, x0, NelderMead(), Optim.Options(iterations=10^5))
        display(res)
        c, l  = res.minimizer
        return MeanZero(), SE(c, l)
    elseif mean == :constant
        function F2(Q)
            c = Q[1]; l = Q[2]; m = Q[3]

            K = cov(SE(c,l), xs, xs) # n × n
            make_posdef!(K)
            A = inv(K + σ^2*I(n)) # n × n

            # (1 × n) * (n × n) * (n × 1)
            return (ys .- m)' * A * (ys .- m) - log(det(A)) # <- minimize log marginal likelihood
        end
        x0 = [se.c, se.l, 0.]
        res = optimize(F2, x0, NelderMead(), Optim.Options(iterations=10^5))
        display(res)
        println(res.minimizer)
        c, l, m  = res.minimizer
        return FunctionMean(x -> m), SE(c, l)
    elseif mean == :affine
        function F3(Q)
            c = Q[1]; l = Q[2]; k = Q[3]; d = Q[4]

            K = cov(SE(c,l), xs, xs) # n × n
            make_posdef!(K)
            A = inv(K + σ^2*I(n)) # n × n

            # (1 × n) * (n × n) * (n × 1)
            y = ys .- (k .* ys .+ d)
            return y' * A * y - log(det(A)) # <- minimize log marginal likelihood
        end
        x0 = [se.c, se.l, 0., 0.]
        res = optimize(F3, x0, NelderMead(), Optim.Options(iterations=10^5))
        display(res)
        println(res.minimizer)
        c, l, k, d  = res.minimizer
        return FunctionMean(x -> k .* x .+ d), SE(c, l)
    end
end

m, k = optimize_params_1D(SE(1.,1.), xtrain, ytrain, mean=:constant)

gp = GP(m, k)

plot_GP_1D(gp, xs)
scatter!(xtrain, ytrain)
