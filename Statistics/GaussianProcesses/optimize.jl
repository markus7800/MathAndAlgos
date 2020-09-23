using Optim

function optimize_params_1D(se::SE, xtrain::AbstractVector{Float64},
        ytrain::AbstractVector{Float64}, mean=:zero; σ=0.)

    if mean == :zero
        n = length(ytrain)
        xs = xtrain'
        ys = ytrain
        function F(θ)
            c = θ[1]; l = θ[2]

            K = cov(SE(c,l), xs, xs) # n × n
            make_posdef!(K)
            A = inv(K + σ^2*I(n)) # n × n

            # (1 × n) * (n × n) * # (n × 1)
            # println(det(A))
            return ys' * A * ys - log(det(A)) # <- minimize
        end
        x0 = [se.c, se.l]
        return optimize(F, x0, NelderMead(), Optim.Options(iterations=10^5))

        # return optimize(F, [0., 0.], [5.,5.], x0, SAMIN(), Optim.Options(iterations=10^5))
    end
end

res = optimize_params_1D(SE(1.,1.), xtrain, ytrain)

s, l = res.minimizer
gp = GP(MeanZero(), SE(s,l))

plot_GP_1D(gp, xs)
scatter!(xtrain, ytrain)

A = randn(10,10)
det(A)

A .*= 10
det(A)
