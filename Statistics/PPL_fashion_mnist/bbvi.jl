using Gen

@gen function model(xs::Vector{Float64})
    slope = @trace(normal(-1, exp(0.5)), :slope)
    intercept = @trace(normal(1, exp(2.0)), :intercept)
    for (i, x) in enumerate(xs)
        @trace(normal(slope * x + intercept, 1), (:y,i))
    end
end

@gen function approx(xs::Vector{Float64})
    @param slope_mu::Float64
    @param slope_log_std::Float64
    @param intercept_mu::Float64
    @param intercept_log_std::Float64
    slope = @trace(normal(slope_mu, exp(slope_log_std)), :slope)
    intercept = @trace(normal(intercept_mu, exp(intercept_log_std)), :intercept)
end

init_param!(approx, :slope_mu, 0.)
init_param!(approx, :slope_log_std, 0.)
init_param!(approx, :intercept_mu, 0.)
init_param!(approx, :intercept_log_std, 0.)

xs = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
ys = [8.23, 5.87, 3.99, 2.59, 0.23, -0.66, -3.53, -6.91, -7.24, -9.90]

observations = choicemap()
for (i, y) in enumerate(ys)
    observations[(:y, i)] = y
end

update = ParamUpdate(GradientDescent(1e-3, 100000), approx)
black_box_vi!(model, (xs,), observations, approx, (xs,), update;
    iters=1000, samples_per_iter=1000, verbose=true)
slope_mu = get_param(approx, :slope_mu)
slope_log_std = get_param(approx, :slope_log_std)
intercept_mu = get_param(approx, :intercept_mu)
intercept_log_std = get_param(approx, :intercept_log_std)

Gen.random(Gen.Normal, 0, 1)

methods(random)

using StatsPlots

scatter(xs, ys)
plot!(t -> slope_mu * t + intercept_mu)


import Distributions
plot(t -> Distributions.pdf(Distributions.Normal(-1, exp(0.5)), t))
vline!([slope_mu])

plot(t -> Distributions.pdf(Distributions.Normal(1, exp(2.0)), t), xlims=(-5,15))
vline!([intercept_mu])


using Flux

m = Chain(Dense(10, 10, sigmoid), Dense(10, 10, sigmoid))


@gen function test(m::Chain)
    @param params(m)
end

m.layers[1].weight

macro chain_params(m)
    println(m)
    args = []
    for i in length(m.layers)
        push!(args, Meta.parse("@param W_$i::Matrix{Float64}"))
        push!(args, Meta.parse("@param b_$i::Vector{Float64}"))
    end

    quote
        # Expr(:toplevel, args...)
        return $(esc(m))
    end
end

begin
    m = Chain(Dense(10, 10, sigmoid), Dense(10, 10, sigmoid))
    @chain_params(m)
end

ex3 = Meta.parse("(4 + 4) / 2; (8 / 2); 3/4")
Meta.show_sexpr(ex3)

toplevel = Expr(:toplevel,
    Meta.parse("println(\"test1\")"),
    quote println("test2") end,
    :(println("test3")))

eval(toplevel)

typeof(toplevel.args[1])

quote @param x::Float64 end
