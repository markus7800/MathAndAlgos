using Plots


f(t,y) =  y * cos(t)
sol = t -> exp(sin(t))
ts1, ys1 = RK(RK2, 2, f, 1.0, t0=0.0, t1=5.0, h0=1.0)
ts2, ys2 = RK(RK2, 2, f, 1.0, t0=0.0, t1=5.0, h0=0.5)
ts3, ys3 = RK(RK2, 2, f, 1.0, t0=0.0, t1=5.0, h0=0.1)
ts4, ys4 = RK(RK2, 2, f, 1.0, t0=0.0, t1=5.0, h0=0.1, ϵ=0.001, adaptive=true)

plot(sol, 0, 5, label="solution");
plot!(ts1, ys1, label="h=1.0");
plot!(ts2, ys2, label="h=0.5");
plot!(ts3, ys3, label="h=0.1");
plot!(ts4, ys4, label="adaptive ϵ=0.001");
title!("RK2")

#=
 For smaller stepsizes the approximation gets more precise.
 For a stepsize of h=0.1 or the adaptive version with ϵ=0.001
 the approximation is almost indistinguishable from the real
 solution.
=#

g(t,y) = [y[1] * cos(t), y[2] * sin(t)]

ts, ys = RK(RK5, 5, g, [1.0, 1.0], t0=0.0, t1=5.0, h0=0.1, ϵ=0.001, adaptive=true)

ys_1 = map(y->y[1], ys)
ys_2 = map(y->y[2], ys)

plot(ts, ys_1)
plot!(t -> exp(sin(t)), 0, 5)

plot!(ts, ys_2)
plot!(t -> exp(1-cos(t)), 0, 5)



h(t,y) = [-y[2], y[1]]

ts, ys = RK(RK5, 5, h, [1.0, 0.0], t0=0.0, t1=5.0, h0=0.1, ϵ=0.001, adaptive=true)

ys_1 = map(y->y[1], ys)
ys_2 = map(y->y[2], ys)

plot(ts, ys_1)
plot!(ts, ys_2)
