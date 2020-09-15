# 9.8

#=
 Implementation of adaptive Runge-Kutta method
=#

function RK(T::Array{Float64,2}, p::Int, f::Function, y0::Vector{Float64};
    t0::Float64, t1::Float64, h0::Float64, ϵ::Float64=0.001, adaptive::Bool=false)::Tuple{Vector{Float64},Vector{Vector{Float64}}}

    @assert T[1,1]==0; @assert h0 >0; @assert ϵ>0;

    # extract Matrix and Vectors from T
    local A = T[1:end-2, 2:end-1]
    local b = T[end-1, 2:end]
    local b_star = T[end, 2:end] # now two b rows
    local c = T[1:end-2, 1]
    local s = size(b, 1)

    @assert c[1] == 0
    @assert all(c[i] ≈ sum(A[i,j] for j in 1:i-1) for i in 2:size(A,1))

    local k = Vector{Vector{Float64}}(undef, s)

    local ys = []; local ts = []
    push!(ys, y0); push!(ts, t0)

    function calc_y(t, y, h)
        # calculate k_i, using the fact that T is in explicit form
        k[1] = f(t, y)
        for i in 2:s
            k[i] = f(t + h * c[i], y + h * sum(A[i,j]*k[j] for j in 1:i-1))
        end
        y_est = y + h * sum(b[i] * k[i] for i in 1:s)
        y_star = y + h * sum(b_star[i] * k[i] for i in 1:s)
        return y_est, y_star
    end

    local h = h0; local t = ts[end]
    while t < t1
        t = ts[end]
        y = ys[end]
        y_est, y_star = calc_y(t, y ,h) # calculate y

        ϵ_est = maximum(abs.(y_est - y_star))
        if ϵ_est > ϵ && adaptive # error too big
            h = h * (ϵ/ϵ_est)^(1/p) # adjust stepsize
            if t + h > t1 # do not overstep interval
                h = t1 - t
            end
            y_est, y_star = calc_y(t, y ,h) # recalculate y with new h
        end
        t = t + h
        if t ≤ t1
            push!(ys, y_est); push!(ts, t)
        end
    end

    return ts, ys
end

function RK(T::Array{Float64,2}, p::Int, f::Function, y0::Float64;
    t0::Float64, t1::Float64, h0::Float64, ϵ::Float64=0.001, adaptive::Bool=false)::Tuple{Vector{Float64}, Vector{Float64}}
    ts, ys = RK(T, p, f, [y0], t0=t0, t1=t1, h0=h0, ϵ=ϵ, adaptive=adaptive)
    ys = map(y->y[1], ys)
    return ts, ys
end

# Butcher-Tableaux for order 2 and 5

# (forward euler and improved euler)
const RK2 = [
    0    NaN    NaN;
    1    1      NaN;
    NaN  1/2    1/2;
    NaN  1      0
]

const RK5 = [
    0      NaN       NaN         NaN         NaN          NaN     NaN;
    1/4    1/4       NaN         NaN         NaN          NaN     NaN;
    3/8    3/32      9/32        NaN         NaN          NaN     NaN;
    12/13  1932/2197 -7200/2197  7296/2197   NaN          NaN     NaN;
    1      439/216   -8          3680/513    -845/4104    NaN     NaN;
    1/2    -8/27     2           -3544/2565  1859/4104    -11/40  NaN;
    NaN    16/135    0           6656/12825  28561/56430  -9/50   2/55
    NaN    25/216    0           1408/2565   2197/4104    -1/5    0;
]
