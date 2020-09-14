using ProgressMeter
using Plots
using LinearAlgebra
using Random


function mouse_trajectory(n::Int, k::Int)
    v = [exp(2π*im/n*k*(j-1)) for j in 1:n]
    return t -> v * exp((exp(2π*im/n*k) - 1)*t)
end

function ∇mouse_trajectory(n::Int, k::Int)
    mouse_traj = mouse_trajectory(n, k)
    return t -> (exp(2π*im/n*k) - 1) * mouse_traj(t)
end

function eigen_values(n::Int)
    return [exp(2π*im/n*k) - 1 for k in 1:n]
end

function eigen_vectors(n::Int)
    return [exp(2π*im/n*k*(j-1)) for j in 1:n, k in 1:n]
end

function diff_eq_matrix(n::Int)
    A = Matrix{Float64}(-I(n))
    A[n,1] = 1
    for j in 2:n
        A[j-1,j] = 1
    end
    return A
end

norm(inv(eigen_vectors(5)) * diff_eq_matrix(5) * eigen_vectors(5) - Diagonal(eigen_values(5)))

function plot_mice(n::Int, T::Float64, cs::Vector{ComplexF64}=vcat(1. + 0im, fill(0im, n-1)))
    n_steps = 500
    ts = LinRange(0, T, n_steps)
    traj = zeros(ComplexF64, n, n_steps)

    for (k,c) in enumerate(cs)
        c == 0. && continue
        traj_func = mouse_trajectory(n, k)
        for (i,t) in enumerate(ts)
            traj[:,i] += c * traj_func(t)
        end
    end

    p = plot(size=(600,600), legend=false)
    for k in 1:n
        plot!(traj[k,:])
    end

    ps = zeros(ComplexF64, n)
    ∇ps = zeros(ComplexF64, n)
    for (k,c) in enumerate(cs)
        traj_func = mouse_trajectory(n, k)
        ∇traj_func = ∇mouse_trajectory(n, k)
        ps += c * traj_func(T)
        ∇ps += c * ∇traj_func(T)
    end

    scatter!(ps, mc=:red)
    quiver!(ps, quiver=(real.(∇ps), imag.(∇ps)), lc=:black)

    midpoint = cs[n]
    scatter!([midpoint], markershape=:x, mc=:black)
    return p
end

function calculate_coeffs(x0::Vector{ComplexF64})
    n = length(x0)
    V = eigen_vectors(n)
    cs = V \ x0
    return cs
end


function anim_mice(n::Int, cs::Vector{ComplexF64}=vcat(1. + 0im, fill(0im, n-1)); t1=5, frames=100)
    anim = Animation()
    ts = LinRange(0, t1, frames)

    @showprogress for t in ts
        p = plot_mice(n, t, cs)
        frame(anim, p)
    end

    return anim
end


plot_mice(3, 1.)

anim = anim_mice(9, t1=10., frames=300)
gif(anim, "9equilat.gif")

n = 9
Random.seed!(1)
x0 = rand(ComplexF64, n)
scatter(x0)
cs = calculate_coeffs(x0)
anim = anim_mice(n, cs, t1=10., frames=300)
gif(anim, "9random.gif")
