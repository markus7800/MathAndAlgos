
#=
 Consider a population dynamics problem, based on the lotka-volterra model:
 We have a fox population y and a rabbit population x.
 The discrete dynamics are given by four parameters α, β, γ, δ > 0:
   x_{t+1} = α * x_t - β * x_t * y_t
   y_{t+1} = γ * y_t + δ * x_t * y_t

 Assume that we know that foxes die off exponentially without prey
   y_{t+1} = 0.9 * y_t (γ = 0.9)
 Further, assume we understand the population dynamics of rabbits
 with predator interaction completely
   x_{t+1} = 1.1 * x_t - 0.15 * x_t * y_t (α = 1.1, β = 0.15)

 Now we like to find the population dynamics of foxes with prey interaction.
 We are interested in the parameter δ and we know that δ ∈ [0.05, 0.2]
=#

global N = 100
global α_exact = 1.1
global β_exact = 0.15
global γ_exact = 0.9
global δ_exact = 0.12
global δ_min = 0.05
global δ_max = 0.2
global σ = 0.05


function model(α::Float64, β::Float64, γ::Float64, δ::Float64)
    local fox_pop = [1.]
    local rabbit_pop = [0.5]

    for i in 1:N
        x = rabbit_pop[end]
        y = fox_pop[end]
        push!(rabbit_pop, α * x - β * x * y)
        push!(fox_pop, γ * y + δ * x * y)
    end

    return fox_pop, rabbit_pop
end


function prior(δ::Float64)::Float64
    return 1 / (δ_max - δ_min)
end

function likelihood(δ::Float64)::Float64
    local fox_pop_est, rabbit_pop_est = model(α_exact, β_exact, γ_exact, δ)
    local S = sum((fox_pop_est .- fox_data).^2) + sum((rabbit_pop_est .- rabbit_data).^2)

    return exp(-S / (2*σ^2))
end


using Random
Random.seed!(1)
global fox_pop, rabbit_pop = model(α_exact, β_exact, γ_exact, δ_exact)
global fox_data = fox_pop + σ * randn(N+1)
global rabbit_data = rabbit_pop + σ * randn(N+1)
