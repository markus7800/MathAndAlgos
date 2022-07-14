


using Distributions
using LinearAlgebra
using StatsPlots
import SpecialFuntions

import Pkg
Pkg.add("https://github.com/JuliaMath/SpecialFunctions.jl")

n = 2
n_samples = 1_000_000

Z = rand(MvNormal(zeros(n), 1. * I(n)), n_samples)

R = vec(sqrt.(sum(Z.^2, dims=1)))


# ∫ f(x) dx = ∫ f ∘ T |detT| dy

# ∫ f(x) dx = ∫ f ∘ T(r, α, θ) r^{n-1}⋅cos(θ_1)⋅...⋅cos(θ_{n-1}) dr dα dθ

# p(r) = r^{n-1} ∫ f ∘ T(r, α, θ) cos(θ_1)⋅...⋅cos(θ_{n-1}) dα dθ
#      = r^{n-1} exp(-r^2/2) * C
#      = 1/ (2^{n/2 - 1} Γ(n/2)) r^{n-1} exp(-r^2/2)

histogram(R, normalize=true)
plot!(r -> 1/(2^(n/2-1) * gamma(n/2)) * r^(n-1) * exp(-r^2/2), lw=5)

p = plot(legend=false, xlims=(0,10));
for n in [1,2,3,4,5,10,15,20,30,40,50]
    plot!(r -> 1/(2^(n/2-1) * gamma(n/2)) * r^(n-1) * exp(-r^2/2));
end
display(p)

gamma(1)

Gamma()
