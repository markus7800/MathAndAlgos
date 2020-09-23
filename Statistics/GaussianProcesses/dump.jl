
import GaussianProcesses

me2 = GaussianProcesses.MeanZero()
se2 = GaussianProcesses.SE(0.,0.)
gp2 = GaussianProcesses.GP(xtrain, ytrain, me2, se2, -Inf)

plot(gp2,xlims=(-3.,3), legend=false)
f = rand(gp2, xs, 10)
plot!(xs, f)

m3, K3 = GaussianProcesses.predict_f(gp2, xs, full_cov=true)

plot(xs, m)
plot!(xs, m2)
plot!(xs, m3)
plot!(x->gp_pos.mean.f([x])[1])

make_posdef!(K3)

me2 = GaussianProcesses.MeanZero()
se2 = GaussianProcesses.SE(0.,0.)
gp2 = GaussianProcesses.GP(xtrain, ytrain, me2, se2, -Inf)

xs = collect(LinRange(-3,3,100))

Kpred = cov(gp.kernel, xs', xs') # Kxx
Kcross = cov(gp.kernel, xtrain', xs') # Kxf
Ktrain = cov(gp.kernel, xtrain', xtrain') # Kff, cK

import PDMats
Lck = PDMats.whiten(PDMats.PDMat(Ktrain), Kcross) # Lck
U = cholesky(Ktrain).U
sum(U'Lck .- Kcross) # U' * Lck = Kcross
sum(U'U .- Ktrain) # U'U == Ktrain
sum(Lck'Lck .- Kcross'inv(Ktrain)*Kcross)

m, K = predict(gp, xs', xtrain', ytrain', σ=0.0)
K2 = Kpred .- Kcross'inv(Ktrain)*Kcross
μ, K3 = GaussianProcesses.predict_f(gp2, xs, full_cov=true)
K4 = Kpred .- Lck'Lck

sum(abs.(K2 .- K))
sum(abs.(K3 .- K))
sum(abs.(K3 .- K2))
sum(abs.(K4 .- K))
sum(abs.(K4 .- K3))

sum(abs.(m .- μ))

make_posdef!(K)
make_posdef!(K2)
make_posdef!(K3)
make_posdef!(K4)





Σ = Float64[1 2 0;
     2 3 2;
     0 2 4;]

Σ = Σ'Σ
isposdef(Σ)

Random.seed!(1)
Z = randn(3, 10000)

import Statistics
Statistics.cov(Z')

Y = unwhiten(Σ, Z)
Y2 = PDMats.unwhiten(PDMats.PDMat(Σ), Z)
Statistics.cov(Y2')

Y3 = PDMats.whiten(PDMats.PDMat(Σ), Y2)
