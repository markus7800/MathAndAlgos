@testset "Conversions" begin

    d = DVal(1π)
    @test DVec(d) isa DVec
    @test DMat(d) isa DMat

    d = DVec([1π, 2π])
    @test DVal(d) isa DVal
    @test DMat(d) isa DMat

    d = DMat([1π 2π; 3π 4π])
    @test DVal(d) isa DVal
    @test DVec(d) isa DVec



    d = DMat([1. 2.; 3. 4.])
    @test demote(d) isa DMat

    d = DVec([1., 2.])
    @test  demote(d) isa DVec

    d = DVal(1.)
    @test demote(d) isa DVal

    d = DMat([1. 2.;])
    @test demote(d) isa DVec

    d = DMat(Matrix(adjoint([1. 2.])))
    @test demote(d) isa DVec

    d = DMat(reshape([1.], 1, 1))
    @test demote(d) isa DVal

end
