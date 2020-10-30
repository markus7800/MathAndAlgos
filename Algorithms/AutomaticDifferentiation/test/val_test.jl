
@testset "Value" begin
    a = DVal(10.)
    b = DVal(-5.)
    c = DVal(2.)
    d = DVal(-0.5)

    e = c * (a + d * b)
    backward(e)
    @test a.∇ == c.s && b.∇ == c.s*d.s

    a = DVal(10.)
    r = a + a + a*a
    backward(r)
    @test a.∇ == 2 + 2*a.s


    a = DVal(2.)
    r = log(exp(a))
    backward(r)
    @test r.∇ == 1

    using Random
    Random.seed!(1)
    _A = rand(10,10)
    _B = rand(10,10)

    A = DVal.(_A)
    B = DVal.(_B)


    C = A * B
    _C = _A * _B
    @test all(Float64.(C) .≈ _C)


    # constants

    a = DVal(5.)
    r = -3 * a
    backward(r)
    @test a.∇ == -3

    a = DVal(5.)
    r = a + 3.
    backward(r)
    @test a.∇ == 1
end
