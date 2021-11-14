@testset "Vector" begin

    v = DVec([1., 2., 3.])
    w = DVec([2., -3., 1.])
    x = DVec([2., -2., 5.])

    u = x⋅(v + w)

    backward(u)
    @test v.∇ == x.s

    v = DVal.([1., 2., 3.])
    w = DVal.([2., -3., 1.])
    x = DVal.([2., -2., 5.]')

    u = x*(v + w)

    backward(u[1])
    map(d -> d.∇, v)
    @test map(d -> d.∇, v) == [2., -2., 5.]


    # constants

    v = DVec([1., 2., 4.])
    w = [-1., 2., -0.5]

    r = w ⋅ v
    backward(r)

    @test v.∇ == w

    v = DVec([1., 2., 4.])
    w = [-1., 2., -0.5]

    c = 2*(v.s + w)

    u = w + v
    r = u ⋅ u

    w = [1., 1., 1.]

    backward(r)


    @test v.∇ == c

    # other

    v = DVec([1., 2., -4.])
    r = sum(v*v)
    backward(r)
    v.∇

    w = DVec([1., 2., -4.])
    r = w ⋅ w
    backward(r)
    w.∇

    @test v.∇ == w.∇
end
