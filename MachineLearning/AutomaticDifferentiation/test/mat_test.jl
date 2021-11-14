@testset "Matrix" begin

    v = DMat(DVec([1., 2., 3.]))
    w = DMat(DVec([2., -3., 1.]))
    x = DMat(Matrix([2., -2., 5.]'))

    u = x*(v + w)

    backward(u)
    v.∇
    @test all(v.∇ .== x.s')



    _A = [1. 2.; 3. 4.]
    _x = [1., 2.]
    _b = [-3., -4.]

    A = DMat(_A)
    x = DVec(_x)
    b = DVec(_b)

    v = A*x
    r = v⋅v
    backward(r)

    @test x.∇ == 2*(A.s')*(A.s*x.s)

    2*(A.s')*(A.s*x.s)

    x.∇



    A = DVal.(_A)
    x = DVal.(_x)
    b = DVal.(_b)

    v = A*x + b
    r = reshape(v,1,:)*v
    backward(r[1])
    @test map(d -> d.∇, x) == 2*(_A')*(_A*_x+_b)


    A = DMat(_A)
    x = DVec(_x)
    b = DVec(_b)

    v = A*x + b
    r = v⋅v
    backward(r)

    @test x.∇ == 2*(_A')*(_A*_x+_b)


    v = DVec([1., 2., 3.])
    r = v⋅v
    backward(r)

    @test v.∇ == 2*v.s

    Random.seed!(1)
    _A = rand(2,3)
    _B = rand(3,2)
    _∇ = rand(2,2)

    A = DMat(_A)
    B = DMat(_B)

    r = A*B

    r.backward(_∇)

    A.∇



    B.∇




    A∇ = zeros(size(_A))
    B∇ = zeros(size(_B))
    m, n = size(_∇)
    for i in 1:m, j in 1:n
        a = DVec(_A[i,:])
        b = DVec(_B[:,j])
        r = a ⋅ b
        backward(r)
        A∇[i,:] .+= a.∇ * _∇[i,j]
        B∇[:,j] .+= b.∇ * _∇[i,j]
    end

    @test all(A∇ .≈ A.∇) && all(B∇ .≈ B.∇)



    # constants

    _A = [1. 2.; -3. 4]
    # _A = _A'_A

    _x = [-1., 1.]

    A = DMat(_A)
    x = DVec(_x)

    r = (x)⋅(A*x)
    backward(r)

    x.∇

    _D = [
        2*_A[1,1] _A[1,2]+_A[2,1];
        _A[1,2]+_A[2,1] 2*_A[2,2]
    ]


    @test x.∇ == _D * _x

end
