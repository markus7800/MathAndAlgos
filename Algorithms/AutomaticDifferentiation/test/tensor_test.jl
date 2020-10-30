@testset "Tensor" begin
    @testset "+" begin
        Random.seed!(1)

        _A = rand(2,3,3)
        _b = reshape(rand(3),1,:,1)
        _c = reshape(rand(2),:,1,1)
        _d = reshape(rand(3),1,1,:)
        _e = rand(2,3)


        A = DTensor(_A)
        b = DTensor(_b)

        r = sum(A + b)
        backward(r)
        A∇, b∇ = Flux.gradient((_A,_b) -> sum(_A .+ _b), _A, _b)
        @test A.∇ == A∇ && b.∇ == b∇


        A = DTensor(_A)
        c = DTensor(_c)

        r = sum(A + c)
        backward(r)
        A∇, c∇ = Flux.gradient((_A,_c) -> sum(_A .+ _c), _A, _c)
        @test A.∇ == A∇ && c.∇ == c∇


        A = DTensor(_A)
        d = DTensor(_d)

        r = sum(A + d)
        backward(r)
        A∇, d∇ = Flux.gradient((_A,_d) -> sum(_A .+ _d), _A, _d)
        @test A.∇ == A∇ && d.∇ == d∇


        A = DTensor(_A)
        e = DTensor(_e)

        r = sum(A + e)
        backward(r)
        A∇, e∇ = Flux.gradient((_A,_e) -> sum(_A .+ _e), _A, _e)
        @test A.∇ == A∇ && e.∇ == e∇
    end
    @testset "*" begin
        Random.seed!(1)

        _A = rand(2,3,3)
        _b = reshape(rand(3),1,:,1)
        _c = reshape(rand(2),:,1,1)
        _d = reshape(rand(3),1,1,:)
        _e = rand(2,3)

        A = DTensor(_A)
        b = DTensor(_b)

        r = sum(A * b)
        backward(r)
        A∇, b∇ = Flux.gradient((_A,_b) -> sum(_A .* _b), _A, _b)
        @test A.∇ == A∇ && b.∇ == b∇


        A = DTensor(_A)
        c = DTensor(_c)

        r = sum(A * c)
        backward(r)
        A∇, c∇ = Flux.gradient((_A,_c) -> sum(_A .* _c), _A, _c)
        @test A.∇ == A∇ && c.∇ == c∇


        A = DTensor(_A)
        d = DTensor(_d)

        r = sum(A * d)
        backward(r)
        A∇, d∇ = Flux.gradient((_A,_d) -> sum(_A .* _d), _A, _d)
        @test A.∇ == A∇ &&d.∇ == d∇


        A = DTensor(_A)
        e = DTensor(_e)

        r = sum(A * e)
        backward(r)
        A∇, e∇ = Flux.gradient((_A,_e) -> sum(_A .* _e), _A, _e)
        @test A.∇ == A∇ && e.∇ == e∇
    end
end
