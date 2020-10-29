# conversion
function DMat(V::Matrix{DVal})
    res = DMat(map(d -> d.s, V), prev=vec(V), op="mat<-[val]")
    res.backward = function bw(∇)
        for (d, g) in zip(res.prev, vec(∇)) # prev stored in vectorised form
            d.∇ += g # just pass down gradients
        end
    end
    return res
end

import Base.+
function +(self::DMat, other::DMat)
    res = DMat(self.s + other.s, prev=[self, other], op="+")
    res.backward = function bw(∇)
        self.∇ .+= ∇
        other.∇ .+= ∇
    end
    return res
end

function +(self::DMat, A::Matrix{T}) where T <: Number
    res = DMat(self.s + A, prev=[self], op="+ C")
    res.backward = function bw(∇)
        self.∇ .+= ∇
    end
    return res
end

+(A::Matrix{T}, other::DMat) where T <: Number = other + A

import Base.-
-(self::DMat) = (-1) * self

-(self::DMat, other::DMat) = self + (-other)
-(self::Matrix{T}, other::DMat) where T <: Number = self + (-other)
-(self::DMat, other::Matrix{T}) where T <: Number = self + (-other)

import Base.*
function *(self::DMat, other::DMat)
    res = DMat(self.s*other.s, prev=[self, other], op="*")
    res.backward = function bw(∇)
        #=
        S = self, O = other S*O = R
        each value of Rij corresponds to a vector vector product Si:⋅O:j
        and has gradient ∇Rij.

        A row Si: contributes to a row of gradients (scalar) in ∇R
        A column O:j contributes to a column of gradients(scalar) in ∇R

        Thus the gradients are
        ∇Si: = ∇Ri1 O1:^T + ∇Ri2 O2:^T + ... = (∇R O^T)i:
        ∇O:j = S1:^T ∇R1j + S2:^T ∇R2j + ... = (S^T ∇R):j

        =#
        self.∇ .+=  ∇ * adjoint(other.s)
        other.∇ .+= adjoint(self.s) * ∇
    end
    return demote(res)
end

function *(self::DMat, A::Matrix{T}) where T <: Number
    res = DMat(self.s*A, prev=[self], op="* C")
    res.backward = function bw(∇)
        self.∇ .+=  ∇ * adjoint(A)
    end
    return demote(res)
end


function *(A::Matrix{T}, other::DMat) where T <: Number
    res = DMat(A*other.s, prev=[other], op="C *")
    res.backward = function bw(∇)
        other.∇ .+= adjoint(A) * ∇
    end
    return demote(res)
end

*(self::DMat, other::DVec) = self * DMat(other)

*(self::DMat, other::Vector{T}) where T <: Number = self * reshape(other,:,1)

*(self::Matrix{T}, other::DVec) where T <: Number = self * DMat(other)

function *(r::Number, other::DMat)
    res = DMat(r*other.s, prev=[other], op="$r *")
    res.backward = function bw(∇)
        other.∇ .+= r * ∇
    end
    return demote(res)
end

*(self::DMat, r::Number) = r * self

v = DMat(DVec([1., 2., 3.]))
w = DMat(DVec([2., -3., 1.]))
x = DMat(Matrix([2., -2., 5.]'))

u = x*(v + w)

backward(u)
v.∇
@assert all(v.∇ .== x.s')



_A = [1. 2.; 3. 4.]
_x = [1., 2.]
_b = [-3., -4.]

A = DMat(_A)
x = DVec(_x)
b = DVec(_b)

v = A*x
r = v⋅v
backward(r)

@assert x.∇ == 2*(A.s')*(A.s*x.s)

2*(A.s')*(A.s*x.s)

x.∇



A = DVal.(_A)
x = DVal.(_x)
b = DVal.(_b)

v = A*x + b
r = reshape(v,1,:)*v
backward(r[1])
@assert map(d -> d.∇, x) == 2*(_A')*(_A*_x+_b)


A = DMat(_A)
x = DVec(_x)
b = DVec(_b)

v = A*x + b
r = v⋅v
backward(r)

@assert x.∇ == 2*(_A')*(_A*_x+_b)


v = DVec([1., 2., 3.])
r = v⋅v
backward(r)

@assert v.∇ == 2*v.s

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



begin
    A∇ = zeros(size(_A))
    B∇ = zeros(size(_B))
    m, n = size(_∇)
    for i in 1:m, j in 1:n
        local a = DVec(_A[i,:])
        local b = DVec(_B[:,j])
        local r = a ⋅ b
        backward(r)
        A∇[i,:] .+= a.∇ * _∇[i,j]
        B∇[:,j] .+= b.∇ * _∇[i,j]
    end

    @assert all(A∇ .≈ A.∇) && all(B∇ .≈ B.∇)
end


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


@assert x.∇ == _D * _x

A = DMat(_A)

_A - A
