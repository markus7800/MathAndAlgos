
import Base.+
function +(self::DMat, other::DMat)
    res = DMat(self.s + other.s, prev=[self, other], op="+")
    res.backward = function bw()
        self.∇ .+= res.∇
        other.∇ .+= res.∇
    end
    return res
end

import Base.*
function *(self::DMat, other::DMat)
    res = DMat(self.s*other.s, prev=[self, other], op="⋅")
    res.backward = function bw()
        self.∇ .+= adjoint(other.s) .* res.∇
        other.∇ .+= adjoint(self.s) .* res.∇
    end
    return res
end
*(self::DMat, other::DVec) = self * DMat(other)
*(self::DVec, other::DMat) = DMat(self) * other

v = DMat(DVec([1., 2., 3.]))
w = DMat(DVec([2., -3., 1.]))
x = DMat(Matrix([2., -2., 5.]'))

u = x*(v + w)

backward(u)
v.∇
@assert all(v.∇ .== x.s')



A = DMat([1. 2.; 3. 4.])
x = DVec([1., 2.])
b = DVec([-3., -4.])

v = A*x+b
backward(v)

A.∇

v = A*x+b
r = v'v
backward(r)

(A.s')*(A.s*x.s + b.s)


x.∇
