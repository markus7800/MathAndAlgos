import Base.+
function +(self::DVec, other::DVec)
    res = DVec(self.s + other.s, prev=[self, other], op="+")
    res.backward = function bw(∇)
        self.∇ .+= ∇
        other.∇ .+= ∇
    end
    return demote(res)
end

@D (a,b) -> a*b (a,b) -> [b, a]
macro D(X::Expr, Y::Expr)

    quote
        $(esc(X.args))
        $(esc(Y))
    end
end


@macroexpand  D(begin a*b end,  begin [b, a] end)



@macroexpand @D function f(a::DVal,b::DVal)
    a*b
end function ∇(a,b)
    b, a
end

@macroexpand @D function f(a::DVal,b::DVal)
    a*b
end function ∇(a,b)
    b, a
end
