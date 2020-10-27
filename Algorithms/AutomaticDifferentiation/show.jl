
import Base.show

function dim_str(ns::Tuple)
    join(string.(ns), "×")
end

short_str(v::DVal) = "val"
short_str(v::DVec) = "$(length(v.s))-d vec"
short_str(v::DMat) = "$(dim_str(size(v.s)))-d mat"


function show(io::IO, v::DType)
    n = length(v.s)
    print(io, short_str(v) * ", prev=$(short_str.(v.prev)), op=$(v.op)")
end


function print_tree(v::DType, tab="")
    s = tab * short_str(v) * " <- $(v.op)"
    if length(v.prev) == 0
        s *= "VAR"
    end
    println(s)
    for p in v.prev
        print_tree(p, tab*"|  ")
    end
end
