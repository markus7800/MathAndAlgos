
import Base.show

function dim_str(ns::Tuple)
    # if sum(ns) == 1
    #     "0"
    # else
    join(string.(ns), "×")
end

short_str(v::DVal) = "val"
short_str(v::DVec) = "$(length(v.s))-d vec"
short_str(v::DMat) = "$(dim_str(size(v.s)))-d mat"
short_str(v::DTensor) = "$(dim_str(size(v.s)))-d tensor"


function show(io::IO, v::DType)
    n = length(v.s)
    l = length(v.prev)
    if l == 0
        print(io, short_str(v))
    else
        if l <= 3
            prev_str = "prev=$(short_str.(v.prev))"
        else
            prev_str = "$l prev"
        end
        print(io, short_str(v) * ", "* prev_str * ", op=$(v.op)")
    end
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
