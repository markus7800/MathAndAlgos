using Plots
using Random
include("perceptron.jl")

Random.seed!(1)
N = 500
x1 = rand(N)
x2 = rand(N)

X = hcat(x1,x2)
y = Int.(sign.(x1 .+ x2 .- 0.7))
w_true = [-0.7, 1, 1]
all(classify(X, w_true) == y)

scatter(x1, x2, mc=y, ms=3, markerstrokecolor=y, legend=false)
plot!(t -> -(w_true[2]*t + w_true[1]) / w_true[3])
xlims!((minimum(x1)-0.05,maximum(x1)+0.05))
ylims!((minimum(x2)-0.05,maximum(x2)+0.05))

Random.seed!(1)
w, ws = perceptron(X, y, w_true = w_true)

function make_anim(x1, x2, y, ws; fps=10)
    anim = Animation()
    x_range = (minimum(x1)-0.05, maximum(x1)+0.05)
    y_range = (minimum(x2)-0.05, maximum(x2)+0.05)
    @progress for (k,w) in enumerate(ws)
        p = scatter(x1, x2, mc=y, ms=4, markerstrokecolor=y, legend=false)
        plot!(t -> -(w[2]*t + w[1]) / w[3])
        xlims!(x_range)
        ylims!(y_range)
        title!("Iteration: $k")
        frame(anim, p)
    end
    @progress for f in 1:fps*3
        k = length(ws)
        w = ws[k]
        p = scatter(x1, x2, mc=y, markerstrokecolor=y, ms=4, legend=false)
        plot!(t -> -(w[2]*t + w[1]) / w[3])
        xlims!(x_range)
        ylims!(y_range)
        title!("Finished in $k iterations")
        frame(anim, p)
    end
    return anim
end

anim = make_anim(x1, x2, y, ws; fps=10)
gif(anim, "perceptron.gif", fps=10)





Random.seed!(1)
w = rand(3)
wrong_1 = y .* (dX * w) .<= 0
wrong_2 = classify(X, w) .!= y
wrong_1 == wrong_2
