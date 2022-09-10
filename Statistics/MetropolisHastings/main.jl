
include("MetropolisND.jl")

using Plots
using DataFrames
import CSV


using LinearAlgebra
function distance_to_polyline(p, line)
    d = Inf

    for i in 1:size(line,1)-1
        startp = line[i,:]
        endp = line[i+1,:]

        v = endp .- startp
        w = p .- startp

        s = v'w / (norm(v) * norm(v))
        proj = startp .+ v .* s

        if 0 ≤ s && s ≤ 1
            d = min(d,norm(proj .- p))
        else
            d = min(d, norm(p .- startp))
            d = min(d, norm(p .- endp))
        end
    end

    return d
end

distance_to_polyline([0,0.5], [0 2; 1 1; 3 2])

xs = LinRange(-1,3,100)
ys = LinRange(-1,3,100)
z = [distance_to_polyline([x,y], [0 2; 1 1; 2 2; 2 -0.5]) for y in ys, x in xs]

contour(xs, ys, z, ratio=:equal)



hello = CSV.read("Statistics/MetropolisHastings/hello.csv", DataFrame)

scatter(hello.x, -hello.y)
hello_array = hcat(hello.x, hello.y)


xs = LinRange(-50,400,250)
ys = LinRange(50,300,250)
z = [distance_to_polyline([x,y], hello_array) for y in ys, x in xs]

contour(xs, ys, exp.(-1/5 .* z), ratio=:equal, yflip=true, levels=20)

surface(xs, ys, exp.(-1/10 .* z), ratio=:equal, yflip=true, levels=20)

using Random
Random.seed!(0)
ps, acr = Metropolis_nD(x -> exp(-1/5 * distance_to_polyline(x, hello_array));
    n_iter=75_000, var=15., q_init=[0.,100.])

contour(xs, ys, z, ratio=:equal, yflip=true, levels=20)
plot!(ps[:,1], ps[:,2])

scatter(ps[:,1], ps[:,2], alpha=0.02, markerstrokecolor=1, markercolor=1, yflip=true, axis=false, ticks=nothing)

begin
    n = size(ps, 1)
    anim = Animation()
    nstep = 100
    trace = 1000

    println("frames: ", length(1:nstep:n))

    @showprogress for i in 1:nstep:n+trace
        tt = max(1, i-trace):min(i,n)
        p = scatter(ps[1:min(i,n),1], ps[1:min(i,n),2], alpha=0.02, markerstrokecolor=1, markercolor=1,
            xlims = (-50, 400), ylims = (50, 300), yflip=true, legend=false, axis=false, ticks=nothing)
        plot!(ps[tt, 1], ps[tt, 2], lc=:gray)
        frame(anim, p)
    end

    p = scatter(ps[:,1], ps[:,2], alpha=0.02, markerstrokecolor=1, markercolor=1,
        xlims = (-50, 400), ylims = (50, 300), yflip=true, legend=false, axis=false, ticks=nothing)
    for i in 1:100
        frame(anim, p)
    end

    gif(anim, "test.gif")
end

#=
trex = read("Statistics/MetropolisHastings/trex.obj", String)

collect(eachmatch(r"v  (.*)", trex))

trex_points = map(m -> parse.(Float64, split(m.match, " ")), eachmatch(r"(?<=v  )(.*)(?=\r)", trex))

trex_y = map(v -> v[1], trex_points)
trex_z = map(v -> v[2], trex_points)
trex_x = map(v -> v[3], trex_points)

using Plots

scatter(trex_x, trex_y, trex_z, ms=1, ratio=:equal)

trex_faces_ix = map(m -> map(p -> parse(Int, split(p, "/")[1]), split(m.match, " ")), eachmatch(r"(?<=f )(.*)(?= \r)", trex))

trex_faces_Δ_ix = Array{Array{Int64,1},1}()
for fix in trex_faces_ix
    if length(fix) == 3
        push!(trex_faces_Δ_ix, fix)
    elseif length(fix) == 4
        push!(trex_faces_Δ_ix, fix[1:3])
        push!(trex_faces_Δ_ix, fix[2:4])
    elseif length(fix) == 5
        push!(trex_faces_Δ_ix, fix[1:3])
        push!(trex_faces_Δ_ix, fix[2:4])
        push!(trex_faces_Δ_ix, fix[3:5])
    else
        error(":(")
    end
end

using LinearAlgebra

fix = trex_faces_ix[2]
A = hcat(trex_x[fix], trex_y[fix], trex_z[fix])
v1 = cross(A[2,:] - A[1,:], A[3,:] - A[1,:])
v2 = cross(A[3,:] - A[1,:], A[4,:] - A[1,:])

v1 ./ v2


(A[4,:]-A[1,:])'cross(A[2,:] - A[1,:], A[3,:] - A[1,:])

scatter(trex_x[fix], trex_y[fix], trex_z[fix])

maximum(length.(trex_faces_ix))

for fix in trex_faces_ix
    A = hcat(trex_x[fix], trex_y[fix], trex_z[fix])
    a = cross(A[2,:] - A[1,:], A[3,:] - A[1,:])
    a = a / norm(a)
    p = A[1,:]

    for i in 4:length(fix)
        y = A[i,:]
        println(abs(y'a - p'a))
        #@assert (A[i,:] - A[1,:])'v == 0.0
    end
end
=#

using Meshes
using MeshIO
using FileIO

import GLMakie

mesh = load("Statistics/MetropolisHastings/trex.obj")


for t in mesh
    println(t)
end


t = mesh[1]

using LinearAlgebra
function closest_point(t, y)
    A = t.points[1]
    B = t.points[2]
    C = t.points[3]

    u = B - A
    v = C - A

    n = cross(u,v)
    n /= norm(n)

    d = A'n

    x = y - (y'n - d)*n

    x - A

    M = hcat(u,v)
    c = M \ (x-A)

    c = clamp.(c, 0, 1)

    return M*c + A
end


y = Float32[1.,2.,3.]
closest_point(t, y)


tpoints = zeros(Float64, 3, length(mesh)*3)
for (i,t) in enumerate(mesh)
    tpoints[:,3*(i-1) + 1] = t.points[1]
    tpoints[:,3*(i-1) + 2] = t.points[2]
    tpoints[:,3*(i-1) + 3] = t.points[3]
end

points = zeros(Float64, 3, length(mesh.position))
for (i,point) in enumerate(mesh.position)
    skip = false
    for j in 1:i-1
        if sum((points[:,j] - point).^2) < 1e-9
            skip = true
            break
        end
    end
    if !skip
        points[:,i] = point
    end
end

unique_points = zeros(Float64, 3, count(any(points .!= 0, dims=1)))
begin
    j = 0
    for i in 1:length(mesh.position)
        if any(points[:,i] .!= 0)
            j += 1
            unique_points[:,j] = points[:,i]
        end
    end
end


tix = [Int[] for _ in 1:size(unique_points, 2)]
for i in 1:size(unique_points, 2)
    for j in 1:size(tpoints, 2)
        if sum((unique_points[:,i] - tpoints[:,j]).^2) < 1e-9
            push!(tix[i], j)
        end
    end
end

using Plots

scatter(points[3,:], points[1,:], points[2,:], ms=2)

minimum(points, dims=2)
maximum(points, dims=2)


function vertex_candidates(y, points, k=10)
    ds = reshape(sum((points .- y) .^2, dims=1), :)
    ix = (0:length(ds)-1) .+ 1# .÷ 3 .+ 1
    c = collect(zip(ix, ds))
    sort!(c, by=x->x[2])
    return map(x->x[1], c[1:k])
end

function vertex_candidates_2(y, points, k=10)
    ds = reshape(sum((points .- y) .^2, dims=1), :)

    candidates = zeros(Int, k)
    max_d = 0
    argmax_d = 0

    for j in 1:k
        candidates[j] = j
        if ds[j] > max_d
            max_d = ds[j]
            argmax_d = j
        end
    end

    for i in k+1:length(ds)
        if ds[i] < max_d
            candidates[argmax_d] = i
            max_d = 0
            for j in 1:k
                l = candidates[j]
                if ds[l] > max_d
                    max_d = ds[l]
                    argmax_d = j
                end
            end
        end
    end

    return candidates
end

@btime vertex_candidates(y, unique_points)
@btime vertex_candidates_2(y, unique_points)

ii = vertex_candidates(y, unique_points)
ii2 = vertex_candidates_2(y, unique_points)

fi = unique(reduce(vcat, tix[ii2]))

function trex_distance(y, mesh)
    d = Inf32
    for t in mesh
        x = closest_point(t, y)
        d = min(d, norm(x-y))
    end
    return d
end

@btime trex_distance(Float32[0,0,0], mesh) # 10ms

function trex_distance_2(y, mesh, points, tix, k=10)
    ii = vertex_candidates_2(y, points, k)

    fis = unique((reduce(vcat, tix[ii]) .- 1) .÷ 3 .+ 1)

    d = Inf
    for t in mesh[fis]
        x = closest_point(t, y)
        d = min(d, norm(x-y))
    end
    return d
end

@btime trex_distance(Float32[0,0,0], mesh) # 10ms
@btime trex_distance_2(Float32[0,0,0], mesh, unique_points, tix) # 140μs

testpoints = [[x,y,z] for x in LinRange(-400,400,100), y in LinRange(0,1200,100), z in LinRange(-1200,1200,100)]

testpoints = reshape(testpoints, :)

ds = Float64[]
ProgressMeter.@showprogress for t in testpoints
    push!(ds, trex_distance_2(t, mesh, unique_points, tix, 25))
end

sub_test_points = testpoints[ds .< 5]
test_x = map(x -> x[3], sub_test_points)
test_y = map(x -> x[1], sub_test_points)
test_z = map(x -> x[2], sub_test_points)
scatter(test_x, test_y, test_z, ms=1, ratio=:equal, camera=(20,60))

minimum(ds)
maximum(ds)

using Random
Random.seed!(0)
ps, acr = Metropolis_nD(x -> exp(-1/10 * trex_distance_2(x, mesh, unique_points, tix, 25));
    n_iter=1_000_000, var=100., q_init=Vector{Float64}(mesh[1].points[1]))


#scatter(points[3,:], points[1,:], points[2,:], ms=2)
plot(ps[:,3], ps[:,1], ps[:,2], ratio=:equal, camera=(20,60), alpha=1)#, , markerstrokecolor=1, markercolor=1, axis=false, ticks=nothing)

scatter(ps[:,3], ps[:,1], ps[:,2], ratio=:equal, camera=(25,45), alpha=0.02, ms=1,markerstrokecolor=1, markercolor=1)#, , markerstrokecolor=1, markercolor=1, axis=false, ticks=nothing)

begin
    anim = Animation()
    ProgressMeter.@showprogress for i in 0:360
        # p = scatter(unique_points[:,3], unique_points[:,1], unique_points[:,2], ratio=:equal, camera=(i,45), alpha=1, ms=1,markerstrokecolor=1, markercolor=1)#, , markerstrokecolor=1, markercolor=1, axis=false, ticks=nothing)

        si = sin(2π * i/360)
        ci = cos(2π * i/360)
        R = [
            ci -si 0;
            si ci 0;
            0 0 1
        ]
        P = R * unique_points[[3,1,2],:]
        p = scatter(P[1,:], P[2,:], P[3,:], ratio=:equal, camera=(20,45),
            alpha=1, ms=3,markerstrokecolor=1, markercolor=1,
            xlims=(-1200,1200), ylims=(-600,800), zlims=(0,1200), legend=false, axis=false, ticks=nothing)
        frame(anim, p)
    end
    gif(anim, "trex.gif")
end

scatter(unique_points[3,:], unique_points[1,:], unique_points[2,:], ratio=:equal, camera=(20,45), alpha=1, ms=3,markerstrokecolor=1, markercolor=1,
    xlims=(-1200,1200), ylims=(-600,800), zlims=(0,1200), legend=false)#, , markerstrokecolor=1, markercolor=1, axis=false, ticks=nothing)



begin
    anim = Animation()
    ProgressMeter.@showprogress for i in 0:360
        si = sin(2π * i/360)
        ci = cos(2π * i/360)
        R = [
            ci -si 0;
            si ci 0;
            0 0 1
        ]
        # P = R * unique_points[[3,1,2],:]
        P = R * adjoint(ps)[[3,1,2],1:min(5_000 * i, n)]
        p = scatter(P[1,:], P[2,:], P[3,:], ratio=:equal, camera=(20,45),
            alpha=0.02, ms=1,markerstrokecolor=1, markercolor=1,
            xlims=(-1200,1200), ylims=(-600,800), zlims=(0,1200), legend=false, axis=false, ticks=nothing)

        n = size(ps,1)
        range = max(1, 5_000 * (i-5)):100:min(5_000 * i, n)
        L = R * adjoint(ps)[[3,1,2], range]
        plot!(L[1,:], L[2,:], L[3,:], lc=:gray)
        frame(anim, p)
    end
    gif(anim, "trex.gif")
end
