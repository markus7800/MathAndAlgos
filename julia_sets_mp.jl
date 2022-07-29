
using Distributed
addprocs(4)

const CURVE_START = 48/64* π
const CURVE_END = 60/64* π
const CURVE_SPAN = CURVE_END-CURVE_START
const CURVE_SCALE = 0.755

function c_from_group(group_size::Int, group_number::Int)
    if group_size == 1
        num_groups = 20
        phi = 2*π-CURVE_END + group_number/(num_groups-1)*CURVE_SPAN
    elseif group_size == 2
        num_groups = 30
        phi = CURVE_END - group_number/(num_groups-1)*CURVE_SPAN
    end

    return CURVE_SCALE*exp(phi*1im)
end

function compute_julia_set_sequential(xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64, im_width::Int, im_height::Int, c::ComplexF64)
    zabs_max = 10
    nit_max = 300

    xwidth  = xmax - xmin
    yheight = ymax - ymin

    julia = zeros(im_width, im_height)
    for ix in 0:(im_width-1)
        for iy in 0:(im_height)-1
            nit = 0
            # Map pixel position to a point in the complex plane
            z = (ix / im_width * xwidth + xmin) + im*(iy / im_height * yheight + ymin)
            # Do the iterations
            while abs(z) <= zabs_max && nit < nit_max
                z = z^2 + c
                nit += 1
            end
            ratio = nit / nit_max
            julia[ix+1,iy+1] = ratio
        end
    end
    return julia
end

@everywhere function compute_julia_patch(args::Tuple)
    x, y, patch, xmin, xmax, ymin, ymax, size, c = args

    im_width, im_height = size, size

    zabs_max = 10
    nit_max = 300

    xwidth  = xmax - xmin
    yheight = ymax - ymin

    patch_width = min(x+patch, im_width) - x # handle smaller patches at edges
    patch_height = min(y+patch, im_height) - y # handle smaller patches at edges
    julia_patch = zeros(patch_width, patch_height)


    for ix in 0:(patch_width-1)
        for iy in 0:(patch_height-1)
            # ix, iy are position in patch
            global_ix = ix + x # add x to get global position in image
            global_iy = iy + y # add y to get global position in image

            nit = 0
            # Map pixel position to a point in the complex plane
            z = (global_ix / im_width * xwidth + xmin) + im*(global_iy / im_height * yheight + ymin)
            # Do the iterations
            while abs(z) <= zabs_max && nit < nit_max
                z = z^2 + c
                nit += 1
            end
            ratio = nit / nit_max

            julia_patch[ix+1,iy+1] = ratio
        end
    end
    return (x, y, julia_patch)
end
@everywhere function do_work(jobs, results) # define work function everywhere
    while true
        args = take!(jobs)
        res = compute_julia_patch(args)
        put!(results, res)
    end
end

function compute_julia_in_parallel_wp(size::Int, xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64, patch::Int, c::ComplexF64)
    # put patches in task_list

    N_tasks = 0
    for x in 0:patch:(size-1)
        for y in 0:patch:(size-1)
            N_tasks += 1
        end
    end

    jobs = RemoteChannel(()->Channel{Tuple}(N_tasks));
    results = RemoteChannel(()->Channel{Tuple}(N_tasks))


    N_tasks = 0
    begin
        for x in 0:patch:(size-1)
            for y in 0:patch:(size-1)
                put!(jobs, (x, y, patch, xmin, xmax, ymin, ymax, size, c))
                #println("Job $N_tasks: julia patch for $x $y.")
                N_tasks += 1
            end
        end
    end

    begin
        for p in workers() # start tasks on the workers to process requests in parallel
            remote_do(do_work, p, jobs, results)
        end
    end

    # stitch together full image
    im_width, im_height = size, size
    julia_img = zeros(im_width, im_height)
    while N_tasks > 0
        N_tasks -= 1
        x, y, julia_patch = take!(results)
        patch_width = min(x+patch, im_width) - x # handle smaller patches at edges
        patch_height = min(y+patch, im_height) - y # handle smaller patches at edges
        julia_img[(x+1):(x+patch_width),(y+1):(y+patch_height)] .= julia_patch
        #println("Result $N_tasks: julia patch for $x $y.")
    end
    return julia_img
end

function compute_julia_in_parallel_threaded(xmin::Float64, xmax::Float64, ymin::Float64, ymax::Float64, im_width::Int, im_height::Int, c::ComplexF64)
    zabs_max = 10
    nit_max = 300

    xwidth  = xmax - xmin
    yheight = ymax - ymin

    julia = zeros(im_width, im_height)
    Threads.@threads for ix in 0:(im_width-1)
        for iy in 0:(im_height)-1
            nit = 0
            # Map pixel position to a point in the complex plane
            z = (ix / im_width * xwidth + xmin) + im*(iy / im_height * yheight + ymin)
            # Do the iterations
            while abs(z) <= zabs_max && nit < nit_max
                z = z^2 + c
                nit += 1
            end
            ratio = nit / nit_max
            julia[ix+1,iy+1] = ratio
        end
    end
    return julia
end

using Plots
using BenchmarkTools

function main(;size=500, showplot=false, bench=false, kind="seq", patch=32)
    xmin = -1.5
    xmax = 1.5
    ymin = -1.5
    ymax = 1.5

    c = c_from_group(1, 2)

    #julia = nothing
    if kind == "seq"
        julia = compute_julia_set_sequential(xmin, xmax, ymin, ymax, size, size, c)
        if bench
            @btime compute_julia_set_sequential($xmin, $xmax, $ymin, $ymax, $size, $size, $c)
        end
    elseif kind == "wp"
        julia = compute_julia_in_parallel_wp(size, xmin, xmax, ymin, ymax, patch, c)
        if bench
            @btime compute_julia_in_parallel_wp($size, $xmin, $xmax, $ymin, $ymax, $patch, $c)
        end
    elseif kind == "th"
        julia = compute_julia_in_parallel_threaded(xmin, xmax, ymin, ymax, size, size, c)
        if bench
            @btime compute_julia_in_parallel_threaded($xmin, $xmax, $ymin, $ymax, $size, $size, $c)
        end
    end

    if showplot
        p = heatmap(julia, legend=false, aspect_ratio=:equal, axis=nothing, border=:none)
        display(p)
    end

end

main(
    size=1500,
    showplot=false,
    bench=true,
    kind="seq"
) # 650ms


main(
    size=1500,
    showplot=false,
    bench=true,
    kind="th"
) # 275ms 4 threads
# 210 ms 8 threads
main(
    size=1500,
    showplot=false,
    bench=true,
    patch=750,
    kind="wp"
) #250ms 4 processes


addprocs(4) # 8
main(
    size=1500,
    showplot=false,
    bench=true,
    patch=100,
    kind="wp"
) # 204ms 8 processes
