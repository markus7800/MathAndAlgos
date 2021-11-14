include("edit_distance.jl")

function brute(alphabet::String, strings::Vector{String}, max_length::Int, i::Int, current::String)::Tuple{String, Int}
    if i > max_length
        return
    end

    best = current
    score = sum(edit_distance(current, s) for s in strings)

    for char in alphabet
        new = current * char
        new_score = sum(edit_distance(new, s) for s in strings)
        if new_score < score
            best = new
            score = new_score
        end

        if i < max_length
            new, new_score = brute(alphabet, strings, max_length, i+1, new)
            if new_score < score
                best = new
                score = new_score
            end
        end
    end

    return best, score
end

import JSON
using DataFrames
function test(path; max_sol_length=10)
    results = DataFrame(file=String[], best=String[], score=Int[], time=Float64[], search_dim=Int[], max_n_char=Int[], n_str=Int[])

    @progress for file in readdir(path)
        if occursin(".txt", file)
            problem = JSON.parsefile(path * "/" * file)
            abc = problem["alphabet"]
            strs = Vector{String}(problem["strings"])
            n_char = maximum(problem["str_length"])

            println(file)
            println(strs)
            sol_length = min(n_char, max_sol_length)
            v,t, = @timed brute(abc, strs, sol_length, 1, "")
            best, score = v
            println("Found $best with score $score in $t seconds with max string length $n_char and search space dimension $sol_length")
            push!(results, [file, best, score, t, sol_length, n_char, length(strs)])
        end
    end
    return results
end

res1 = test("Optimization/MedianString/instances/p1", max_sol_length = 10)

res2 = test("Optimization/MedianString/instances/p2", max_sol_length = 10)

append!(res1, res2)

import CSV

CSV.write("results10.csv", res1)


function rand_string(alphabet, n)
    return reduce(*, rand(alphabet, n))
end

function find_contradiction(n_iter)
    abc = "ACGT"
    for _ in 1:n_iter
        A = rand_string(abc, 3)
        B = rand_string(abc, 3)
        C = rand_string(abc, 3)

        best3, score3 = brute(abc, [A,B,C], 3, 1, "")
        best4, score4 = brute(abc, [A,B,C], 4, 1, "")

        if score4 < score3
            println([A,B,C])
            println("$best3 with score $score3")
            println("$best4 with score $score4")
            break
        end
    end
end

using Random
Random.seed!(0)
find_contradiction(1000)

function get_loss(strs::Vector{String})
    return function L(str::String)::Int
        sum(edit_distance(str, s) for s in strs)
    end
end

get_loss(["GGT", "GGC", "CTT"])("GCT")
get_loss(["GGT", "GGC", "CTT"])("GGCT")

edit_distance("CCT", "GCT")
edit_distance("GGC", "GCT")
edit_distance("GGT", "GCT")
