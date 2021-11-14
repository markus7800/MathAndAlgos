

using JSON
include("edit_distance.jl")
include("GA.jl")


problem = JSON.parsefile("Optimization/MedianString/p1_15_20-4.txt")


alphabet = problem["alphabet"]
strs = Vector{String}(problem["strings"])
problem["str_length"]


Random.seed!(0)
@time best = GeneticOptimization(alphabet, strs,
    N_pop=1000, max_length=30, n_max=250,
    cre_p=0.25, dupl_p=0.25, mut_p=0.25, cross_p=0.25)


sum(edit_distance(best, s) for s in strs)
length(best)


using BenchmarkTools

@btime sum(edit_distance(best, s) for s in strs)
