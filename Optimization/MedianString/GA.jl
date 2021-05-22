
using Random

# Helper
import Base.argmax
function argmax(f::Function, A::Array)
    fA = f.(A)
    i = Base.argmax(fA)
    return A[i]
end
import Base.argmin
function argmin(f::Function, A::Array)
    fA = f.(A)
    i = Base.argmin(fA)
    return A[i]
end

function rand_string(alphabet, n)
    return reduce(*, rand(alphabet, n))
end

# fitness is determined by the number of individuals
# a individual is prevailed by
function pareto_fitness(f_pop::Array{Int,1})
    N_pop = length(f_pop)
    fitness = zeros(Int, N_pop)
    for k in 1:N_pop
        fk = f_pop[k]
        n = sum(f_pop .< fk) # number of other individuals k is prevailed by
        fitness[k] = -n # take minus sign since fewer is better
    end
    return fitness
end

# In tournament selections we randomly select K individuals with replacement
# and put the one with best fitness in the mating pool
function tournament_select(fitness::Array{Int,1}, N::Int, K::Int=2)
    N_pop = length(fitness)
    mating_pool = zeros(Int, N_pop)
    for k in 1:N
        contender = rand(1:N_pop, K)
        mating_pool[k] = argmax(c -> fitness[c], contender)
    end
    return mating_pool
end

# reproduction consists of creation, duplication, mutation and crossovers
function reproduce(mating_pool::Array{Int,1}, pop::Vector{String}, N::Int,
    alphabet::String, max_length::Int,
    cre_p::Float64, dupl_p::Float64, mut_p::Float64, cross_p::Float64)

    # helper for selecting creation with probability cre_p,
    # duplication with probability dupl_p, ...
    p_tot = cre_p + dupl_p + mut_p + cross_p
    cre_F, dupl_F, mut_F, cross_F = cumsum([cre_p, dupl_p, mut_p, cross_p])

    next_pop = Vector{String}(undef, size(pop))
    for i in 1:N
        p = rand() * p_tot
        k = mating_pool[i]

        if p ≤ cre_F
            # create
            slength = rand(1:max_length)
            new = rand_string(alphabet, slength)
        elseif p ≤ dupl_F
            # duplicate
            new = pop[k]
        elseif p ≤ mut_F
            # mutate
            p2 = rand()
            template = pop[k]
            t = rand(1:length(template))
            if p2 ≤ 0.33
                # insert
                new = template[1:t] * rand(alphabet) * template[t:end]
            elseif p2 ≤ 0.66
                # substitute
                new = template[1:t-1] * rand(alphabet) * template[t:end]
            else
                # delete
                new = template[1:t-1] * template[t:end]
            end
        else
            # crossover
            a, b = mating_pool[rand(1:N,2)] # parents
            t1 = rand(1:length(a)) # crossover point
            t2 = rand(1:length(b))
            new = pop[a][1:t1] * pop[b][t2:end]
        end
        if length(new) > max_length
            new = new[1:max_length]
        end
        next_pop[i] = new
    end
    return next_pop
end

function GeneticOptimization(alphabet::String, strs::Vector{String}; N_pop::Int,
    max_length::Int, n_max::Int=10^2,
    cre_p::Float64, dupl_p::Float64, mut_p::Float64, cross_p::Float64)

    # creation
    population = [rand_string(alphabet, rand(1:max_length)) for i in 1:N_pop]
    #display(population)

    #pop_hist = []; fit_hist = []; mat_hist = []

    n = 1
    best = population[1]
    while n < n_max
        n += 1
        # evaluate  on population
        f_pop = [sum(edit_distance(s, population[k]) for s in strs) for k in 1:N_pop]
        # find best individual
        k = argmin(j -> f_pop[j], collect(1:N_pop))
        if sum(edit_distance(s, population[k]) for s in strs) < sum(edit_distance(s, best) for s in strs)
            best = population[k]
        end

        # evaluate fitness
        fitness = pareto_fitness(f_pop)
        mating_pool = tournament_select(fitness, N_pop, 2) # mating_pool are indices

        # push!(pop_hist, population)
        # push!(fit_hist, fitness)
        # push!(mat_hist, mating_pool)

        population = reproduce(
            mating_pool, population, N_pop, alphabet, max_length,
            cre_p, dupl_p, mut_p, cross_p
        )
    end

    return best#, pop_hist, fit_hist, mat_hist
end
