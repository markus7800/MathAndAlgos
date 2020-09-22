import ProgressMeter
import Statistics
import StatsBase

#=
 1 dimensional Metropolis algorithm with
 Gaussian proposal function.
=#

function Metropolis_1D(prior::Function, likelihood::Function,
               M::Int, var::Float64,
               q_min::Float64, q_max::Float64,
               q_init::Float64; lag::Int=500)::Tuple

    local q = fill(NaN, M)
    q[1] = q_init
    accepted = 0

    ProgressMeter.@showprogress  1 "Iterations: " for n in 1:M-1

        # gaussian proposal qq ∼ N(q[n], var)
        local qq::Float64 = q[n] + sqrt(var) * randn(Float64)
        qq = clamp(qq, q_min, q_max)

        # proposal function is symmetric so it cancels in the fraction
        local A::Float64 = min(1, (likelihood(qq)   * prior(qq)) /
                                  (likelihood(q[n]) * prior(q[n])))

        if rand(Float64) ≤ A
            q[n+1] = qq
            accepted += 1
        else
            q[n+1] = q[n]
        end
    end

    # 14.17 acceptance ratio
    acceptance_ratio = accepted / (M-1)

    # 14.18 auto-correlation
    L = length(q)
    q_mean = sum(q) / L
    q_centered = q .- q_mean
    ACF = q_centered[1:L-lag]'q_centered[lag+1:L] / sum(q_centered.^2)

    return q, acceptance_ratio, ACF
end


#=
 Implementation of statistics calculation based on MCM-chain
 Based on the Markov-Chain a histogram is calculated and
 the bin with highest count corresponds to MAP and
 confidence interval around MAP is also calculated by using the bin counts.
=#

function MAP(q::Vector{Float64}; burn_in::Int, bin_width::Float64)
    q = q[burn_in:end] # discard burn-in period
    sort!(q)
    q_min = q[1]
    q_max = q[end]

    # calculate number of bins based on width
    n_bins = Int(ceil((q_max - q_min)/bin_width))

    bins = [q_min + bin_width * i for i in 0:n_bins-1]
    counts = zeros(n_bins)

    # calculate histogram by counting
    current = 1
    for v in q
        if current + 1 ≤ length(bins) &&  v > bins[current+1]
            current += 1
        end
        counts[current] += 1
    end

    # MAP = argmax_q π(q|d)
    map = argmax(counts) # index
    MAP = bins[map] + bin_width/2 # take midpoint of bin

    # calculate 95% confidence interval
    current = 0
    count = counts[map]
    while count / length(q) < 0.95
        current += 1
        # extend interval symmetrically
        count += counts[map-current] + counts[map+current]
    end
    QI_min = bins[map-current] # lower bound
    QI_max = bins[map+current+1] # upper bound

    return MAP, QI_min, QI_max, bins, counts
end
