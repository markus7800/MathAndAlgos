using ProgressMeter

function Metropolis_nD(pdf::Function; # needs not be normalised
               n_iter::Int, var::Float64, q_init::Vector{Float64})::Tuple

    local d = length(q_init)

    local q = Array{Float64}(undef, n_iter, d)
    q[1,:] = q_init
    accepted = 0

    ProgressMeter.@showprogress  1 "Iterations: " for n in 1:n_iter-1

        # gaussian proposal qq ∼ N(q[n], var)
        local qq::Vector{Float64} = q[n,:] .+ sqrt(var) * randn(Float64, d)

        # proposal function is symmetric so it cancels in the fraction
        local A::Float64 = min(1, pdf(qq) / pdf(q[n,:]))

        if rand(Float64) ≤ A
            q[n+1,:] = qq
            accepted += 1
        else
            q[n+1,:] = q[n,:]
        end
    end

    acceptance_ratio = accepted / (n_iter-1)

    return q, acceptance_ratio
end


function log_Metropolis_nD(log_pdf::Function; # needs not be normalised
               n_iter::Int, var::Float64, q_init::Vector{Float64})

    local d = length(q_init)

    local q = Array{Float64}(undef, n_iter, d)
    q[1,:] = q_init
    accepted = 0

    ProgressMeter.@showprogress  1 "Iterations: " for n in 1:n_iter-1

        # gaussian proposal qq ∼ N(q[n], var)
        local qq::Vector{Float64} = q[n,:] .+ sqrt(var) * randn(Float64, d)

        # proposal function is symmetric so it cancels in the fraction
        local A::Float64 = exp(min(0, log_pdf(qq) - log_pdf(q[n,:])))

        if rand(Float64) ≤ A
            q[n+1,:] = qq
            accepted += 1
        else
            q[n+1,:] = q[n,:]
        end
    end

    acceptance_ratio = accepted / (n_iter-1)

    @info "Summary" acceptance_ratio
    return q
end
