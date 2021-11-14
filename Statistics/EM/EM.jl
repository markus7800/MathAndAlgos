abstract type ExpectationMaximation end

function expectation(EM::ExpectationMaximation, θ_old)::AbstractMatrix
    error("Not implemented!")
end

function maximisation(EM::ExpectationMaximation, ps::AbstractMatrix, θ_old::T)::T where T
    error("Not implemented!")
end

function log_likelihood(EM::ExpectationMaximation, θ)::Float64
    error("Not implemented!")
end

function solve(EM::ExpectationMaximation, θ_0; max_iter=10^4)
    θ = θ_0
    ll = 0.
    for i in 1:max_iter
        ps = expectation(EM, θ)
        θ = maximisation(EM, ps, θ)
        ll_new = log_likelihood(EM, θ)

        if abs(ll - ll_new) ≤ 1e-6
            return θ
        end
        println("$i: $ll")
        ll = ll_new
    end
end
