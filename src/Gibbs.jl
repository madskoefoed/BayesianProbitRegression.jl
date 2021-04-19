"""
    gibbs(y, x, β₀, M)

A Gibbs algorithm for probit regression.

# Arguments
- `y`: boolean vector of outcomes
- `x`: real valued vector or matrix of inputs
- `β₀`: a univariate or multivariate normal distribution for the prior
- `M`: the number of draws

# Output
- `chain`: matrix of draws from the target distribution

The function can be called without 'x' in which case a constant-only model is estimated.

The Gibbs algorithm is based on the paper Bayesian Analysis of Binary
and Polychotomous Response Data (Albert and Chib, 1993). 
"""

function gibbs(y::Vector{T} where T<:Bool, β₀::Normal, M = 10_000::Integer)
    return gibbs(y, ones(length(y), 1), β₀, M)
end

function gibbs(y::Vector{T} where T<:Bool, x::Vector{T} where T<:Real, β₀::Normal, M = 10_000::Integer)
    gibbs(y, repeat(x, 1, 1), MvNormal([β₀.μ], β₀.σ^2*I), M)
end

function gibbs(y::Vector{T} where T<:Bool, x::Matrix{T} where T<:Real, β₀::MvNormal, M = 10_000::Integer)
    N, J = size(x)
    @assert length(y) == N "y and x must have the same number of rows."
    @assert J == length(β₀.μ) "The number of columns in x must match the dimension of β₀."

    chain = zeros(M, J)
    chain[1, :] = β₀.μ
    z = zeros(N)
    Q₀ = inv(β₀.Σ)
    Σ = inv(Symmetric(Q₀ + x'x)) # β|z,x  ~ MvN(μ, Σ)
    for i in 1:(M - 1)
        # Draw latent variable z from its full conditional: z|θ, y, x
        for n in 1:N
            if y[n] == 0
                z[n] = rand(truncated(Normal(dot(x[n, :], chain[i, :]), 1.0), -Inf, 0.0))
            else
                z[n] = rand(truncated(Normal(dot(x[n, :], chain[i, :]), 1.0), 0.0, Inf))
            end
        end
        # Compute posterior mean of θ
        μ = vec(Σ * (Q₀ * β₀.μ + x'z))
        # Draw variable θ from its full conditional θ|z, x
        chain[i + 1, :] = rand(MvNormal(μ, Σ))
    end
    return chain
end