"""
    simulate(μ, x)

Simulation of a dichotomous vector via the probit link.

# Arguments
- `μ::Vector{<:Real}`: real-valued vector of coefficients
- `x::Matrix{<:Real}`: real-valued vector or matrix of inputs

# Output
- `y::Vector{Bool}`: boolean vector of outcomes
- `p::Vector{float}`: vector of probabilities
- `z::Vector{float}`: vector of latent variables
"""

function simulate(μ::T where T<:Real, x::Vector{T} where T<:Real)
    z = x * μ
    p = cdf(Normal(0, 1), z)
    y = [rand(Bernoulli(i)) for i in p]
    return (y = y, p = p, z = z)
end

function simulate(μ::Vector{T} where T<:Real, x::Matrix{T} where T<:Real)
    N, J = size(x)
    @assert length(μ) == J "μ is a $(length(μ)) and x is a $Nx$J matrix."
    z = x * μ
    p = cdf(Normal(0, 1), z)
    y = [rand(Bernoulli(p[n])) for n = 1:N]
    return (y = y, p = p, z = z)
end