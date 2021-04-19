function MH(y::Vector{T} where T<:Bool, β₀::Normal, M = 10_000::Integer)
    return MH(y, ones(length(y), 1), β₀, β, M)
end

function MH(y::Vector{T} where T<:Bool, x::Vector{T} where T<:Real, β₀::Normal, β::Normal, M = 10_000::Integer)
    β₀ = MvNormal([β₀.μ], β₀.σ^2*I)
    β  = MvNormal([β.μ], β.σ^2*I)
    return (MH(y, repeat(x, 1, 1), β₀, β, M))
end

function MH(y::Vector{T} where T<:Bool, x::Matrix{T} where T<:Real, β₀::MvNormal, β::MvNormal, M = 10_000::Integer)
    N, J = size(x)
    @assert length(y) == N "y and x must have the same number of rows."
    @assert J == length(β₀.μ) "The number of columns in x must match the dimension of β₀."
    @assert J == length(β.μ) "The number of columns in x must match the dimension of β."

    chain = zeros(M, J)
    chain[1, :] = β₀.μ
    posterior = MH_posterior(y, x, β₀, chain[1, :])
    accept = zeros(Bool, M)
    for i in 1:(M - 1)
        # Proposal likelihood
        μ = MH_candidate(β, chain[i, :])
        p = MH_posterior(y, x, β₀, μ)
        # Compare log-posteriors
        if log(rand()) < (p - posterior)
            chain[i + 1, :] = μ
            posterior = copy(p)
            accept[i + 1] = 1
        else
            chain[i + 1, :] = chain[i, :]
        end
    end
    return (chain, accept)
end

function MH_prior(β₀::MvNormal, μ::Vector{T} where T<:AbstractFloat)
    return sum(logpdf(β₀, μ))
end

function MH_loglik(y, x, μ::Vector{T} where T<:AbstractFloat)
    z = x * μ
    p = cdf(Normal(0, 1), z)
    d = 0.0
    for n in 1:length(y)
        d += logpdf(Bernoulli(p[n]), y[n])
    end
    return d
end

function MH_posterior(y, x, β₀::MvNormal, μ::Vector{T} where T<:AbstractFloat)
    return MH_prior(β₀, μ) + MH_loglik(y, x, μ)
end

function MH_candidate(β::MvNormal, μ::Vector{T} where T<:AbstractFloat)
    μ += rand(β)
    return μ
end