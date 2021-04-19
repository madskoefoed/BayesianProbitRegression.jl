function effective_sample_size(chain::Matrix{T} where T<:AbstractFloat, k = 10::Integer)
    M, J = size(chain)
    @assert M > k "The chain of β must be longer than k."
    ρ = sum(pacf(chain, 1:k); dims = 1)
    τ = 1 .+ 2 .* sum(ρ; dims = 1)
    ESS = round.(Int, vec(M ./ τ))
    return ESS
end

function latent(x::Matrix{T} where T<:Real, chain::Matrix{T} where T<:AbstractFloat)
    @assert size(x, 2) == size(chain, 2)
    return x * chain'
end
latent(x::Vector{T} where T<:Real, chain::Matrix{T} where T<:AbstractFloat) = latent(repeat(x, 1, 1), chain)

function probability(x::Matrix{T} where T<:Real, chain::Matrix{T} where T<:AbstractFloat)
    z = latent(x, chain)
    return cdf(Normal(0, 1), z)
end
probability(x::Vector{T} where T<:Real, chain::Matrix{T} where T<:AbstractFloat) = probability(repeat(x, 1, 1), chain)