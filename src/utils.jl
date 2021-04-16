function effective_sample_size(chain::Matrix{<:AbstractFloat}, k = 10::Integer)
    M, J = size(chain)
    @assert M > k "The chain of βs must be longer than k."
    ρ = sum(pacf(chain, 1:k); dims = 1)
    τ = 1 .+ 2 .* sum(ρ; dims = 1)
    ESS = round.(Int, vec(M ./ τ))
    return ESS
end

function latent(x::Matrix{<:Real}, chain::Matrix{<:AbstractFloat})
    @assert size(x, 2) == size(chain, 2)
    return x * chain'
end
latent(x::Vector{<:Real}, chain::Matrix{<:AbstractFloat}) = latent(repeat(x, 1, 1), chain)

function probability(x::Matrix{<:Real}, chain::Matrix{<:AbstractFloat})
    z = latent(x, chain)
    return cdf(Normal(0, 1), z)
end
probability(x::Vector{<:Real}, chain::Matrix{<:AbstractFloat}) = probability(repeat(x, 1, 1), chain)