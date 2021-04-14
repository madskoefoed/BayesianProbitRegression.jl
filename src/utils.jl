function effective_sample_size(β::Vector{<:AbstractFloat}, k = 10::Integer)
    M = length(β)
    @assert M > k "The chain of βs must be longer than k."
    ρ = sum(pacf(β, 1:k))
    τ = 1 + 2 * sum(ρ)
    ESS = round.(Int, vec(M ./ τ))
    return ESS
end

function effective_sample_size(β::Matrix{<:AbstractFloat}, k = 10::Integer)
    M, J = size(β)
    @assert M > k "The chain of βs must be longer than k."
    ρ = sum(pacf(β, 1:k); dims = 1)
    τ = 1 .+ 2 .* sum(ρ; dims = 1)
    ESS = round.(Int, vec(M ./ τ))
    return ESS
end

latentvariables(x::Matrix{<:Real}, β::MvNormal) = x * β.μ
latentvariables(x::Vector{<:Real}, β::Normal) = x * β.μ

latentvariables(x::Matrix{<:Real}, β::Vector{<:Real}) = x * β
latentvariables(x::Vector{<:Real}, β::Real) = x * β

probabilities(z::Vector{<:AbstractFloat}) = cdf(Normal(0, 1), z)
probabilities(x::Matrix{<:Real}, β::MvNormal) = cdf(Normal(0, 1), latentvariables(x, β))
probabilities(x::Vector{<:Real}, β::Normal) = cdf(Normal(0, 1), latentvariables(x, β))