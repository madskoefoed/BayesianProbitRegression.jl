function effective_sample_size(β::Matrix{<:AbstractFloat}, k = 10::Integer)
    M, J = size(β)
    @assert M > k "The chain of βs must be longer than k."
    ρ = sum(pacf(β, 1:k); dims = 1)
    τ = 1 .+ 2 .* sum(ρ; dims = 1)
    ESS = round.(Int, vec(M ./ τ))
    return ESS
end

function probabilities(x::Matrix{<:Real}, β::MvNormal)
    @assert size(x, 2) == length(β₀.μ) "The number of columns in x must match the dimension of β₀."
    z = x * β.μ
    p = cdf(Normal(0, 1), z)
    return p
end