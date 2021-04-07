
# https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476321

function gibbs(y::Vector{<:Bool}, x::Matrix{<:Real}, β₀::MvNormal, M = 10_000::Integer)
    N, J = size(x)
    @assert size(y, 1) == N "y and x must have the same number of rows."
    @assert J == length(β₀.μ) "The number of columns in x must match the dimension of β₀."

    β = zeros(M, J)
    β[1, :] = β₀.μ
    z = zeros(N)
    Q₀ = inv(β₀.Σ)
    Σ = inv(Q₀ + x'x) # β|z,x  ~ N*(μ, Σ)

    for i in 1:(M - 1)
        # Draw latent variable z from its full conditional: z|θ, y, x
        for n in 1:N
            if y[n] == 0
                z[n] = rand(truncated(Normal(dot(x[n, :], β[i, :]), 1.0), -Inf, 0.0))
            else
                z[n] = rand(truncated(Normal(dot(x[n, :], β[i, :]), 1.0), 0.0, Inf))
            end
        end
        # Compute posterior mean of θ
        μ = Σ * (Q₀ * β₀.μ + x'z)
        # Draw variable θ from its full conditional θ|z, x
        β[i + 1, :] = rand(MvNormal(μ, Σ))
    end
    return β
end

function effective_sample_size(β::Matrix{<:AbstractFloat}, k = 10::Integer)
    M, J = size(β)
    @assert M > k "The chain of βs must be longer than k."
    ρ = sum(pacf(β, 1:k); dims = 1)
    τ = 1 .+ 2 .* sum(ρ; dims = 1)
    ESS = round.(Int, vec(M ./ τ))
    return ESS
end