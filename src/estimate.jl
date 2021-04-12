
# https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476321

function gibbs(y::Vector{<:Bool}, x::Vector{<:Real}, β₀::Normal, M = 10_000::Integer)
    N = length(x)
    @assert length(y) == N "y and x must have the same number of rows."

    chain = zeros(M)
    chain[1] = β₀.μ
    z = zeros(N)
    Q₀ = 1/β₀.σ^2
    Σ = 1/(Q₀ + sum(x.^2)) # β|z,x  ~ N(μ, Σ)
    for i in 1:(M - 1)
        # Draw latent variable z from its full conditional: z|θ, y, x
        for n in 1:N
            if y[n] == 0
                z[n] = rand(truncated(Normal(x[n] * chain[i], 1.0), -Inf, 0.0))
            else
                z[n] = rand(truncated(Normal(x[n] * chain[i], 1.0), 0.0, Inf))
            end
        end
        # Compute posterior mean of θ
        μ = Σ * (Q₀ * β₀.μ + dot(x, z))
        # Draw variable θ from its full conditional θ|z, x
        chain[i + 1] = rand(Normal(μ, sqrt(Σ)))
    end
    β = Normal(mean(chain), std(chain))
    return β, chain
end

function gibbs(y::Vector{<:Bool}, x::Matrix{<:Real}, β₀::MvNormal, M = 10_000::Integer)
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
    β = MvNormal(vec(mean(chain; dims = 1)), cov(chain))
    return β, chain
end