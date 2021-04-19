
# https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476321

function gibbs(y::Vector{<:Bool}, β₀::Normal, M = 10_000::Integer)
    return gibbs(y, ones(length(y), 1), β₀, M)
end

function gibbs(y::Vector{<:Bool}, x::Vector{<:Real}, β₀::Normal, M = 10_000::Integer)
    gibbs(y, repeat(x, 1, 1), MvNormal([β₀.μ], β₀.σ^2*I), M)
end

function gibbs(y::Vector{<:Bool}, x::Matrix{<:Real}, β₀::Normal, M = 10_000::Integer)
    if size(x, 2) == 1
    gibbs(y, repeat(x, 1, 1), MvNormal([β₀.μ], β₀.σ^2*I), M)
    else
        throw(DimensionMismatch("β₀ is a univariate normal prior, but x is a matrix with more than 1 column."))
    end
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
    return chain
end

function MH(y::Vector{<:Bool}, x::Matrix{<:Real}, β₀::MvNormal, β::MvNormal, M = 10_000::Integer)
    N, J = size(x)
    @assert length(y) == N "y and x must have the same number of rows."
    @assert J == length(β₀.μ) "The number of columns in x must match the dimension of β₀."

    chain = zeros(M, J)
    chain[1, :] = β₀.μ
    loglik = MH_loglik(y, x, β)
    for i in 1:(M - 1)
        
    end
    return chain
end

function MH_prior(β₀::MvNormal, β::MvNormal)
    return sum(logpdf(β₀, β.μ))
end

function MH_loglik(y, x, β::MvNormal)
    z = x * β.μ
    p = cdf(Normal(0, 1), z)
    d = 0.0
    for n in 1:length(y)
        d += logpdf(Bernoulli(p[n]), y[n])
    end
    return d
end

function MH_posterior(y, x, β₀::MvNormal, β::MvNormal)
    return MH_prior(β₀, β) + MH_loglik(y, x, β)
end

function MH_candidate(β::MvNormal)
    β.μ += rand(β)
    return β
end