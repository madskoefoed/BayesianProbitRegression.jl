function simulate(μ::Real, N = 1_000::Int; intercept = true::Bool)
    if intercept
        x = ones(N)
    else
        x = randn(N)
    end
    z = x * μ
    p = cdf(Normal(0, 1), z)
    y = [rand(Bernoulli(p[n])) for n = 1:N]
    return (y = y, p = p, z = z, x = x, μ = μ)
end

function simulate(μ::Vector{<:Real}, N = 1_000::Int; intercept = true::Bool)
    J = length(μ)
    x = randn(N, J)
    if intercept
        x[:, 1] .= 1.0
    end
    z = x * μ
    p = cdf(Normal(0, 1), z)
    y = [rand(Bernoulli(p[n])) for n = 1:N]
    return (y = y, p = p, z = z, x = x, μ = μ)
end