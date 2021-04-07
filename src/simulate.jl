
function simulate(μ::Vector{<:Real}, N = 1_000::Int, addIntercept = true::Bool)
    J = length(μ)
    x = randn(N, J)
    if addIntercept
        x[:, 1] .= 1.0
    end
    z = x * μ
    p = cdf(Normal(0, 1), z)
    y = [rand(Bernoulli(p[n])) for n = 1:N]
    return (y = y, p = p, z = z, x = x, μ = μ)
end