function simulate(μ::Vector{<:Real}, x::Matrix{<:Real})
    N, J = size(x)
    @assert length(μ) == J "μ is a $(length(μ)) and x is a $Nx$J matrix."
    z = x * μ
    p = cdf(Normal(0, 1), z)
    y = [rand(Bernoulli(p[n])) for n = 1:N]
    return (y = y, p = p, z = z, x = x, μ = μ)
end

simulate(μ::Real, x::Matrix{<:Real}) = simulate([μ], x)
simulate(μ::Real, x::Vector{<:Real}) = simulate([μ], repeat(x, 1, 1))