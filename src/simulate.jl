function simulate(μ::T where T<:Real, x::Vector{T} where T<:Real)
    z = x * μ
    p = cdf(Normal(0, 1), z)
    y = [rand(Bernoulli(i)) for i in p]
    return (y = y, p = p, z = z)
end

function simulate(μ::Vector{T} where T<:Real, x::Matrix{T} where T<:Real)
    N, J = size(x)
    @assert length(μ) == J "μ is a $(length(μ)) and x is a $Nx$J matrix."
    z = x * μ
    p = cdf(Normal(0, 1), z)
    y = [rand(Bernoulli(p[n])) for n = 1:N]
    return (y = y, p = p, z = z)
end

#simulate(μ::Real, x::Matrix{T}) = simulate([μ], x)
#simulate(μ::Real, x::Vector{T} where T<:Real) = simulate([μ], repeat(x, 1, 1))