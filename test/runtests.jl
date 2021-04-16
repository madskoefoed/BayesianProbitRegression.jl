using BayesianProbitRegression
using Test

@testset "BayesianProbitRegression.jl" begin
    # Simulatation tests
    y, p, z, x, μ = simulate(0, 1_000)
    @test isa(μ, Integer)
    @test μ == 0
    @test minimum(p) >= 0
    @test maximum(p) <= 1
    @test sum(y .== 0) + sum(y .== 1) == 1_000

    # Estimation tests
    β, chain = gibbs(y, x, Normal(5, 10), 11_000)    
    @test isa(β, Normal)
    @test size(chain, 1) == 11_000
    @test size(chain, 2) == 1
end
