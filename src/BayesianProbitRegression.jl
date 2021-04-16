module BayesianProbitRegression

# Load packages
using Distributions: Normal, MvNormal, pdf, cdf, Bernoulli, truncated
using LinearAlgebra: dot, I, Symmetric
using StatsBase: mean, std, cov, quantile, pacf
using Random: seed!

# Load files
include("./src/utils.jl")
include("./src/simulate.jl")
include("./src/estimate.jl")

end
