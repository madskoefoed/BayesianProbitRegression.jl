module BayesianProbitRegression

using Distributions: Normal, MvNormal, pdf, cdf, Bernoulli, truncated, logpdf
using LinearAlgebra: dot, I, Symmetric
using StatsBase: mean, std, cov, quantile, pacf
using Random: seed!

# Load files
include("./src/utils.jl")
include("./src/simulate.jl")
include("./src/Gibbs.jl")
include("./src/Metropolis.jl")

# Export
export gibbs,
metropolis,
simulate,
probability,
latent,
effective_sample_size

end
