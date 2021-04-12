module BayesianProbitRegression

# Load packages
using Distributions, LinearAlgebra, StatsBase

# Load files
include("./src/simulate.jl")
include("./src/estimate.jl")
include("./src/utils.jl")

end
