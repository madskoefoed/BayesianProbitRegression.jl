module BayesianProbitRegression

# Load packages
using Distributions, LinearAlgebra, StatsBase

# Load files
include("./src/utils.jl")
include("./src/simulate.jl")
include("./src/estimate.jl")

end
