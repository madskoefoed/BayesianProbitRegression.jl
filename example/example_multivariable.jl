# Load packages
using Plots
using StatsBase: sample

# Set seed
seed!(1234)

# Number of observations
N = 1_000;

# True coefficient values to be used to simulate data
μ = [0, 1, 2];

# Simulate probit data using 3 independent variables of which the first is an intercept.
x = [ones(N) sample(-1:1, N) randn(N)];
y, p, z = simulate(μ, x);

# Construct prior: N₃(0, 10^2I)
β₀ = MvNormal(zeros(3), 100I);

# Estimate the model (y, x) using gibbs sampling
chain = gibbs(y, x, β₀, 11_000);

# Discard first 1,000 draws
chain = chain[1001:end, :];

# Posterior mean
β̅ = vec(mean(chain; dims = 1));
println("Posterior mean of βs: $β̅")

# Plot chains and compare with true values
histogram(chain, label = ["β₁" "β₂" "β₃"], title = "Histogram of βs", legend = :outerright)
vline!([μ], color = :grey, linewidth = 3, label = "True")
savefig("./example/multivariable_histogram")

# Effective Sample Size (ESS)
println("Effective Sample Size: $(ESS(chain))")

# Metropolis-Hastings (MH)
β = MvNormal(zeros(3), 0.01*I);
chainₘ, acceptance = metropolis(y, x, β₀, β, 11_000);
println("Acceptance rate: $(mean(acceptance[1001:end]))")