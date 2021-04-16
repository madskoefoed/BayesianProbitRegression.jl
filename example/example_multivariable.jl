# Load packages
using Plots
using StatsBase: sample

# Set seed
seed!(1234)

# True coefficient values to be used to simulate data
μ = collect(0:2);

# Simulate probit data using 3 independent variables of which the first is an intercept.
x = [ones(1_000) sample(-1:1, 1_000) randn(1_000)];
y, p, z, x, μ = simulate(μ, x);

# Construct prior: N₃(0, 10^2I)
β₀ = MvNormal(zeros(3), 100I);

# Estimate the model (y, x) using gibbs sampling
chain = gibbs(y, x, β₀, 11_000);

# Discard first 1,000 draws
chain = chain[1001:end, :]

# Posterior mean
β̅ = vec(mean(chain; dims = 1));
println("Posterior mean of βs: $β̅")

# Plot chains and compare with true values
histogram(chain, label = ["β₁" "β₂" "β₃"], title = "Histogram of βs", legend = :outerright)
vline!([μ], color = :grey, linewidth = 3, label = "True")
savefig("./example/multivariable_histogram")

# Calculate probabilities
p̂ = probability(x, chain);

# Calculate mean probabilities
p̄ = mean(p̂; dims = 2);

histogram(x[:, 3], p̄, title = "Posterior mean probabilities", legend = false, markersize = 4)
xlabel!("x₃")
ylabel!("Probability")
savefig("./example/multivariable_probabilities")

# Effective Sample Size (ESS)
println("Effective Sample Size: $(effective_sample_size(chain))")