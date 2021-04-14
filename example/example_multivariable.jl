# Set seed
seed!(1234)

# True coefficient values to be used to simulate data
μ = collect(1:3);

# Simulate probit data using 2 independent variables of which the first is an intercept.
y, p, z, x, μ = simulate(μ, 1_000, intercept = false);

# Construct prior: N₃(0, 10^2I)
β₀ = MvNormal(zeros(length(μ)), 100I);

# Estimate the model (y, x) using gibbs sampling
β, chain = gibbs(y, x, β₀, 11_000);

# Posterior mean
β̅ = vec(mean(chain[1001:end, :]; dims = 1));
println("Posterior mean of βs: $β̅")

# Plot chains and compare with true values
using Plots
histogram(chain[1001:end, :], label = ["β₁" "β₂" "β₃"], title = "Histogram of βs", legend = :topright)
vline!([μ], color = :grey, linewidth = 3, label = "True")
savefig("./example/multivariable_histogram")

scatter(x[:, 3], probabilities(x, β), title = "Posterior probabilities", legend = false, markersize = 4)
xlabel!("x₃")
ylabel!("Probability")
savefig("./example/multivariable_probabilities")

# Effective Sample Size (ESS)
println("Effective Sample Size: $(effective_sample_size(chain[1001:end, :]))")