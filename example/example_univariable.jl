# Set seed
seed!(1234)

# True coefficient values to be used to simulate data
μ = -0.5;

# Simulate probit data using only an intercept.
x = ones(500);
y, p, z, x, μ = simulate(μ, x);

# Construct prior: N₁₀(0, 10I)
β₀ = Normal(0, 10);

# Estimate the model (y, x) using gibbs sampling
chain = gibbs(y, x, β₀, 11_000);

# Plot the chain
using Plots
plot(chain, label = "", title = "Chain of β")
savefig("./example/univariable_chain")

# Discard burn-in period
chain = chain[1001:end, :];

# Posterior mean
β̅ = mean(chain)
println("Posterior mean of β: $β̅")

# Plot chains and compare with true values
histogram(chain, label = "Coefficient", title = "Histogram of β", legend = :topright)
vline!([μ], color = :grey, linewidth = 3, label = "True")
vline!([β̅], color = :yellow, linewidth = 3, label = "Posterior mean")
vline!([quantile(Normal(0, 1), mean(y))], color = :red, linewidth = 3, label = "MLE")
savefig("./example/univariable_histogram")

# Effective Sample Size (ESS)
println("Effective Sample Size: $(effective_sample_size(chain))")