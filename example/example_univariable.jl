# Set seed
seed!(1234)

# True coefficient values to be used to simulate data
μ = -0.5;

# Simulate probit data using only an intercept.
y, p, z, x, μ = simulate(μ, 500, intercept = true);

# Construct prior: N₁₀(0, 10I)
β₀ = Normal(0, 10);

# Estimate the model (y, x) using gibbs sampling
β, chain = gibbs(y, x, β₀, 11_000);

# Posterior mean
β̅ = mean(chain[1001:end]);
println("Posterior mean of β: $β̅")

# Plot chains and compare with true values
using Plots
using StatsBase: quantile
histogram(chain[1001:end], label = "Coefficient", title = "Histogram of β", legend = :topright)
vline!([μ], color = :grey, linewidth = 3, label = "True")
vline!([β̅], color = :yellow, linewidth = 3, label = "Posterior mean")
vline!([quantile(Normal(0, 1), mean(y))], color = :red, linewidth = 3, label = "MLE")
savefig("./example/univariable_histogram")

# Plot the chain
plot(chain, label = "", title = "Chain of β")
savefig("./example/univariable_chain")