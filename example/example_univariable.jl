# True coefficient values to be used to simulate data
μ = -0.5

# Simulate probit data using 2 independent variables of which the first is an intercept.
y, p, z, x, μ = simulate(μ, 500, intercept = false);

# Construct prior: N₁₀(0, 10I)
β₀ = Normal(0, 10)

# Estimate the model (y, x) using gibbs sampling
β, chain = gibbs(y, x, β₀);

# Plot chains and compare with true values
using Plots
histogram(chain[1001:10:end], label = "β₁", title = "Histogram", legend = :topright)
vline!([μ], color = :grey, linewidth = 3, label = "True")

# Plot the true probabilities against the estimated probabilities
scatter(probabilities(x, β), p, legend = false, title = "Probabilities", xlab = "Estimated probabilities", ylab ="True probabilities")
plot!(0:0.01:1, 0:0.01:1, color = :grey, linewidth = 3)