# True coefficient values to be used to simulate data
μ = [-0.5, 2.0]

# Simulate probit data using 2 independent variables of which the first is an intercept.
y, p, z, x, μ = simulate(μ, 500, intercept = false);

# Construct prior: N₁₀(0, 10I)
β₀ = MvNormal(zeros(length(μ)), 100I)

# Estimate the model (y, x) using gibbs sampling
β, chain = gibbs(y, x, β₀, 11_000);

# Plot chains and compare with true values
using Plots
histogram(chain[1001:10:end, :], label = ["β₁" "β₂"], title = "Histogram", legend = :topright, xlim = (-1, 3))
vline!(μ, color = :grey, linewidth = 3, label = "True")

# Plot the true probabilities against the estimated probabilities
scatter(probabilities(x, β), p, legend = false, title = "Probabilities",
    xlab = "Estimated probabilities", ylab ="True probabilities")
plot!(0:0.01:1, 0:0.01:1, color = :grey, linewidth = 3)

X = range(-1, 0, length=100)
Y = range(1, 3, length=100)
Z = [pdf(β, [x, y]) for y in Y, x in X] # Note x-y "for" ordering
contourf(X, Y, Z, color=:viridis, xlab = "β₁", ylab = "β₂")

# Plot the chains
plot(chain, label = "", title = "Chain of βs")
hline!([μ], color = :grey, linewidth = 3, label = "True")