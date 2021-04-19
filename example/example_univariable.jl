# Set seed
seed!(1234)

# Number of observations
N = 500

# True coefficient values to be used to simulate data
μ = -0.5;

# Simulate probit data using only an intercept.
x = ones(N);
y, p, z = simulate(μ, x);

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
vline!([quantile(Normal(0, 1), mean(y))], color = :black, linewidth = 3, label = "MLE")
savefig("./example/univariable_histogram")

# Effective Sample Size (ESS)
println("Effective Sample Size: $(effective_sample_size(chain))")

# Metropolis-Hastings (MH)
β = Normal(0, 0.1)
chainₘ, acceptance = MH(y, x, β₀, β, 11_000);
println("Acceptance rate: $(mean(acceptance[1001:end]))")

plot(chain[:, 1], label = "Gibbs", title = "Chain of β")
plot!(chainₘ[:, 1], label = "MH")
hline!([μ], color = :grey, linewidth = 3, label = "True")
hline!([quantile(Normal(0, 1), mean(y))], color = :black, linewidth = 3, label = "MLE")
savefig("./example/univariable_gibbs_vs_mh")