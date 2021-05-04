# BayesianProbitRegression.jl

# Installation
A Julia package for Bayesian probit regression in Julia via Gibbs (and Metropolis) sampling. The package allows for estimation and simulation of a dichotomous variable. The package is unregistered at the moment, so you need to clone it from the repository:

```julia
Pkg.clone("git://github.com/madskoefoed/BayesianProbitRegression.jl)
```

The package can then be loaded by typing `using BayesianProbitRegression` in Julia.

Below I provide some examples of how to use the package.

# Introduction
Probit regression is arguably the second-most common way of estimating a binary variable (after Logistic regression), that is a variable which can take one of only two values. In Probit regression, the cumulative distribution function of the standard normal distribution is assumed: $Pr(y = 1|X) = \Phi(X\beta)$, where $y$ is an N-dimensional vector of dichotomous values, $X$ is an $N \times J$ matrix of independent variables, and $\beta$ is a J-dimensional vector of parameters.

Assuming that $y_{n}$ is a dichotomization of an underlying latent variable, $z_{n}$ for $n = 1,...,N$, the model can be written as:

$$
y_{n} = \begin{cases} 1, & \text{if $z_{n} > 0$} \\ 0, & \text{if $z_{n} \leq 0$} \end{cases} \\
z_{n} = x_{n} \beta + \epsilon_{n} \\
\epsilon_{n} \sim \mathcal{N}(0, 1)
$$

By introducing the vector of latent variables $z$, the vector of dependent variables $y$ becomes an indicator of whether $z > 0$ or $z \leq 0$. Of course, the latent variables $z$ is not actually observed, only the binary outcomes $y$.

## Bayesian Probit Regression
Given that we are feeling _Bayesian_ today, we need to define some priors for the parameters: $\beta \sim \pi(\beta)$. To ensure conjugacy, we let the priors be multivariate normally distributed: $\beta \sim \mathcal{MVN}(\beta_{0}, Q_{0})$.

By introducing the latent variables $z$ and assuming that $\beta$ is distributed according to a multivariate normal distribution, we can sample using Gibbs sampiling from the conditional distributions $p(\beta|z,y,X)=p(\beta|z,X)$ and $p(z|\beta,y,X)$ as follows:

$$
z_{i}|\beta,y_{i},x_{i} \sim \begin{cases} \mathcal{TN}(x_{i} \beta, 1, 0, \infty), & \text{if $y_{i} = 1$} \\ \mathcal{TN}(x_{i} \beta, 1, -\infty, 0), & \text{if $y_{i} = 0$} \end{cases} \\
\beta|z,X \sim \mathcal{N}(\mu, \Sigma) \\
\mu = \Sigma(Q_{0}^{-1} \beta_{0} + X^{'} z) \\
\Sigma = (Q_{0}^{-1} + X^{'} X)^{-1}
$$

By repeatedly sampling from the two conditional distributions above our target (joint) distribution $p(z,\beta|y,X)$ is obtained.

## Examples
I demonstrate the algorithm with two examples, one with a constant and another with two independent variables and no constant (though it will be estimated).

### Example With 1 (Constant) Input
For the [first example](/example/example_univariable.jl) I generate $N = 500$ observations of $y_{n}$ using $\beta = -0.5$ and $x_{n} = 1$, which implies that for all 500 observations the probability of $y = 1$ is the same:

```julia
y, p, z, x, μ = simulate(-0.5, 500, intercept = true);
```

The Maximum Likelihood (MLE) estimate of the probability is simply the sample mean (in this case 0.33) and hence the MLE estimate of $\beta$ is given by the inverse CDF $\Phi^{-1}(\hat{p}) \approx -0.44$; not too far from the true value of -0.5.

```julia
quantile(Normal(0, 1), mean(y))
```

Using a non-informative prior $N(0, 10)$, the parameter $\beta$ is estimated using 11,000 draws (of which 1,000 will be discarded later as _burn in_ samples).

```julia
β₀ = Normal(0, 10)
β, chain = gibbs(y, x, β₀, 11_000);
```

The MCMC chain starts at the prior value of 0, but quickly convergences to a stationary distribution.

![Chain](/example/univariable_chain.png)

Given our weak prior the posterior distribution's mean is expected to be around the ML estimate of -0.44 - as it indeed is.

![Histogram](/example/univariable_histogram.png)

By estimating the coefficient using a Bayesian method we get the full posterior distribution of $\beta$, which allows us to calculate credible intervals. The 95% credible interval, for example, is $(-0.555, -0.325)$ which implies that the credible interval for the probability is $(0.289, 0.372)$.

### Example With 3 Inputs
For the [second example](/example/example_multivariable.jl) I generate $n = 1,...,1_000$ observations of $y_{n}$ using the coefficients 0, 1, and 2. The three inputs are defined as follows:
* $x_{1}$ is constant
* $x_{2}$ takes the values -1, 0, and 1
* $x_{3}$ is NID

Given that 

```julia
y, p, z, x, μ = simulate([1, 2, 3], 1_000, intercept = false);
```

Similarly to the first example above, I choose normally distributed priors and estimate using 11,000 samples.

```julia
β₀ = MvNormal(zeros(3), 100I)
β, chain = gibbs(y, x, β₀, 11_000);
```

The histograms of the posterior distributions of the $\beta$s show that they are generally close to their true values.

![Histogram](/example/multivariable_histogram.png)

To check convergence the Effective Sample Size (ESS) can be calculated, which takes the serial correlation of the (Markov) chains into account. The package allows for easy calculation of the ESS by running the command `effective_sample_size(chain[1001:end, :], 10)`, where `10` specifies the number of lags to include in the calculation and the first 1,000 samples are left out. The resulting vector `[3023, 2789, 3079]` suggests that we would need to roughly triple the number of draws to 30,000 to have an effective sample size of 10,000 due to the serial correlation in the chains.

## Metropolis-Hastings
The package also provides a standard Metropolis-Hastings (MH) algorithm as an alternative to the Gibbs algorithm. Using _Example With 3 Inputs_ above, we could use MH rather than Gibbs as follows:

```julia
β = MvNormal(zeros(3), 0.01*I)
chainₘ, acceptance = MH(y, x, β₀, β, 11_000)
```

The MH algorithm requires a distribution for the candidate parameters $\beta$, here chosen to be multivariate normally distributed. And unlike the Gibbs sampler - which accepts all proposals - the MH function returns two outputs, the second of which is $\alpha$, which provides a vector of Boolean values indicating whether the candidate parameters were accepted or not. The acceptance rate is then simply the mean of the vector (discarding the burn-in elements): `mean(acceptance[1001:end])`, which for the example is approx. 34%.
