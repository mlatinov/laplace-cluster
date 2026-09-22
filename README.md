# laplace-cluster

A [Laplace](https://github.com/mlatinov/laplace) library of clustering and mixture-model building blocks for Stan — soft k-means, Gaussian mixtures from spherical to full covariance, Poisson and negative-binomial mixtures, latent class analysis, Naive Bayes / LDA document clustering, mixtures of regressions, and the generated-quantities utilities that go with all of them (responsibilities, assignment, entropy, co-clustering). Import it into any `.laplace` model and call it with namespaced calls (`cluster::function_name(...)`).

Like all Laplace libraries, `cluster` compiles down to plain, readable Stan functions. Nothing about how you use it hides what actually ends up in your `.stan` file.

## How it's organised

The library is layered. Every family function returns the same shape — an `N x K` matrix of component log-densities — so any one of them can be dropped into any of the functions above it without changes.

1. **Core** (`mixture_pointwise`, `mixture_lpdf`, `responsibilities`, `log_responsibilities`) is the family-agnostic mixing step: given an `N x K` matrix of component log-densities `L` and log-weights, it marginalizes over clusters and recovers posterior membership. This is the only layer every model in the library goes through.
2. **Components** (`sq_dist`, `kmeans_loglik`, `spherical_loglik`, `diag_loglik`, `mvn_chol_loglik`, `pois_lik`, `negbin_loglik`, `latent_class_loglik`, `multinomial_loglik`, `regression_loglik`) each compute one family's `L`. Swapping the component family is a one-line change; the core layer doesn't change.
3. **Weights** (`stick_breaking_log_weights`, `softmax_log_weights`) build the log-weights that Core needs, for a truncated Dirichlet-process prior or for covariate-dependent mixing proportions.
4. **Utilities** (`assign_rng`, `map_assign`, `assignment_entropy`, `complete_loglik`, `coclustering`) turn responsibilities into labels, uncertainty measures, and diagnostics, for `generated quantities`.
5. **Models** (`mixture_ss_lpdf`, `lda_lpmf`) are named, complete likelihoods built on top of the layers above: semi-supervised mixtures and Latent Dirichlet Allocation.

Every family and every weight function in this library returns log-scale quantities and stays on the log scale until the last possible moment — that's what makes them safe to mix, sum, and reuse across layers without underflow.

## Notation

| Symbol | Meaning |
| --- | --- |
| $N, D, K$ | observations, features, clusters |
| $\mathbf Y \in \mathbb R^{N\times D}$ | data matrix |
| $\theta \in \Delta^{K-1}$ | mixing weights (a simplex) |
| $\mathbf L \in \mathbb R^{N\times K}$, $L_{nk} = \log f(y_n \mid \phi_k)$ | component log-densities — what every function in Layer 2 returns |
| $\mathbf R \in [0,1]^{N\times K}$, $r_{nk}$ | responsibilities — posterior probability that observation $n$ belongs to cluster $k$ |

$\mathbf L$ is *not* a log-likelihood of the model. It's the log-likelihood of each observation under each cluster, before mixing. The model's log-likelihood only exists after Core sums over clusters:

$$
\ell_n = \log\sum_{k=1}^{K}\exp(\log\theta_k + L_{nk}), \qquad \log p(\mathbf Y) = \sum_n \ell_n.
$$

## Family log-densities

Each function returns an `N x K` matrix `L`. Normalizing constants are kept (except `kmeans_loglik`, noted below), so these are safe to compare across families with `loo`.

| Function | Arguments after `y` | Models |
| --- | --- | --- |
| `kmeans_loglik` | `mu, beta` | Soft k-means: unit-shape Gaussian, constants dropped |
| `spherical_loglik` | `mu, sd` | Gaussian, shared shape, one scale per cluster |
| `diag_loglik` | `mu, S` | Gaussian, independent scale per cluster and per feature |
| `mvn_chol_loglik` | `mu, C` | Gaussian, full covariance per cluster (Cholesky factor) |
| `pois_lik` | `lambda` | Poisson counts, features conditionally independent |
| `negbin_loglik` | `mu, psi` | Negative binomial (mean/precision) counts |
| `latent_class_loglik` | `p, m` | Independent Bernoulli items, with a missingness mask |
| `multinomial_loglik` | `phi` | Bag-of-words / count vectors (Naive Bayes document clustering) |
| `regression_loglik` | `X, B, sd` | A separate linear regression per cluster (mixture of regressions) |

`sq_dist(y, mu)` is the shared building block behind the Gaussian families — the pairwise squared-distance matrix, computed with one matrix product instead of an `N * K * D` loop:

$$
\delta_{nk} = \lVert y_n - \mu_k\rVert^2 = \lVert y_n\rVert^2 + \lVert\mu_k\rVert^2 - 2\,y_n\cdot\mu_k.
$$

**`kmeans_loglik` is not normalized.** It's $L_{nk} = -\tfrac{\beta}{2}\lVert y_n - \mu_k\rVert^2$, the unit-variance Gaussian with the $-\tfrac{D}{2}\log 2\pi$ constant dropped. Fine for sampling, but its value is off by that constant from a true log-likelihood — don't compare it against a normalized family with `loo`. Use `spherical_loglik` with `sd` fixed if you need the constant kept.

## Weights

| Function | Returns | What it does |
| --- | --- | --- |
| `stick_breaking_log_weights(v)` | `vector[K]` | Stick-breaking weights from `K-1` break proportions; base of the truncated Dirichlet-process prior |
| `softmax_log_weights(X, G)` | `matrix[N, K]` | Covariate-dependent weights via softmax regression, last class fixed as the zero-logit reference |

Every Core, Utility, and Model function that takes `log_theta` has two overloads: a `vector` for weights shared across all observations, and a `matrix` for observation-specific weights (what `softmax_log_weights` returns). Pick whichever matches how you built your weights.

## Choosing K

With $\theta \sim \mathrm{Dirichlet}(\alpha \mathbf 1_K)$ and more clusters than the truth, the concentration $\alpha$ controls what happens to the extra ones, relative to $d$, the number of free parameters in one component:

$$
\alpha < d/2: \text{ redundant components empty out}, \qquad \alpha > d/2: \text{ redundant components stay occupied}.
$$

| Component family | $d$ |
| --- | --- |
| Spherical Gaussian in $\mathbb R^D$ | $D + 1$ |
| Diagonal Gaussian | $2D$ |
| Full-covariance Gaussian | $D + D(D+1)/2$ |
| Latent class ($J$ items) | $J$ |

For a univariate normal mixture, $d = 2$, so $\alpha < 1$ is the sparse regime.

## Utilities

| Function | Returns | What it does |
| --- | --- | --- |
| `assign_rng(R)` | `array[N] int` | One sampled cluster label per observation, `z_n ~ Categorical(r_n.)` |
| `map_assign(R)` | `array[N] int` | Most probable cluster per observation, `argmax_k r_nk` |
| `assignment_entropy(R)` | `vector[N]` | Per-observation uncertainty, $H_n \in [0, \log K]$ |
| `complete_loglik(R, L, log_theta)` | `real` | ICL-style criterion: $\sum_n \ell_n - \sum_n H_n$, penalizes overlapping clusters |
| `coclustering(R)` | `matrix[N, N]` | $S_{ij} = \sum_k r_{ik} r_{jk}$, posterior probability that $i$ and $j$ share a cluster |

`coclustering` and `assignment_entropy` are invariant to relabeling components; `assign_rng` and `map_assign` are not — see **Label switching** below before summarizing labels across posterior draws.

## Models

| Form | What it does |
| --- | --- |
| `mixture_ss_lpdf(L, log_theta, z)` | Semi-supervised mixture: known labels in `z` contribute directly, `z[n] = 0` marginalizes as usual |
| `lda_lpmf(Y, Theta, Phi)` | Latent Dirichlet Allocation, marginalized to word counts |

`mixture_ss_lpdf` with every entry of `z` at `0` is exactly `mixture_lpdf`; with every entry known it's supervised Naive Bayes classification.

Every function carries `@brief`, `@param`, `@return`, `@math`, and `@example` documentation, so you can read it from the terminal without leaving your model:

```
laplace doc cluster::spherical_loglik
```

## Installation

`cluster` is distributed as a git-hosted Laplace library — there's no published registry entry yet, so it's added by pointing `laplace` (or `cmdlaplacer`, if you're working from R) directly at the repository. The package lives in the repository's `laplace/` subdirectory, so pass it as the subdir.

### Via the `laplace` CLI

From inside a Laplace project (a directory with its own `laplace.toml`):

```
laplace add cluster --git https://github.com/mlatinov/laplace-cluster --tag 0.1.0 --subdir laplace
```

### Via R (`cmdlaplacer`)

```r
library(cmdlaplacer)

laplace_install_git(
  "cluster",
  "https://github.com/mlatinov/laplace-cluster",
  tag = "0.1.0",
  subdir = "laplace"
)
```

Either way, this pins the dependency in your project's `laplace.toml`/`laplace.lock` at tag `0.1.0`. Check the [tags](https://github.com/mlatinov/laplace-cluster/tags) for newer versions as they become available.

## Usage

Import the library in a `library { }` block and call its functions with the `cluster::` namespace prefix.

### Soft k-means, fixed temperature

The simplest model in the library: uniform weights, fixed spherical variance, `beta` as the inverse temperature. `beta = 1` recovers plain soft k-means; larger `beta` sharpens the assignment toward hard k-means.

```
library {
    import cluster
}

data {
  int<lower=1> N;
  int<lower=1> D;
  int<lower=1> K;
  matrix[N, D] y;
  real<lower=0> beta;
}

parameters {
  matrix[K, D] mu;
}

model {
  to_vector(mu) ~ normal(0, 5);

  matrix[N, K] L = cluster::kmeans_loglik(y, mu, beta);
  vector[K] log_theta = rep_vector(-log(K), K);   // uniform weights

  target += cluster::mixture_lpdf(L, log_theta);
}

generated quantities {
  matrix[N, K] L = cluster::kmeans_loglik(y, mu, beta);
  vector[K] log_theta = rep_vector(-log(K), K);
  matrix[N, K] R = cluster::responsibilities(L, log_theta);

  array[N] int z_hat = cluster::map_assign(R);
  vector[N] H = cluster::assignment_entropy(R);
}
```

### Spherical Gaussian mixture with a Dirichlet prior

Learned weights, learned per-cluster scale, an `ordered` constraint on one coordinate to keep label switching in check for a 1-D-separable problem.

```
library {
    import cluster
}

data {
  int<lower=1> N;
  int<lower=1> D;
  int<lower=1> K;
  matrix[N, D] y;
  real<lower=0> alpha;                  // Dirichlet concentration; alpha < (D+1)/2 for a sparse prior
}

parameters {
  ordered[K] mu_1;                      // first coordinate only, to fix labeling
  matrix[K, D - 1] mu_rest;
  vector<lower=0>[K] sd;
  simplex[K] theta;
}

transformed parameters {
  matrix[K, D] mu = append_col(mu_1, mu_rest);
}

model {
  mu_1   ~ normal(0, 5);
  to_vector(mu_rest) ~ normal(0, 5);
  sd     ~ exponential(1);
  theta  ~ dirichlet(rep_vector(alpha, K));

  matrix[N, K] L = cluster::spherical_loglik(y, mu, sd);
  target += cluster::mixture_lpdf(L, log(theta));
}

generated quantities {
  matrix[N, K] L = cluster::spherical_loglik(y, mu, sd);
  matrix[N, K] R = cluster::responsibilities(L, log(theta));

  array[N] int z_hat  = cluster::map_assign(R);
  real cll             = cluster::complete_loglik(R, L, log(theta));
  vector[N] log_lik    = cluster::mixture_pointwise(L, log(theta));   // for loo
}
```

### Naive Bayes document clustering, then LDA

`multinomial_loglik` gives you unsupervised document clustering (one topic per document) in a couple of lines; `lda_lpmf` is the same idea generalized to a topic per token.

```
library {
    import cluster
}

data {
  int<lower=1> M;                       // documents
  int<lower=1> V;                       // vocabulary size
  int<lower=1> K;                       // topics
  array[M, V] int<lower=0> Y;           // word counts
  real<lower=0> alpha;
  real<lower=0> beta;
}

parameters {
  simplex[V] phi[K];                    // topic-word distributions
}

transformed parameters {
  matrix[K, V] Phi;
  for (k in 1:K) Phi[k] = phi[k]';
}

model {
  for (k in 1:K) phi[k] ~ dirichlet(rep_vector(beta, V));

  // treat Y as real for multinomial_loglik's matrix signature
  matrix[N, K] L = cluster::multinomial_loglik(to_matrix(Y), Phi);
  vector[K] log_theta = rep_vector(-log(K), K);

  target += cluster::mixture_lpdf(L, log_theta);
}
```

Switching to full LDA (a topic per token, not per document) means adding a `Theta` per-document simplex and calling `cluster::lda_lpmf(Y, Theta, Phi)` directly in `model` instead — see `lda_lpmf`'s `@example` via `laplace doc cluster::lda_lpmf`.

### From R

With `cmdlaplacer`, the `.laplace` file compiles straight to a `cmdstanr` model, and the generated `.stan` file stays on disk next to it:

```r
library(cmdlaplacer)

mod <- laplace_model("spherical_mixture.laplace")
fit <- mod$sample(data = list(N = N, D = D, K = 3, y = y, alpha = 0.5))

fit$draws("z_hat")
```

## Things to know

- **`L` is a log-likelihood, not *the* log-likelihood.** `L[n, k]` is how plausible point `n` is under cluster `k` alone. The model's actual log-likelihood only exists after `mixture_lpdf`/`mixture_pointwise` sums over `k`. Don't call a family function and `target +=` it directly — it always goes through Core first.
- **`kmeans_loglik` drops constants; the other families don't.** Compare `kmeans_loglik` fits against each other, never against `spherical_loglik` or `diag_loglik` fits with `loo` — the additive constant differs.
- **Vectorized Stan `_lpdf`/`_lpmf` calls sum.** `normal_lpdf(y | mu, sigma)` on a vector returns one scalar. Every family function in this library loops explicitly (or uses a matrix-product identity) specifically to keep the `N x K` shape — don't try to shortcut them with Stan's built-in vectorized forms, you'll lose the per-observation structure the rest of the library needs.
- **Use `target +=`, not `~`.** As with `laplace-ts`, `cluster::mixture_lpdf(L, log_theta)` is a `target +=` call, not a sampling statement.
- **Unbounded likelihood spikes.** `spherical_loglik`, `diag_loglik`, and `mvn_chol_loglik` all send `L[n,k] -> +inf` if a component's scale shrinks to zero around a single point. Put a weakly informative prior (or a lower bound) on every scale parameter — `exponential(1)` on `sd` is a reasonable default.
- **Standardize before using distance-based families.** `sq_dist`, `kmeans_loglik`, and `spherical_loglik` are Euclidean and unit-sensitive. Center and scale continuous features first, or clusters will be dominated by whichever feature happens to have the largest numeric range.
- **Label switching.** The mixture likelihood is invariant to permuting components, so `mu_k`, `sd_k`, `theta_k` for a specific `k`, and anything from `assign_rng`/`map_assign`, are not safe to average across posterior draws without a fix. `mixture_pointwise` (for `loo`), `coclustering`, and `assignment_entropy` are safe as-is. An `ordered` constraint on one coordinate of `mu` (shown above) is the cheapest fix when clusters separate on that coordinate; post-hoc relabeling is the general one.
- **`complete_loglik` and `mixture_ss_lpdf` need the matching `log_theta`.** Both come in a `vector` and `matrix` overload — use whichever one built the `R`/`L` you're passing in, or the ICL identity ($\ell_n - \sum_k r_{nk}a_{nk} = H_n$) won't hold.
- **`_lpmf` requires an integer first argument.** Stan's naming convention enforces it at compile time. `multinomial_loglik` and `latent_class_loglik` take counts as `matrix` (so they aren't `_lpmf`-suffixed) specifically so their data can flow into matrix products without a cast; `lda_lpmf` takes `array[,] int Y` because LDA's likelihood only needs elementwise indexing, not a product with `Y` itself.
- **`multinomial_loglik` and `latent_class_loglik` are Naive Bayes.** Feed either one's `L` into `mixture_ss_lpdf` with partially known `z` for semi-supervised classification instead of pure clustering.
- **No Hidden Markov Model wrapper in 0.1.0.** Stan's `hmm_hidden_state_prob` and `hmm_latent_rng` require their inputs to be provably data-only (no autodiff support), which rules out passing them any `L` built from parameters — the usual case here. `hmm_marginal` (differentiable, usable for `target +=`) works fine with any of this library's `L` transposed to `K x N`, but per-draw state decoding has to be done outside Stan, from saved `L`, transition matrix, and initial-state draws. Not shipped as a wrapper here; do it in `generated quantities` by saving those and post-processing in R.
- **No unknown-K / infinite mixtures beyond stick-breaking.** `stick_breaking_log_weights` gives you a truncated Dirichlet-process prior at a `K` you still choose. A fully nonparametric sampler (reversible-jump, or a marginalized DP) isn't in this library.

## License

See [LICENSE](https://github.com/mlatinov/laplace-cluster/blob/main/LICENSE).