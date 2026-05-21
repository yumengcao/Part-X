import numpy as np


def compute_tau_rosenbrock_nd(
    delta,
    domain,
    n_samples=200_000,
    random_seed=None,
    return_ci=False,
    ci_alpha=0.05,
    ci_bootstrap_B=1000
):
    """
    Compute tau = y(delta, S) for n-D Rosenbrock on a hyper-rectangular domain S.

    Parameters
    ----------
    delta : float
        Quantile level in (0,1) e.g. 0.1
    domain : sequence of length 2*n
        [a0, b0, a1, b1, ..., a_{n-1}, b_{n-1}] or array shaped (n,2)
    n_samples : int
        Number of uniform samples in S (default 200k). Increase for higher dims.
    random_seed : int or None
        RNG seed for reproducibility
    return_ci : bool
        If True, compute bootstrap percentile CI for tau
    ci_alpha : float
        significance level for CI (default 0.05 -> 95% CI)
    ci_bootstrap_B : int
        number of bootstrap resamples (default 1000)

    Returns
    -------
    result : dict with keys
      'tau' : float, empirical quantile
      'empirical_fraction' : float, fraction of sample with f<=tau (should ~ delta)
      'tau_ci' : (lower, upper) if return_ci True else None
      'n_samples' : int
      'delta' : float
    """
    assert 0.0 < delta < 1.0
    domain = np.asarray(domain)
    if domain.ndim == 1 and domain.size % 2 == 0:
        n = domain.size // 2
        domain = domain.reshape(n, 2)
    elif domain.ndim == 2 and domain.shape[1] == 2:
        n = domain.shape[0]
    else:
        raise ValueError("domain must be length-2n list or (n,2) array: [a0,b0,a1,b1,...]")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Uniform sampling in hyper-rectangle
    lows = domain[:, 0]
    highs = domain[:, 1]
    # draw (n_samples, n)
    U = np.random.rand(n_samples, n)
    samples = lows + U * (highs - lows)  # broadcasting

    # evaluate
    fvals = rosenbrock_nd(samples)  # shape (n_samples,)

    # empirical quantile index (1-based m = ceil(delta * N))
    m = int(np.ceil(delta * n_samples))
    k = m - 1
    tau = float(np.partition(fvals, k)[k])

    empirical_fraction = float(np.mean(fvals <= tau))

    tau_ci = None
    if return_ci:
        # percentile bootstrap on quantile: resample indices with replacement
        B = ci_bootstrap_B
        q = int(np.round(delta * 100))
        boot_qs = []
        rng = np.random.randint  # alias
        n = n_samples
        for _ in range(B):
            idx = rng(0, n, size=n)
            boot_vals = fvals[idx]
            boot_q = np.percentile(boot_vals, 100.0 * delta)
            boot_qs.append(boot_q)
        lower = float(np.percentile(boot_qs, 100.0 * (ci_alpha / 2.0)))
        upper = float(np.percentile(boot_qs, 100.0 * (1.0 - ci_alpha / 2.0)))
        tau_ci = (lower, upper)

    return {
        'tau': tau,
        'empirical_fraction': empirical_fraction,
        'tau_ci': tau_ci,
        'n_samples': n_samples,
        'delta': delta
    }


# --------------------------
# Example usage (3D Rosenbrock)
# --------------------------
if __name__ == "__main__":
    # Example: 3D Rosenbrock domain: commonly use [-2,2] for each dimension (adjust if needed)
    n_dim = 3
    domain_vec = []
    for i in range(n_dim):
        domain_vec.extend([-2.0, 2.0])  # a_i, b_i for each dim
    delta = 0.10

    res = compute_tau_rosenbrock_nd(delta, domain_vec, n_samples=300000, random_seed=123, return_ci=True, ci_bootstrap_B=500)
    print("delta =", res['delta'])
    print("tau  =", res['tau'])
    print("empirical fraction (<=tau) =", res['empirical_fraction'])
    if res['tau_ci'] is not None:
        print("bootstrap 95% CI for tau =", res['tau_ci'])

    # Example g(x) function for algorithm usage (single point)
    def g_nd(x_point, tau_value):
        """
        x_point: array-like shape (n,)
        returns scalar g(x) = f(x) - tau
        """
        return rosenbrock_nd(np.asarray(x_point)) - tau_value

    # Example check:
    # x_test = np.zeros(n_dim)  # some test point
    # print("g(x_test) =", g_nd(x_test, res['tau']))