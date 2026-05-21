import numpy as np
import math

def testfunction_ackley(X, r, mu=0.0, sigma=0.0):
    """
    Generic n-dimensional Ackley.
    X: sequence-like length d
    r: number of replications (returns list length r)
    mu, sigma: Gaussian noise parameters (default deterministic if sigma=0)
    Returns: list of length r
    Recommended domain: [-32.768, 32.768]^d
    """
    X = np.asarray(X, dtype=float)
    d = X.size
    a = 20.0
    b = 0.2
    c = 2.0 * math.pi

    sum_sq = np.sum(X**2)
    sum_cos = np.sum(np.cos(c * X))

    term1 = -a * np.exp(-b * math.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    val = term1 + term2 + a + math.e

    noise = np.random.normal(mu, sigma, int(r))
    return (val + noise - 19.3185).tolist()##2d 17.7316 3d 19.3185