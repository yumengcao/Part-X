import numpy as np
import math

def testfunction_branin(X, r, mu=0.0, sigma=0.0):
    """
    Branin function (2D).
    X: length-2 array-like [x1, x2]
    r: replications
    Returns list length r.
    Recommended domain: x1 in [-5, 10], x2 in [0, 15]
    """
    X = np.asarray(X, dtype=float)
    if X.size != 2:
        raise ValueError("Branin is 2-dimensional")
    x1, x2 = float(X[0]), float(X[1])

    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r_ = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    val = a * (x2 - b * x1**2 + c * x1 - r_)**2 + s * (1 - t) * math.cos(x1) + s

    noise = np.random.normal(mu, sigma, int(r))
    return (val + noise- 5.93727).tolist()
