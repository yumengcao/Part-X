import numpy as np
import math

def testfunction_shifted_ackley(X, r, shift=None, mu=0.0, sigma=0.0):
    """
    Shifted Ackley: computes Ackley at (X - shift).
    shift: None -> fixed shift of +5 on every coordinate; or provide array-like of same dimension.
    Recommended domain (original X before shift): [-32.768, 32.768]^d
    """
    X = np.asarray(X, dtype=float)
    d = X.size
    if shift is None:
        shift_vec = np.full(d, 5.0)   # default shift +5
    else:
        shift_vec = np.asarray(shift, dtype=float)
        if shift_vec.size != d:
            raise ValueError("shift must have same dimension as X")

    Xs = X - shift_vec
    # reuse ackley formula
    a = 20.0
    b = 0.2
    c = 2.0 * math.pi

    sum_sq = np.sum(Xs**2)
    sum_cos = np.sum(np.cos(c * Xs))

    term1 = -a * np.exp(-b * math.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    val = term1 + term2 + a + math.e

    noise = np.random.normal(mu, sigma, int(r))
    return (val + noise - 17.7117 ).tolist()
#17.7117   2d
# 19.3184 3d

#20.3193 6d