import numpy as np


def testfunction_hartmann3(X, r, mu=0.0, sigma=0.0):
    
    """
    Hartmann 3-dimensional (standard).
    Domain: [0,1]^3
    Returns list length r.
    """
    X = np.asarray(X, dtype=float)
    if X.size != 3:
        raise ValueError("Hartmann-3 is 3-dimensional")
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0],
                  [3.0, 10.0, 30.0],
                  [0.1, 10.0, 35.0]])
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [381, 5743, 8828]])
    # compute
    inner = np.zeros(4, dtype=float)
    for i in range(4):
        inner[i] = np.sum(A[i] * (X - P[i])**2)
    val = -np.dot(alpha, np.exp(-inner))
    # standard Hartmann3 is negative; that's fine.
    noise = np.random.normal(mu, sigma, int(r))
    return (val + noise+2.49781).tolist()
