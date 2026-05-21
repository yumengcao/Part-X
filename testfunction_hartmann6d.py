import numpy as np
import math

def testfunction_hartmann6(X, r, mu=0.0, sigma=0.0):
    """
    Hartmann 6-dimensional (standard).
    Domain: [0,1]^6
    Returns list length r.
    """
    X = np.asarray(X, dtype=float)
    if X.size != 6:
        raise ValueError("Hartmann-6 is 6-dimensional")
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])
    inner = np.zeros(4, dtype=float)
    for i in range(4):
        inner[i] = np.sum(A[i] * (X - P[i])**2)
    val = -np.dot(alpha, np.exp(-inner))
    noise = np.random.normal(mu, sigma, int(r))
    return (val + noise + 0.724782).tolist()
