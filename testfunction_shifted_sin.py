

import numpy as np
import math

def testfunction_shifted_sin_nd(X, r, mu=0.0, sigma=0.0):

    X = np.asarray(X, dtype=float).ravel()
    d = X.size
    total = 0.0

    i = 0
    while i + 1 < d:
        xk = float(X[i])
        xk1 = float(X[i+1])

        term = -2.5 * math.sin(math.pi*(xk + 60.0)/180.0) * \
               math.sin(math.pi*(xk1 + 60.0)/180.0) \
               - math.sin(math.pi*(xk + 60.0)/36.0) * \
               math.sin(math.pi*(xk1 + 60.0)/36.0)

        total += term
        i += 2

    if d % 2 == 1:
        xk = float(X[-1])
        term = -2.5 * math.sin(math.pi*(xk + 60.0)/180.0) * \
               math.sin(math.pi*(xk + 60.0)/180.0) \
               - math.sin(math.pi*(xk + 60.0)/36.0) * \
               math.sin(math.pi*(xk + 60.0)/36.0)
        total += term

    
    shift_const = 5.096464
    total = total + shift_const

    if r is None or int(r) <= 0:
        return [float(total)]

    noise = np.random.normal(loc=mu, scale=sigma, size=int(r))
    return (total + noise).tolist()