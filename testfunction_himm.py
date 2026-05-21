# testfunction_himm.py
import numpy as np

# Interface: fun(X, r=1) -> list-like of length r OR scalar
# X is a list/array-like of length d (d can be any positive int).
def fun(X, r=1):
    """
    Example: Himmelblau-like shifted example for 2D.
    Returns list of length r (to match PBnB style).
    """
    x = float(X[0])
    y = float(X[1])
    val = (x**2 + y - 11)**2 + (x + y**2 - 7)**2 - 40
    # zero-noise
    if r == 1:
        return [val]
    else:
        # replicate same deterministic value r times
        return [val for _ in range(r)]