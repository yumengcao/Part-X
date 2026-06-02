"""Synthetic benchmark helpers for the Part-X demo.

Provides a 2D smooth robustness function and a grid helper for plotting.
"""
from typing import Tuple
import numpy as np


def synthetic_robustness_function(X: np.ndarray) -> np.ndarray:
    """Evaluate the synthetic robustness function.

    X should be shape (n,2) or (2,) representing points in [0,1]^2.

    Returns an array of robustness values (negative = falsifying).
    """
    X = np.asarray(X)
    single = False
    if X.ndim == 1:
        X = X.reshape(1, -1)
        single = True
    x1 = X[:, 0]
    x2 = X[:, 1]
    rho = 0.25 - ((x1 - 0.7) ** 2 / 0.05 + (x2 - 0.35) ** 2 / 0.03)
    rho += 0.1 * np.sin(8 * x1) * np.cos(6 * x2)
    if single:
        return rho[0]
    return rho


def make_grid(n: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a grid over [0,1]^2 and evaluate the robustness function on it.

    Returns (X, Y, Z) suitable for contour plotting.
    """
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, ys)
    points = np.column_stack([X.ravel(), Y.ravel()])
    Z = synthetic_robustness_function(points).reshape(X.shape)
    return X, Y, Z
