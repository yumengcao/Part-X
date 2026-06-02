"""Visualization utilities for Part-X demo."""
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .benchmark import make_grid, synthetic_robustness_function


def plot_partitions(ax, regions: Iterable, facealpha: float = 0.15):
    """Draw rectangular partitions on the given axes."""
    for r in regions:
        (x0, x1), (y0, y1) = r.bounds
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0, edgecolor="k", facecolor="none")
        ax.add_patch(rect)


def plot_iteration(savepath: str, regions: Iterable, iteration: int, grid_n: int = 200):
    """Create and save a figure for the current iteration showing:
    - true robustness contour
    - zero contour (falsification boundary)
    - sampled points
    - rectangular partitions
    - region shading by estimated falsification rate (simple color mapping)
    """
    X, Y, Z = make_grid(grid_n)
    fig, ax = plt.subplots(figsize=(6, 6))
    # contour of robustness
    cs = ax.contourf(X, Y, Z, levels=30, cmap="RdYlBu_r", alpha=0.9)
    ax.contour(X, Y, Z, levels=[0.0], colors="k", linewidths=1.0)

    # draw regions and samples
    max_p = 0.0
    for r in regions:
        p = 0.0
        if r.values is not None and len(r.values) > 0:
            p = float((r.values < 0.0).mean())
        max_p = max(max_p, p)

    for r in regions:
        (x0, x1), (y0, y1) = r.bounds
        p = 0.0
        if r.values is not None and len(r.values) > 0:
            p = float((r.values < 0.0).mean())
        # color mapping: redder for higher p
        color = (1.0, 1.0 - p / max(1e-6, max_p), 1.0 - p / max(1e-6, max_p)) if max_p > 0 else (0.8, 0.8, 0.8)
        # edge color based on label
        edge = "k"
        if hasattr(r, "label"):
            if r.label == "+":
                edge = "green"
            elif r.label == "-":
                edge = "red"
            else:
                edge = "k"
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0, edgecolor=edge, facecolor=color, alpha=0.25)
        ax.add_patch(rect)
        # plot samples
        if r.samples is not None and len(r.samples) > 0:
            pts = r.samples
            vals = r.values
            ax.scatter(pts[vals >= 0, 0], pts[vals >= 0, 1], c="blue", s=10, alpha=0.6)
            ax.scatter(pts[vals < 0, 0], pts[vals < 0, 1], c="red", s=20, alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Part-X Demo — Iteration {iteration}")
    fig.colorbar(cs, ax=ax, label="Robustness")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
