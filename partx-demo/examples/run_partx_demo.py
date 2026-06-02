"""Example runner for the Part-X demo."""
import os
import sys
from pathlib import Path
import numpy as np

# Ensure the package root (partx-demo) is on sys.path so `import src` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.partx import PartX
from src.benchmark import synthetic_robustness_function


def main():
    out_dir = Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    px = PartX(eval_fn=synthetic_robustness_function, rng=rng, initial_samples=60, samples_per_region=30, max_iters=6, out_fig_dir=str(out_dir))
    px.run()
    summary = px.summary()
    print("Run summary:")
    print(f"  total samples: {summary['total_samples']}")
    print(f"  estimated global falsification probability: {summary['p_global']:.4f}")
    print(f"  number of regions: {summary['n_regions']}")
    print(f"  minimum robustness observed: {summary['min_robustness']}")
    print(f"  any falsifier found: {summary['any_falsifier']}")


if __name__ == "__main__":
    main()
