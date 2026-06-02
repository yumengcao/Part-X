"""Simplified Part-X style partitioning and stochastic search algorithm for the demo."""
from typing import Dict, List, Optional
import numpy as np
from .region import Region, create_root_region
from .sampler import sample_regions
from .estimator import estimate_region_falsification_probability, estimate_global_falsification_probability, compute_confidence_interval
from .visualization import plot_iteration
from .classification import classify_region


class PartX:
    """A small, simplified Part-X-like controller.

    This implementation is intentionally simple and interpretable for educational use.
    """

    def __init__(
        self,
        eval_fn,
        rng: Optional[np.random.Generator] = None,
        initial_samples: int = 100,
        samples_per_region: int = 20,
        max_iters: int = 6,
        min_samples_per_region: int = 10,
        out_fig_dir: str = "figures",
    ):
        self.eval_fn = eval_fn
        self.rng = rng or np.random.default_rng(0)
        self.initial_samples = initial_samples
        self.samples_per_region = samples_per_region
        self.max_iters = max_iters
        self.min_samples_per_region = min_samples_per_region
        self.out_fig_dir = out_fig_dir

        self.regions: Dict[str, Region] = {}
        self.iteration = 0

    def initialize(self):
        root = create_root_region()
        self.regions = {root.region_id: root}
        # initial sampling spread across root
        sample_regions([root], n_per_region=self.initial_samples, rng=self.rng, eval_fn=self.eval_fn)

    def update_region_statistics(self):
        # placeholder in this simplified demo; estimators live in estimator.py
        pass

    def select_region_to_refine(self) -> Optional[Region]:
        """Select a region to refine using a simple heuristic: choose the region with the highest
        upper confidence estimate for falsification (p_hat + uncertainty).
        """
        best = None
        best_score = -1.0
        # compute for each region
        for r in self.regions.values():
            p_hat, ci = estimate_region_falsification_probability(r)
            # uncertainty measured as CI half-width
            unc = 0.5 * (ci[1] - ci[0])
            score = p_hat + unc
            # prefer regions with some samples or moderate volume
            if score > best_score:
                best_score = score
                best = r
        return best

    def step(self):
        self.iteration += 1
        # choose region
        to_refine = self.select_region_to_refine()
        if to_refine is None:
            return
        # split
        children = to_refine.split()
        # remove parent region from active set and add children
        del self.regions[to_refine.region_id]
        for c in children:
            self.regions[c.region_id] = c
        # sample children
        sample_regions(children, n_per_region=self.samples_per_region, rng=self.rng, eval_fn=self.eval_fn)
        # optional: also sample other regions with low samples
        for r in list(self.regions.values()):
            if r.samples is None or len(r.samples) < self.min_samples_per_region:
                sample_regions([r], n_per_region=self.min_samples_per_region, rng=self.rng, eval_fn=self.eval_fn)

        # Reclassification step: compute CI and update labels
        for r in self.regions.values():
            if r.values is None or len(r.values) == 0:
                # no data => remain 'r'
                r.label = 'r'
                continue
            n = len(r.values)
            k = int((r.values < 0.0).sum())
            ci_low, ci_high = compute_confidence_interval(k, n)
            # For demo purposes, interpret CI relative to 0 probability threshold
            # If lower bound > 0 => likely falsifying; if upper bound < 0 => likely safe
            # Note: our CI here is for falsification probability (0..1), but robustness sign check
            # is used in original code; for simplicity, we map small p to '+' (safe), large p to '-'.
            # Use thresholds: if ci_high < 0.01 -> safe ('+'); if ci_low > 0.5 -> unsafe ('-'); else 'r'
            new_label = 'r'
            reclass_tag = None
            # simple mapping: high probability -> '-' , low probability -> '+'
            if ci_high < 0.01:
                new_label = '+'
            elif ci_low > 0.5:
                new_label = '-'
            # update
            if r.label != new_label:
                # call classify_region to get tag
                new_label, reclass_tag = classify_region(ci_low, ci_high, r.label)
                r.label = new_label
                if reclass_tag is not None:
                    # print short reclass info for trace
                    print(f"reclass {r.region_id}: {reclass_tag}")

        # save figure
        savepath = f"{self.out_fig_dir}/iteration_{self.iteration}.png"
        plot_iteration(savepath, list(self.regions.values()), iteration=self.iteration)

    def run(self):
        self.initialize()
        # save initial figure
        plot_iteration(f"{self.out_fig_dir}/iteration_0.png", list(self.regions.values()), iteration=0)
        for _ in range(self.max_iters):
            self.step()

    def summary(self) -> dict:
        total_samples = sum(len(r.samples) if r.samples is not None else 0 for r in self.regions.values())
        p_global, estimates = estimate_global_falsification_probability(self.regions.values())
        min_rho = float("inf")
        any_falsifier = False
        for r in self.regions.values():
            if r.values is not None and len(r.values) > 0:
                min_rho = min(min_rho, float(r.values.min()))
                if (r.values < 0.0).any():
                    any_falsifier = True
        if min_rho == float("inf"):
            min_rho = None
        return {
            "total_samples": int(total_samples),
            "p_global": float(p_global),
            "n_regions": int(len(self.regions)),
            "min_robustness": min_rho,
            "any_falsifier": bool(any_falsifier),
        }
