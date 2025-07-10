import numpy as np
from typing import Dict

def sample_allo_mother(scores: Dict[str, float], total_budget: int) -> Dict[str, int]:
    region_ids = list(scores.keys())
    region_scores = np.array([scores[k] for k in region_ids])
    region_amount = len(region_scores)

    base_rate = 2
    base_bud = max(1, total_budget // (base_rate * region_amount))
    remaining_budget = total_budget - base_bud * region_amount

    # Smoothed inverse score
    score_eps = 1e-4
    safe_scores = np.maximum(region_scores, score_eps)
    log_scores = np.log1p(safe_scores)
    smoothed_scores = 1.0 / log_scores
    prob = smoothed_scores / np.sum(smoothed_scores)

    # Initial allocation
    raw_alloc = prob * remaining_budget
    min_extra = 5
    max_extra = 30
    clipped = np.clip(raw_alloc, min_extra, max_extra)

    # Rescale to match total budget
    clipped = clipped / np.sum(clipped) * remaining_budget
    rounded = np.round(clipped).astype(int)

    # Fix rounding drift safely
    diff = remaining_budget - np.sum(rounded)
    while diff != 0:
        for i in range(len(rounded)):
            if diff == 0:
                break
            if diff > 0:
                rounded[i] += 1
                diff -= 1
            elif diff < 0 and rounded[i] > min_extra:
                rounded[i] -= 1
                diff += 1

    allocation = base_bud + rounded
    return {k: int(a) for k, a in zip(region_ids, allocation)}