
import numpy as np
from typing import Dict

def sample_allo_mother(scores: Dict[str, float], total_budget: int, last_iteration: int) -> Dict[str, int]:
    region_ids = list(scores.keys())
    region_scores = np.array([scores[k] for k in region_ids], dtype=float)
    n = len(region_scores)

    if n == 0 or total_budget <= 0:
        return {}

    # priority: smaller score -> higher priority
    score_eps = 1e-4
    safe_scores = np.maximum(region_scores, score_eps)
    priority = 1.0 / np.log1p(safe_scores)
    priority = np.where(np.isfinite(priority), priority, 0.0)

    if np.sum(priority) <= 0:
        priority = np.ones(n, dtype=float) / n
    else:
        priority = priority / np.sum(priority)

    # Case 1: budget smaller than number of regions
    # impossible to give every region at least 1 sample
    if total_budget < n:
        alloc = np.zeros(n, dtype=int)
        top_idx = np.argsort(-priority)[:total_budget]
        alloc[top_idx] = 1
        return {k: int(a) for k, a in zip(region_ids, alloc)}

    # Base allocation
    if last_iteration == 0:
        base_bud = max(1, total_budget // (2 * n))
    else:
        base_bud = 1

    base_bud = min(base_bud, total_budget // n)
    allocation = np.full(n, base_bud, dtype=int)

    remaining_budget = total_budget - np.sum(allocation)

    # Dynamic minimum extra
    min_extra = 5
    min_extra_eff = min(min_extra, remaining_budget // n)

    # If not enough, let it go to zero
    if min_extra_eff < 0:
        min_extra_eff = 0

    # Give each region a soft minimum extra if feasible
    extra = np.full(n, min_extra_eff, dtype=int)
    remaining_budget -= np.sum(extra)

    if remaining_budget < 0:
        # fallback: reduce extras
        extra[:] = 0
        remaining_budget = total_budget - np.sum(allocation)

    # Distribute the remaining budget by priority
    if remaining_budget > 0:
        raw = priority * remaining_budget
        rounded = np.floor(raw).astype(int)
        frac = raw - rounded

        # cap extra per region
        max_extra = 30
        rounded = np.minimum(rounded, max_extra)

        diff = remaining_budget - np.sum(rounded)
        if diff > 0:
            order = np.argsort(-frac)
            for idx in order:
                if diff == 0:
                    break
                if rounded[idx] < max_extra:
                    rounded[idx] += 1
                    diff -= 1

        allocation = allocation + extra + rounded
    else:
        allocation = allocation + extra

    return {k: int(a) for k, a in zip(region_ids, allocation)}

# def sample_allo_mother(scores: Dict[str, float], total_budget: int, last_iteration: int) -> Dict[str, int]:
#     region_ids = list(scores.keys())
#     region_scores = np.array([scores[k] for k in region_ids])
#     region_amount = len(region_scores)
    
#     base_rate = 2
#     if last_iteration == 0:
#         base_bud = max(1, total_budget // (base_rate * region_amount))
#     else:
#         base_bud = 1
#     remaining_budget = total_budget - base_bud * region_amount
    
#     # Smoothed inverse score
#     score_eps = 1e-4
#     safe_scores = np.maximum(region_scores, score_eps)
#     log_scores = np.log1p(safe_scores)
#     smoothed_scores = 1.0 / log_scores
#     prob = smoothed_scores / np.sum(smoothed_scores)

#     # Initial allocation
#     raw_alloc = prob * remaining_budget
#     min_extra = 5
#     max_extra = 30
#     clipped = np.clip(raw_alloc, min_extra, max_extra)

#     # Rescale to match total budget
#     clipped = clipped / np.sum(clipped) * remaining_budget
#     rounded = np.round(clipped).astype(int)

#     # Fix rounding drift safely
#     diff = remaining_budget - np.sum(rounded)
#     while diff != 0:
#         for i in range(len(rounded)):
#             if diff == 0:
#                 break
#             if diff > 0:
#                 rounded[i] += 1
#                 diff -= 1
#             elif diff < 0 and rounded[i] > min_extra:
#                 rounded[i] -= 1
#                 diff += 1

#     allocation = base_bud + rounded
#     return {k: int(a) for k, a in zip(region_ids, allocation)}