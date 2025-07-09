import numpy as np
from typing import Dict


import numpy as np
from typing import Dict

def sample_allo_mother(scores: Dict[str, float], total_budget: int) -> Dict[str, int]:
    region_ids = list(scores.keys())
    region_scores = np.array([scores[k] for k in region_ids])
    region_amount = len(region_scores)
    
    # Base rate component
    base_rate = 2
    base_bud = max(1, total_budget // (base_rate * region_amount))
    remaining_budget = total_budget - base_bud * region_amount

    # Score normalization: avoid very small scores dominating
    score_eps = 1e-4
    safe_scores = np.maximum(region_scores, score_eps)
    
    # Use 1 / log(score + 1) to reduce sensitivity to tiny scores
    inverse_scores = 1.0 / (np.log(safe_scores + 1.0))
    
    # Normalize to probabilities
    prob = inverse_scores / np.sum(inverse_scores)

    # Allocate remaining samples
    extra_allocation = np.random.multinomial(remaining_budget, prob)
    allocation = base_bud + extra_allocation

    return {k: int(a) for k, a in zip(region_ids, allocation)}