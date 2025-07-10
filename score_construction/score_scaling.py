

def score_scale(avg_mu_iter: dict, score: dict, ) -> dict:
    """
    Scale scores per region based on sign-specific max absolute mu values.

    Args:
        avg_mu_iter (dict): region_id -> average mu in iteration k
        score (dict): region_id -> raw score in iteration k

    Returns:
        dict: region_id -> scaled score
    """

    mu_vals = list(avg_mu_iter.values())
    mu_max_pos = max([abs(mu) for mu in mu_vals if mu > 0], default=1)
    mu_max_neg = max([abs(mu) for mu in mu_vals if mu < 0], default=1)

 
    score_scaled = {}
    for region in score:
        mu = avg_mu_iter.get(region, 0)
        scale = mu_max_pos if mu >= 0 else mu_max_neg
        score_scaled[region] = score[region] / scale

    return score_scaled