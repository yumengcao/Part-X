import numpy as np

def score_based_partition(scores: dict, iteration: int) -> tuple:
    """
    Decide how to partition each region based on its score.
 
    Args:
        scores (dict): region_id -> score (float)
        iteration (int): current iteration number
    Returns:
        tuple:
            - split_counts (dict): region_id -> number of subregions (1, 2, or 3)
            - total_subregions (int): total number of subregions across all regions
    """
    split_counts = {}
    

    if iteration < 4:
        # For early iterations, always split into 2 subregions
        for region in scores:
            split_counts[region] = 2
            
    else:
        score_values = np.array(list(scores.values()))
        tau1 = np.quantile(score_values, 0.2)
        tau2 = np.quantile(score_values, 0.8)

        for region, score in scores.items():
            if score <= tau1:
                count = 3
            #elif score <= tau2:
               # count = 2
            else:
                count = 2
            split_counts[region] = count
    total_subregions = len(split_counts)       
    return split_counts, total_subregions