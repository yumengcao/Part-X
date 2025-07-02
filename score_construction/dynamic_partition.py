import numpy as np

def score_based_partition(scores):
    """
    based on the scores of each region, determine how to split the regions.

    args:
        scores: np.array of shape [n_regions]: scores for each region
        

   return:
        split_counts: np.array, each element indicates how many subregions the region will be split into.
     
    """
    tau1 = np.quantile(scores, 0.3)
    tau2 = np.quantile(scores, 0.7)

    split_counts = np.zeros_like(scores, dtype=int)
 

    for i, score in enumerate(scores):
        if score <= tau1:
            split_counts[i] = 3
        elif score <= tau2:
            split_counts[i] = 2
        else:
            split_counts[i] = 1


    return split_counts