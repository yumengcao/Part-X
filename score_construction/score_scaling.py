
import numpy as np


def score_scale(avg_mu_iter, score, k):
    
    """calculate score for each region based on the average statistics.

    Args:
        avg_mu_iter (np.array): containing the post average mu for each region in iteration k.
        score(np.array): containing the unscaled score for each region in iteration k.
        k (int): the number of iterations.
    Returns:
        score_scaled: containing the score for each region in iteration k, 
        scaled by the maximum absolute value of the average mu for positive and negative regions.
    
    """ 
    mu_max_pos = np.max([abs(avg_mu_iter[m]) for m in avg_mu_iter if m>0]) if np.any(avg_mu_iter >0) else 1
    mu_max_neg = np.max([abs(avg_mu_iter[m]) for m in avg_mu_iter if m<0]) if np.any(avg_mu_iter <0) else 1
    scaling_iter = np.array([mu_max_pos if m>=0 else mu_max_neg for m in avg_mu_iter])
    
    
    score_sacled = score/ scaling_iter
    
    return score_sacled