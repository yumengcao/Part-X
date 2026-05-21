import numpy as np
from random import sample
import copy
from typing import List

def vol(bounds, dim):
    """Compute the volume of a region with given bounds."""
    volume = 1.0
    for low, up in bounds:
        volume *= up - low
    return volume

def undefined_vol(undefined_regions: dict, dim) -> float:
    """
    Compute the total volume of all undefined subregions in a given iteration.

    Args:
        undefined_regions (dict): Dictionary mapping subregion ID to region bounds.
                                  Format: {'r1_L1': [(low1, up1), (low2, up2), ...], ...}
        dim (int): Dimensionality of the region

    Returns:
        float: Total volume of all undefined subregions.
    """
    total_vol = 0.0
    for key, region in undefined_regions.items():
        if not isinstance(region, list) or not all(isinstance(b, tuple) and len(b) == 2 for b in region):
            raise ValueError(f"Malformed region {key}: {region}")
        if len(region) != dim:
            raise ValueError(f"Region {key} has dimension {len(region)} but expected {dim}")
        total_vol += vol(region, dim)
    return total_vol


def select_regions(sample: np.array, subregion: list, 
                   robustness: np.array, dim: int):
    '''
    select sample points with its corresponding robustness
    in the target region

    Input:
    sample: np.array : sample points
    subregion (list): target subregion
    roubstness (np.array): roubstness values
    dim (int): dimension
    Return:
    sample_select (np.array): selected samples
    robust_select (np,array): corresponding roubstness
    '''

    sample_select = []
    robust_select = []
    for i in range(len(sample)):
        tell = 0
        for j in range(dim):
            if tell == 0:
                if subregion[j][0] > sample[i][j] or  \
                    sample[i][j] > subregion[j][1]:
                    tell = 1
                else:
                    tell = 0
        if tell == 0:
            sample_select.append(sample[i])
            robust_select.append(robustness[i])
    return np.array(sample_select), robust_select




# def del_grouping(theta_plus_iter: dict, theta_minus_iter: dict, grouping: dict) -> dict:
    
#     for key in grouping['group1'].copy().keys():
#         if key in theta_minus_iter.keys():
#             del grouping['group1'][key]
    
#     for key in grouping['group6'].copy().keys():
#         if key in theta_plus_iter.keys():
#             del grouping['group6'][key]
    
#     return grouping
def extract_regions_and_parents_iter(tree: dict, iteration_key:str) -> tuple:
    """
    Extract regions and their parent names from the partitioning tree for a specific iteration.     
    Args:
        tree (dict): The partitioning tree structure.
        iteration_key (str): The key for the iteration level to extract regions from.
    Returns:
        tuple: A tuple containing two dictionaries:
            - regions: A dictionary mapping region names to their bounds.   
            - parents: A dictionary mapping region names to their parent names.         
    Raises:
        KeyError: If the iteration_key does not exist in the tree.  
    """
    if iteration_key not in tree:
        return {}, {}

    regions = {}
    parents = {}
    for parent_key, children in tree[iteration_key].items():
        parent_name = parent_key.replace('parent_', '')
        for region_name, bounds in children.items():
            regions[region_name] = bounds
            parents[region_name] = parent_name
    return regions, parents


def find_parent_region(tree: dict, target_region: str) -> str:
    
    """ Find the parent region str of a target region in the partitioning tree.     
    Args:
        tree (dict): The partitioning tree structure.
        target_region (str): The key of the target region to find its parent.
    Returns:
        str: The key of the parent region if found, otherwise None. 
    """
    
    for iter_key, iter_dict in tree.items():
        for parent_key, children_dict in iter_dict.items():
            for region_key in children_dict:
                if region_key == target_region:
                    return parent_key.replace('parent_', '')
    return None 

import numpy as np


def check_all_same_sign(y: np.ndarray, eps: float = 1e-8) -> str:
    """
    Determine whether all values in a NumPy array share the same sign.

    Parameters
    ----------
    y : np.ndarray
        One-dimensional NumPy array containing real-valued samples.
    eps : float, optional
        Tolerance threshold used to classify values as positive, negative,
        or approximately zero. Default is 1e-8.

    Returns
    -------
    str
        One of the following strings:
            - 'positive'  : all values are strictly greater than eps
            - 'negative'  : all values are strictly less than -eps
            - 'zero-only' : all values are within [-eps, eps]
            - 'mixed'     : values contain a mixture of signs

    Raises
    ------
    ValueError
        If the input array is empty.

    Notes
    -----
    A value is considered:
        positive if y > eps
        negative if y < -eps
        zero     if -eps <= y <= eps
    """

    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a NumPy array.")

    if y.size == 0:
        raise ValueError("Input array 'y' must not be empty.")

    positive_mask = y > eps
    negative_mask = y < -eps
    zero_mask = (~positive_mask) & (~negative_mask)

    if np.all(positive_mask):
        return "positive"

    if np.all(negative_mask):
        return "negative"

    if np.all(zero_mask):
        return "zero-only"

    return "mixed"


def estimate_true_quantile(f_true, lo, hi, delta=0.1, grid_per_dim=200, mc_samples=200000, dim_threshold=3, seed=123):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = len(lo)
    if d <= dim_threshold:
        # grid
        counts = [grid_per_dim] * d
        axes = [np.linspace(lo[i], hi[i], num=counts[i]) for i in range(d)]
        mesh = np.meshgrid(*axes, indexing='xy')
        pts = np.stack([m.ravel() for m in mesh], axis=1)
        vals = f_true(pts)
        vals = np.asarray(vals).ravel()
        q = float(np.quantile(vals, delta))
        return q
    else:
        rng = np.random.RandomState(seed)
        pts = rng.uniform(low=lo, high=hi, size=(int(mc_samples), d))
        vals = f_true(pts)
        vals = np.asarray(vals).ravel()
        q = float(np.quantile(vals, delta))
        return q

