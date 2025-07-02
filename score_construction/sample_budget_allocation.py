import numpy as np


def sample_allo_mother(scores, total_budget, region_amount):
    """sllocate sample budget to nother node based on the scores.
    Args:
        scores (np.array): input array of scores for each region.
        total_budget (int_): sample budget in iteration k.
        region_amount(int): number of subregions in iteration k.
    Returns:
        allocation (np.array): sampled budget allocation for each region.
    """
    base_rate = 5
    base_bud = max(1, total_budget // (base_rate * region_amount))
    if np.sum(scores) == 0:
        allo_prob= np.full_like(scores, (total_budget- base_bud*region_amount) // len(scores))
    else:
        prob = (1/scores) / np.sum(1/scores) 
        allo_prob = np.random.multinomial(total_budget- base_bud*region_amount, prob)
        
    allocation =  base_bud+ allo_prob
        
    return allocation

def sample_allo_child(allocation, branch_num, seed = None, mode='random'):
    """allocate sample budget from mother to children node

    Args:
        allocation (np.array): allocation for each region in ieration k
        branch_num np.array): branching number for each subnode(k) in iteration k+1.
    
    Returns:
        allocation_child （np.array): allocation for each subnode in iteration k+1.
    """
    if seed is not None:
        np.random.seed(seed)

    allocation_child = []
    for parent_n, child_n in zip(allocation, branch_num):
        base = parent_n // child_n
        rem = parent_n % child_n

        alloc = np.full(child_n, base)

        if mode == 'random' and rem > 0:
            indices = np.random.choice(child_n, size=rem, replace=False)
        else:
            indices = np.arange(rem)

        alloc[indices] += 1
        allocation_child.append(alloc)

    return np.concatenate(allocation_child)