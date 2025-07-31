

def allo__b_uni__(total_budget_iter: int) -> tuple:
    """
    Allocate the sampling budget between uniform sampling and Bayesian optimization.

    Args:
        total_budget_iter (int): Total number of samples available for this iteration.

    Returns:
        tuple: (n_bo_iter, n_unif_iter) such that both are positive integers and sum to total_budget_iter.

    Raises:
        ValueError: If input budget is invalid or allocation is not feasible.
    """
    if total_budget_iter <= 0:
        raise ValueError("Total budget must be a positive integer.")
    
    base_ratio = 0.4
    n_unif_iter = int(round(base_ratio * total_budget_iter))
    n_bo_iter = total_budget_iter - n_unif_iter

    # Ensure neither value is zero
    if n_unif_iter == 0 or n_bo_iter == 0:
        raise ValueError(f"Budget too small ({total_budget_iter}) to divide between uniform and BO sampling.")

    return n_bo_iter, n_unif_iter



def allo_new (total_budget_iter: int) -> tuple:
    """
    Allocate the sampling budget between uniform sampling and Bayesian optimization (BO),
    ensuring BO gets a power of 2 sample count for Sobol sequence.

    Args:
        total_budget_iter (int): Total number of samples available for this iteration.

    Returns:
        tuple: (n_bo_iter, n_unif_iter)
    """
    if total_budget_iter <= 2:
        raise ValueError("Total budget must be at least 3 to allow allocation.")

    def next_lower_power_of_two(n):
        return 1 << (n.bit_length() - 1)

    # Increase uniform ratio slightly to make space for bo being power of 2
    unif_ratio = 0.4
    n_unif_iter = int(round(unif_ratio * total_budget_iter))
    remaining = total_budget_iter - n_unif_iter

    n_bo_iter = next_lower_power_of_two(remaining)

    # Adjust unif again to fill up leftover
    n_unif_iter = total_budget_iter - n_bo_iter

    if n_unif_iter <= 0 or n_bo_iter <= 0:
        raise ValueError("Cannot allocate budget: resulting parts not positive.")

    return n_bo_iter, n_unif_iter