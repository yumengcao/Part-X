

def __allo__b_uni__(total_budget_iter: int) -> tuple:
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
    
    base_ratio = 0.2
    n_unif_iter = int(round(base_ratio * total_budget_iter))
    n_bo_iter = total_budget_iter - n_unif_iter

    # Ensure neither value is zero
    if n_unif_iter == 0 or n_bo_iter == 0:
        raise ValueError(f"Budget too small ({total_budget_iter}) to divide between uniform and BO sampling.")

    return n_bo_iter, n_unif_iter