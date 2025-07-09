import numpy as np

def merge_parent_samples_to_child(
    samples_x_all: dict,
    samples_y_all: dict,
    iteration: int,
    child_region: str,
    parent_region: str,
    child_bounds: list,
    x_child: np.ndarray,
    y_child: list
):
    """
    Merge parent samples falling in child region into current child's data.

    Returns:
        updated samples_x_all, samples_y_all
    """
    iter_name = f'iter_{iteration + 1}'
    if iter_name not in samples_x_all:
        samples_x_all[iter_name] = {}
        samples_y_all[iter_name] = {}

    # If first iteration, no parent to merge from
    if iteration == 0:
        samples_x_all[iter_name][child_region] = x_child
        samples_y_all[iter_name][child_region] = np.array(y_child)
        return samples_x_all, samples_y_all

    prev_iter_name = f'iter_{iteration}'
    x_parent = samples_x_all.get(prev_iter_name, {}).get(parent_region)
    y_parent = samples_y_all.get(prev_iter_name, {}).get(parent_region)

    if x_parent is None or y_parent is None:
        # Ensure current child at least holds its own samples
        samples_x_all[iter_name][child_region] = x_child
        samples_y_all[iter_name][child_region] = np.array(y_child)
        return samples_x_all, samples_y_all

    dim = len(child_bounds)
    mask = np.ones(len(x_parent), dtype=bool)
    for d in range(dim):
        low, high = child_bounds[d]
        mask &= (x_parent[:, d] >= low) & (x_parent[:, d] <= high)

    x_filtered = x_parent[mask]
    y_filtered = y_parent[mask]

    if len(x_filtered) == 0:
        samples_x_all[iter_name][child_region] = x_child
        samples_y_all[iter_name][child_region] = np.array(y_child)
        return samples_x_all, samples_y_all

    # Combine and deduplicate
    x_combined = np.vstack([x_child, x_filtered])
    y_combined = np.concatenate([y_child, y_filtered])

    unique_x, indices = np.unique(x_combined, axis=0, return_index=True)
    samples_x_all[iter_name][child_region] = unique_x
    samples_y_all[iter_name][child_region] = y_combined[indices]

    return samples_x_all, samples_y_all