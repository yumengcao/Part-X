import numpy as np
from typing import Callable, List

def uniform_sampling(subregion: list, dim: int, number: int) -> np.ndarray:
    '''
    Uniform sampling method for bounded subregion.

    Parameters:
        subregion (list): list of [lower, upper] bounds for each dimension
        dim (int): number of dimensions
        number (int): number of samples to generate

    Returns:
        np.ndarray: array of shape (number, dim)
    '''
    if len(subregion) != dim:
        raise ValueError("Dimension of subregion does not match 'dim'")
    samples = np.empty((number, dim))
    for i in range(dim):
        low, high = subregion[i]
        assert low < high, f"Invalid subregion bounds: {low} >= {high} in dimension {i}"
        samples[:, i] = np.random.uniform(low, high, size=number)
    return samples
    





def robustness_values(sample: np.ndarray, test_function: Callable[[np.ndarray], float]) -> List[float]:
    """
    Compute robustness values for a batch of samples.

    Parameters:
        sample (np.ndarray): shape (n_samples, dim)
        test_function (Callable): function that accepts a 1D array and returns a float

    Returns:
        List[float]: robustness values
    """
    return [test_function(x) for x in sample]