import numpy as np

def uniform_sampling(subregion: list,  dim:int, number: int)-> np.array:
    '''
    unifrom sampling method
    '''
    samples = [[]*number]*dim
    for i in range(dim):
        samples[i] = np.random.uniform(subregion[i][0],subregion[i][1],number)
    sample_arr = np.array(samples)
    sample = sample_arr.T
    return sample



def robustness_values(sample: np.array, test_function: callable)-> list:
    '''
    corresponding robustness values
    '''
    Y = []
    for i in range(len(sample)):
        Y.append(test_function(sample[i]))

    return Y