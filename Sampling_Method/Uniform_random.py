import numpy as np

def uniform_sampling(subregion, number, dim ):
    samples = [[]*number]*dim
    for i in range(dim):
        samples[i] = np.random.uniform(subregion[i][0],subregion[i][1],number)
    sample_arr = np.array(samples)
    sample = sample_arr.T
    return sample



def roubdtness_values(sample, test_function):
    Y = []
    for i in range(len(sample)):
        Y.append(test_function(sample[i]))

    return Y