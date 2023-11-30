import numpy as np

def vol(sub_u:list,i_dim: int) -> int:    #calculate the volume of undefined area
    '''
    calculate defined regions‘ volume
    Parameters:
        sub_u(list) :defined regions [[1,2], [3,4]]
        i_dim(int) : dimension of these regions
    

    Returns:
        int: the volume of that regions

    '''
    v = 0
    a = []
    for j in range(i_dim):
        a.append ((sub_u[j][1] - sub_u[j][0]))
    v += np.prod(a)
    return v

def undefined_vol(undefined_region: dict) -> int:
    '''
    calculate undefined regions‘ total volume in eacg iter
    Parameters:
        ndefined_region (dict) : undefined regions dict['level_2'] = {'1': [], '2': []}
     
    

    Returns:
        int: the volume of undefined subregions

    '''
    undefined_volumn = 0
    for i in undefined_region.keys():
        a = []
        for j in range(len(undefined_region[i])):
            a.append ((undefined_region[i][j][1] - \
                       undefined_region[i][j][0]))
        undefined_volumn += np.prod(a) 
    return undefined_volumn

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


def uni_number(subregions: dict, 
               uni_rob_iter: dict, dim: int) -> float:
    '''
    uniform in the whole region
     
    subregions(dict): dict of subregions
    uni_rob_iter(dict): cumm roboustness values of each iterations
    dim(int)

    return:
    uni_number(float): density of uniform sampling points 
    (need to *vol(subregion))

    '''
    
    index = list(uni_rob_iter)
    k = list(len(uni_rob_iter[key]) for key in uni_rob_iter.keys())
    minLen = min(k)
    rob = list(uni_rob_iter.values())
    keys = [rob.index(x) for x in rob if len(x) == minLen]
    if len(keys) != 1:
        target_vol = max(list(vol(subregions[index[key]], dim) for \
                           key in keys))

    else:
        target_vol = vol(subregions[index[keys[0]]], dim)
    
    uni_number = minLen / target_vol
    return uni_number
    
