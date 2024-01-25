import numpy as np
from random import sample
import copy

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


def _uni_number_(subregions: dict, 
               uni_rob_iter: dict, dim: int) -> dict:
    '''
    uniform in the whole region
     
    subregions(dict): dict of subregions
    uni_rob_iter(dict): cumm roboustness values of each iterations
    dim(int)

    return:
    ni_rob_select(dict): selected robs
    (need to *vol(subregion))

    '''
    
    uni_density = min(list(len(uni_rob_iter[key])/vol(subregions[key], dim) for \
                           key in uni_rob_iter.keys()))
    uni_rob_select = {}
    for key in subregions.keys():
        sub_num = int(uni_density * vol(subregions[key], dim))
        uni_rob_select[key] = sample(uni_rob_iter[key], sub_num)
    
    return uni_rob_select

# def del_grouping(theta_plus_iter: dict, theta_minus_iter: dict, grouping: dict) -> dict:
    
#     for key in grouping['group1'].copy().keys():
#         if key in theta_minus_iter.keys():
#             del grouping['group1'][key]
    
#     for key in grouping['group6'].copy().keys():
#         if key in theta_plus_iter.keys():
#             del grouping['group6'][key]
    
#     return grouping