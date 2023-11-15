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
        for j in range(len(undefined_region[str(i)])):
            a.append ((undefined_region[str(i)][j][1] - \
                       undefined_region[str(i)][j][0]))
        undefined_volumn += np.prod(a) 
    return undefined_volumn