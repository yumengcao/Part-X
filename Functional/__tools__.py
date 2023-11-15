import numpy as np
def vol(sub_u:list,i_dim: int) -> int:    #calculate the volume of undefined area
    '''
    calculate defined regions‘ volume
    Parameters:
        sub_u(list) :defined regions
        i_dim(int) : dimension of these regions
    

    Returns:
        int: the volume of that regions

    '''
    v = 0
    for i in range(len(sub_u)):
        a = []
        for j in range(i_dim):
            a.append ((sub_u[i][j][1] - sub_u[i][j][0]))
        #print(a)
        v += np.prod(a)# * (sub_u[i][2][1] - sub_u[i][2][0])* (sub_u[i][3][1] - sub_u[i][3][0])* (sub_u[i][4][1] - sub_u[i][4][0])
    
    #print(v)
    return v

def undefined_vol(undefined_region: dict) -> int:
    undefined_volumn = 0
    for i in undefined_region.keys():
        a = []
        for j in range(len(undefined_region[str(i)])):
            a.append ((undefined_region[str(i)][j][1] - \
                       undefined_region[str(i)][j][0]))
        undefined_volumn += np.prod(a) 
    return undefined_volumn