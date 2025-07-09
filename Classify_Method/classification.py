def region_classify(subregion: list, CI_lower: float, 
                    CI_upper: float,  
                    index: str, theta_undefined: dict, 
                    theta_minus_iter: dict, theta_plus_iter: dict):
    '''
    subregion CIassification
    
    input: 
    subregion (list): target subregion
    CI_lower (int) : lower bound of the subregion
    CI_upper (int): upper bound of the CI of the subregion
    index (str): serial number of subregion
    iteration (int): iteration of the algorithm

    '''

    if CI_lower > 0:
        theta_plus_iter[index] = subregion

    elif CI_upper < 0:
        theta_minus_iter[index] = subregion
    
    else:
        theta_undefined[index] = subregion
    
    return theta_minus_iter, theta_plus_iter, theta_undefined


