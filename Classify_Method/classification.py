def region_classify(subregion: list, CI_lower: int, 
                    CI_upper: int,  
                    score: int, grouping: int, 
                    theta_plus_iter: dict, theta_minus_iter: dict,
                    theta_undefined_iter: dict, index: str, level_quantile: list):
    '''
    subregion CIassification
    input: subregion (list): target subregion
    CI_lower (int) : lower bound of the subregion
    CI_upper (int): upper bound of the CI of the subregion
    level_quantile (list): level quantile of this iteration 
    score (int): score of the subregion

    '''
    if CI_lower > 0:
        if grouping != None:
            if score > level_quantile[4]:
                theta_plus_iter[index] = subregion
        else:
            theta_plus_iter[index] = subregion

    if CI_upper < 0:
        if grouping != None:
            if score < level_quantile[0]:
                theta_minus_iter[index] = subregion
        else:
            theta_minus_iter[index] = subregion
    
    else:
        theta_undefined_iter[index] = subregion
    
    return theta_minus_iter, theta_plus_iter, theta_undefined_iter
