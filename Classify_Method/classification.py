def region_classify(subregion: list, CI_lower: float, 
                    CI_upper: float,  
                    index: str, theta_undefined: dict, 
                    theta_minus_iter: dict, theta_plus_iter: dict):
    '''
    subregion CIassification
    input: subregion (list): target subregion
    CI_lower (int) : lower bound of the subregion
    CI_upper (int): upper bound of the CI of the subregion
    level_quantile (list): level quantile of this iteration 
    score (int): score of the subregion

    '''

    if CI_lower > 0:
        theta_plus_iter[index] = subregion

    elif CI_upper < 0:
        theta_minus_iter[index] = subregion
    
    else:
        theta_undefined[index] = subregion
    
    return theta_minus_iter, theta_plus_iter, theta_undefined


def group_classify(level_quantile: list, theta_plus_iter: dict,
                   theta_minus_iter: dict, theta_undefined: dict, score_iter: dict, 
                   subregions: dict) -> dict:
   
    if theta_plus_iter != {}:
        for key in theta_plus_iter.copy().keys():      
            if score_iter[key] < level_quantile[4]:
                del theta_plus_iter[key] 
                theta_undefined[key] = subregions[key]
    
    if theta_minus_iter != {}:
        print(theta_minus_iter.keys())
        for key in theta_minus_iter.copy().keys():      
            if score_iter[key] > level_quantile[0]:
                del theta_minus_iter[key] 
                theta_undefined[key] = subregions[key]
    return theta_minus_iter, theta_plus_iter, theta_undefined