import numpy as np


def criteria( robust_unif: list):
    '''
    calculate the grouping criteria
    
    Input：
    robust_unif (list): summary of robustness in subregions

    return:
    group_crit (list): grouping criteria for this iteration
    '''
    group_crit = []
    for i in [5, 25, 50, 75, 95]:
        group_crit.append(np.percentile(robust_unif, i))
    return group_crit

def _grouping_ (scores: dict, group_crit: list, subregions: dict, part_number: int) -> dict:
    '''
    Grouping subregions with criterias

    Input: 
    scores (dict): list if subregions' scores in this iteration
    subregions (dict): dictionary of subregions

    Return:
    group_sample_num (dict): grouping result as sample numbers for the subregions
    for next iteration
    '''
    group_sample_num = {}
    group_result = {}
    for key in scores.keys():
        for j in range(part_number):
            sub_series = str(eval(key)*part_number + j - (part_number - 1))
            if scores[key] < group_crit[0]:
                group_sample_num[sub_series] = 10
                group_result['0- 5%'][key] = subregions[key]
                continue
            if scores[key] in range(group_crit[0], group_crit[1]):
                group_sample_num[sub_series] = 15
                group_result['5- 25%'][key] = subregions[key]
                continue
            if scores[key] in range(group_crit[1], group_crit[2]):
                group_sample_num[sub_series] = 20
                group_result['25 -50%'][key] = subregions[key]
                continue
            if scores[key] in range(group_crit[2], group_crit[3]):
                group_sample_num[sub_series] = 20
                group_result['50 -75%'][key] = subregions[key]
                continue
            if scores[key] in range(group_crit[3], group_crit[4]):
                group_sample_num[sub_series] = 15
                group_result['75 -95%'][key] = subregions[key]

                continue
            if scores[key] > group_crit[4]:
                group_sample_num[sub_series] = 10
                group_result['95-100%'][key] = subregions[key]

        return group_sample_num, group_result