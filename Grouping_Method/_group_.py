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
    group_result['group1'] = {}
    group_result['group2'] = {}
    group_result['group3'] = {}
    group_result['group4'] = {}
    group_result['group5'] = {}
    group_result['group6'] = {}

    for key in list(scores):

        for j in range(part_number):

            sub_series = str(eval(key)*part_number + j - (part_number - 1))

            if scores[key] < group_crit[0]:
                group_sample_num[sub_series] = 5
                group_result['group1'][key] = subregions[key]
                continue
            if group_crit[0]< scores[key] <group_crit[1]:
                group_sample_num[sub_series] = 7
                group_result['group2'][key] = subregions[key]
                continue
            if group_crit[1]< scores[key] <group_crit[2]:
                group_sample_num[sub_series] = 10
                group_result['group3'][key] = subregions[key]
                continue
            if group_crit[2]< scores[key] <group_crit[3]:
                group_sample_num[sub_series] = 10
                group_result['group4'][key] = subregions[key]
                continue
            if group_crit[3]< scores[key] <group_crit[4]:
                group_sample_num[sub_series] = 7
                group_result['group5'][key] = subregions[key]
                continue
            if scores[key] > group_crit[4]:
                group_sample_num[sub_series] = 5
                group_result['group6'][key] = subregions[key]
 
    return group_sample_num, group_result