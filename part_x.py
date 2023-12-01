
# print(sys.path)
# sys.path.append('/Users/yumengcao/opt/anaconda3/bin/python')
# sys.path.append('/usr/bin/python3')
# sys.path.append('/Users/yumengcao/part_x_python')
import numpy as np
import argparse
import logging


from Functional.__tools__ import vol, undefined_vol, select_regions, _uni_number_
from partitioning_algorithm.partitioning_algorithm import partitioning
from Sampling_Method.Uniform_random import uniform_sampling, robustness_values
from Model_constuction.GP_Model import GP_model
from Classify_Method.classification import region_classify, group_classify
from Sampling_Method.Bayesian_optimization.Bayesian_optimizer import Bayesian_Optimizer
from Grouping_Method._group_ import criteria, _grouping_

class Part_X:

    def __init__(self, region, method, function, 
                 budget, grouping):
        '''
        region: region need to be classified
        method: 'uniform_sampling', 'BO'
        test_function: callable function
        budget: total budget for sampling points
        '''
        self.region = region
        self.method = method
        self.function = function
        self.budget = budget
        self.grouping = grouping

    def uni_sample_num(self, iter, 
                       subregion_index, group_sample_num = None):
        
        if self.grouping == None or iter == 0:
            if self.method == 'uniform_sampling':
                uni_number = 20
            elif self.method == 'BO':
                uni_number = 10
        else:
           uni_number = group_sample_num[subregion_index]
        return uni_number
    
    def test_function(self, X):
        return eval(self.function)

    def __exe__(self):
        
        region = eval(self.region)
        dim = len(region)
        budget_cum = 0
        dim_index = list(range(1, dim + 1))*(round(100/dim) +1) ##iteration = 100
        Tree = {}
        uni_sample_iter = {}
        uni_rob_iter = {}
        grouping = {}
        region_vol = vol(region)
        theta_plus = {}
        theta_minus = {}
        
        for iter in range(100):
            
            score_iter = {}
        
            if iter == 0: 
                theta_undefined = region
            und_v = undefined_vol(theta_undefined)
            
            if budget_cum < self.budget and \
                und_v > 0.01 * region_vol:
                branching= partitioning(theta_undefined, dim_index[iter], dim, 
                                        uni_sample_iter, uni_rob_iter, iter,
                                        part_number = 2)
                part_subregions, uni_select_X, uni_select_Y = branching.partitioning_algorithm()
                Tree['level'+ str(iter + 1)] = part_subregions
                
                uni_sample_iter = {}
                uni_rob_iter = {}

                for key in part_subregions.keys():
                    subregion = part_subregions[key]
                    uni_number = self.uni_sample_num(iter, key, group_sample_num)
                    sample_uni = uniform_sampling(subregion, dim, uni_number)
                    robustness_uni = robustness_values(sample_uni, self.test_function)
                    budget_cum += uni_number

                    if self.method == 'BO':
                        __exe_BO_ = Bayesian_Optimizer(sample_uni, robustness_uni, self.function, subregion, n_bo = 10)
                        subr_bo = __exe_BO_.Bayesian_optimization()
                        subr_sample = np.array(subr_bo.X.append(uni_select_X))
                        subr_robust = np.array(subr_bo.Y.append(uni_select_Y))
                        budget_cum += 10
                    else:
                        subr_sample = np.array(sample_uni.append(uni_select_X))
                        subr_robust = np.array(robustness_uni.append(uni_select_Y)) 
                    
                    uni_rob_iter[key]  = robustness_uni.append(uni_select_Y)
                    #uni_sample_iter[key] = np.array(sample_uni.append(uni_select_X))
                    
                    exe_gp = GP_model(subr_sample, subr_robust, dim, subregion, und_v)
                    score, CI_lower, CI_upper = exe_gp.confidence_interval()
                    score_iter[key] = score
                    theta_minus_iter, theta_plus_iter, theta_undefined  = region_classify(subregion, 
                                                                                          CI_lower, CI_upper, key )

               
                if self.grouping != None:
                    uni_rob_select = _uni_number_(part_subregions, uni_rob_iter, dim)
                    group_crit = criteria(sum(list(uni_rob_select()), []))
                    group_sample_num, group_result = _grouping_(score_iter, group_crit, 
                                                                part_subregions, part_number=2)
                    grouping['level'+ str(iter + 1)] = group_result
                    theta_minus_iter, theta_plus_iter, theta_undefined = group_classify(group_crit, theta_plus_iter, 
                                                                                        theta_minus_iter, theta_undefined, 
                                                                                        part_subregions)
                theta_plus['level'+ str(iter + 1)] = theta_plus_iter
                theta_minus['level'+ str(iter + 1)] = theta_minus_iter
        
        return theta_minus, theta_plus, theta_undefined, budget_cum, grouping, Tree         

if __name__ == "__main__":

    arguments_parser = argparse.ArgumentParser(
        description="level-set classification")
    arguments_parser.add_argument(
        "-r",
        "--region",
        type = str,
        help = "region needed to be classified, as [[,], [,], [,]...]"
    )
    arguments_parser.add_argument(
        "-m",
        "--method",
        type = str,
        help = "sampling method, 'BO' or 'uniform_sampling' "
    )
    arguments_parser.add_argument(
        "-f",
        "--function",
        type = str,
        help = " target black-box function as 'X[1]+X[0]...' "
    )
    arguments_parser.add_argument(
        "-b",
        "--budget",
        type = int,
        help = "total budget (sampling points)"
    )

    arguments_parser.add_argument(
        "-g",
        "--grouping",
        type = str,
        help = "use grouping method?"
    )
    args = arguments_parser.parse_args()
    # Convert data
    bart = Part_X(args.region, args.method, args.function, args.budget, args.grouping)
    logging.info("Input region: {}".format(args.input))
    logging.info("Outputs: {}".format(args.output))
    theta_minus, theta_plus, theta_undefined, budget_cum, grouping, Tree  = bart.__exe__()
    logging.info("---- Process end ----")
                    

