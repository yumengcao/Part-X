
import numpy as np


from Functional import __tools__
from partitioning_algorithm.partitioning_algorithm import partitioning
from Sampling_Method import Uniform_random
from Model_constuction.GP_Model import GP_model
from Classify_Method.classification import region_classify
from Sampling_Method.Bayesian_optimization.Bayesian_optimizer import Bayesian_Optimizer
from Grouping_Method import grouping

class Part_X:

    def __init__(self, region: list, method: str, function: str, 
                 budget: int, grouping: str) -> dict:
        '''
        region: region need to be classified
        method: 'uniform_sampling', 'BO', 'grouping'
        test_function: callable function
        budget: total budget for sampling points
        '''
        self.region = region
        self.method = method
        self.function = function
        self.budget = budget
        self.grouping = grouping

    def uni_sample_num(self, group_sampe_num: dict, iter:int, subregion_index: int):
        
        if self.grouping == None or iter == 0:
            if self.method == 'uniform_sampling':
                uni_number = 20
            elif self.method == 'BO':
                uni_number = 10
        else:
           uni_number = group_sampe_num[subregion_index]
        return uni_number
    
    def test_function(self, X):
        return eval(self.function)

    def __exe__(self):
        
        dim = len(self.region)
        budget_cum = 0
        dim_index = list(range(1, dim + 1))*(round(100/dim) +1) ##iteration = 100
        Tree = {}
        uni_sample_iter = {}
        uni_rob_iter = {}

        region_vol = __tools__.vol(self.region)
        for iter in range(100):
            
            score_iter = {}
        
            if iter == 0: 
                theta_undefined = self.region
            und_v = __tools__.undefined_vol(theta_undefined)
            
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
                    uni_number = self.uni_sample_num()
                    sample_uni = Uniform_random.uniform_sampling(subregion, dim, uni_number)
                    robustness_uni = Uniform_random.robustness_values(sample_uni, self.test_function)
                    

                    if self.method == 'BO':
                        __exe_BO_ = Bayesian_Optimizer(sample_uni, robustness_uni, self.function, subregion)
                        subr_bo = __exe_BO_.Bayesian_optimization()
                        subr_sample = np.array(subr_bo.X.append(uni_select_X))
                        subr_robust = np.array(subr_bo.Y.append(uni_select_Y))
                    else:
                        subr_sample = np.array(sample_uni.append(uni_select_X))
                        subr_robust = np.array(robustness_uni.append(uni_select_Y)) 
                    
                    uni_rob_iter[key]  = robustness_uni.append(uni_select_Y)
                    uni_sample_iter[key] = np.array(sample_uni.append(uni_select_X))
                    
                    exe_gp = GP_model(subr_sample, subr_robust, dim, subregion, und_v)
                    score, CI_lower, CI_upper = exe_gp.confidence_interval()
                    score_iter[key] = score
                    theta_plus, theta_undefined, theta_minus = region_classify(subregion, CI_lower, CI_upper,)

               
                if self.grouping != None:
                    group_crit = grouping.criteria(sum(list(uni_rob_iter()), []))
                    group_sample_num, group_result = grouping.grouping(score_iter, group_crit, 
                                                                part_subregions, part_number=2)
                
                        


                    

