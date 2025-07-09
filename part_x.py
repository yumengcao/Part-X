
import sys
import numpy as np
import argparse
import logging
import warnings
import math 
import copy
import time
from treelib import Node, Tree
from Graphing.partition_plot import part_plot
from Functional.__tools__ import vol, undefined_vol , extract_regions_and_parents_iter
from partitioning_algorithm.partitioning_algorithm import Partitioning
from Sampling_Method.Uniform_random import uniform_sampling, robustness_values
from Model_construction.GP_Model import GP_model
from Classify_Method.classification import region_classify
from Sampling_Method.Bayesian_optimization.Bayesian_optimizer import Bayesian_Optimizer
from Sampling_Method.allocate_bo_uni import __allo__b_uni__
from Sampling_Method.Merge_Filter_sample import merge_parent_samples_to_child
from Graphing.partition_plot import part_plot
from Graphing.grouping_plot import group_plot
from Graphing.sampling_plot import sample_plot
from score_construction.sample_budget_allocation import sample_allo_mother
from score_construction.score_scaling import score_scale
from score_construction.dynamic_partition import score_based_partition

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

class Part_X:

    def __init__(self, region, method, function, budget):
        self.region = region
        self.method = method
        self.function = function
        self.budget = budget
        
    def test_function(self, X):
        return eval(self.function)

    def __exe__(self):
        start = time.time()
        region = eval(self.region)
        dim = len(region)
        budget_cum = 20
        total_budget = 0

        tree = {
            'iter_0': {
                'parent_NULL': {
                    'r1_L0': region
                }
            }
        }
        mother_allocation = {'r1_L0': 20}
        dim_index = {'r1_L0': 0}
        part_number = {'r1_L0': 2}
        region_counter = 0
        region_vol = vol(region, dim)

        theta_plus = {}
        theta_minus = {}

        sample_all = {'iter_1' : {}}
        rob_all = {'iter_1' : {}}
        score_scaled = {}
        score_unscaled = {}

        for iteration in range(15):
            budget_cum += total_budget

            score_iter = {}
            avg_mu_iter = {}
            avg_sigma_iter = {}
            theta_minus_iter = {}
            theta_plus_iter = {}

            if iteration == 0:
                theta_undefined = tree['iter_0']['parent_NULL']
            und_v = undefined_vol(theta_undefined, dim)

            if budget_cum < self.budget and \
                und_v > 0.01 * region_vol:
                    
                logging.info(f"Starting iteration {iteration+1}, cumulative budget: {budget_cum}")
                iter_key = 'iter_' + str(iteration + 1)
                partitioner = Partitioning(tree, mother_allocation, dim_index, part_number, dim, region_counter)
                tree, child_allocation, dim_index, region_counter = partitioner.partition()
                
                part_subregions, parent_iter = extract_regions_and_parents_iter(tree, iter_key)
                
                
                theta_minus_iter = {}
                theta_plus_iter = {}
                theta_undefined = {}
                sample_all[iter_key] = {}
                rob_all[iter_key] = {}
                
                for key in part_subregions.keys():
                    subregion = part_subregions[key]

                    if self.method == 'BO':
                        n_bo, n_unif = __allo__b_uni__(child_allocation[key])
                    else:
                        n_unif = child_allocation[key]

                    sample_uni = uniform_sampling(subregion, dim, n_unif)
                    robustness_uni = robustness_values(sample_uni, lambda x: self.test_function(x))

                    if self.method == 'BO':
                        __exe_BO_ = Bayesian_Optimizer(sample_uni, robustness_uni, self.function, subregion, n_bo)
                        subr_sample, subr_robust = __exe_BO_.Bayesian_optimization()
                    else:
                        subr_sample = sample_uni.copy()
                        subr_robust = robustness_uni.copy()
                    
                    
                    
                    sample_all, rob_all = merge_parent_samples_to_child(sample_all, rob_all, iteration, key,
                        parent_iter[key], subregion, subr_sample, subr_robust)
                    #print('sample_all:', sample_all)
                    #print('rob_all:', rob_all)
                    
                    exe_gp = GP_model(sample_all['iter_' + str(iteration+1)][key],
                                      rob_all['iter_' + str(iteration+1)][key],
                                      dim, subregion, 16)
                    avg_mu, avg_sigma, score, CI_lower, CI_upper = exe_gp.confidence_interval()
                    avg_mu_iter[key] = avg_mu
                    avg_sigma_iter[key] = avg_sigma
                    score_iter[key] = score

                    theta_minus_iter, theta_plus_iter, theta_undefined = region_classify(
                        subregion, CI_lower, CI_upper, key, theta_undefined,
                        theta_minus_iter, theta_plus_iter)

                
                score_unscaled[iter_key] = score_iter
                score_scaled[iter_key] = score_scale(avg_mu_iter, score_iter)
                part_number, total_subregions = score_based_partition(score_scaled[iter_key], iteration)
                total_budget = total_subregions * 15 
                print(f"Total budget for iteration {iteration+1}: {total_budget}")
                mother_allocation = sample_allo_mother(score_scaled[iter_key], total_budget) 
                print('score',score_iter)
                print('score_scaled',score_scaled[iter_key])
                print('mother_allocation:', mother_allocation)
                print(part_number)
                theta_plus[iter_key] = theta_plus_iter
                theta_minus[iter_key] = theta_minus_iter
            else:
                logging.info("Stopping criteria met: budget limit or low undefined volume.")
                break
            
        logging.info(f"Final unscaled scores: {score_unscaled}")
        logging.info(f"Final scaled scores: {score_scaled}")
        logging.info(f"Total cumulative budget used: {budget_cum}")
        logging.info(f"Remaining undefined volume proportion: {und_v/region_vol:.4f}")
        end = time.time()
        logging.info(f"Total run time: {end - start:.2f} seconds")

        return theta_minus, theta_plus, theta_undefined, budget_cum, sample_all, rob_all, tree

if __name__ == "__main__":
    arguments_parser = argparse.ArgumentParser(description="level-set classification")
    arguments_parser.add_argument("-r", "--region", type=str, help="region needed to be classified, as [[,], [,], [,], ...]")
    arguments_parser.add_argument("-m", "--method", type=str, help="sampling method, 'BO' or 'uniform_sampling' ")
    arguments_parser.add_argument("-f", "--function", type=str, help=" target black-box function as 'X[1]+X[0]...' ")

    args = arguments_parser.parse_args()
    bart = Part_X(args.region, args.method, args.function, budget = 3000)
    logging.info("Input region: {}".format(args.region))
    region = eval(args.region)
    #test_function = eval(args.function) 
    theta_minus, theta_plus, theta_undefined, budget_cum, sample_all, rob_all, tree = bart.__exe__()

    part_plot(theta_minus, theta_plus, theta_undefined, region, args.function, args.method,
            sample_all, rob_all)
    logging.info("---- Process end ----")
                    

