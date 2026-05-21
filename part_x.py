
# -*- coding: utf-8 -*-
"""
This is a modified version of your Part_X class and main execution script, designed to be more modular and user-friendly for running experiments with different test functions and parameters.

input example:
python3 part_x.py -pm partx_params -fm testfunction_himm --out_csv partx_iter_stats.csv

"""
import sys
import numpy as np
import argparse
import logging
import warnings
import math
import copy
import time
import importlib
import csv
import os

# 你的原始 imports（保持不变）
from treelib import Node, Tree
from Graphing.partition_plot import part_plot
from Graphing.score_boxplot import plot_one_iteration_boxplot
from Graphing.allocation_comparison import plot_allocation_bar
from Graphing.score_comparison import plot_score_comparison_bar
from Functional.__tools__ import vol, undefined_vol, extract_regions_and_parents_iter, check_all_same_sign
from partitioning_algorithm.partitioning_algorithm import Partitioning
from Sampling_Method.Uniform_random import uniform_sampling, robustness_values, sobol_sampling
from Model_construction.GP_Model import GP_model
from Classify_Method.classification import region_classify
from Sampling_Method.Bayesian_optimization.Bayesian_optimizer import Bayesian_Optimizer
from Sampling_Method.allocate_bo_uni import allo_new, allo__b_uni__
from Sampling_Method.Merge_Filter_sample import merge_parent_samples_to_child
from score_construction.sample_budget_allocation import sample_allo_mother
from score_construction.score_scaling import score_scale
from score_construction.dynamic_partition import score_based_partition
from Functional.miscfy_cal import compute_misclassification_mc_iter, _flatten_theta_regions 
# logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# ---------- helpers: load modules and function ----------
def load_params_module(name):
    if name is None:
        return None
    try:
        return importlib.import_module(name)
    except Exception as e:
        logging.error(f"Failed to import params module {name}: {e}")
        raise

def load_function_callable(name):
    try:
        m = importlib.import_module(name)
    except Exception as e:
        logging.error(f"Failed to import function module {name}: {e}")
        raise
    # try common names
    for n in ('fun', 'test_function', 'testfunction', 'function', 'fun_blackbox'):
        if hasattr(m, n):
            return getattr(m, n)
    # fallback: first callable in module
    for k, v in vars(m).items():
        if callable(v) and not k.startswith("_"):
            return v
    raise AttributeError(f"function module {name} has no callable function named 'fun' or 'test_function'")

def make_vectorized_wrapper(fun_callable):
    # returns f_true(pts) accepting (N,d) or (d,) point
    def f_true(pts):
        arr = np.asarray(pts, dtype=float)
        if arr.ndim == 1:
            # single point
            v = fun_callable(list(arr), 1)
            if hasattr(v, "__iter__"):
                return float(np.mean(v))
            else:
                return float(v)
        else:
            out = np.empty(arr.shape[0], dtype=float)
            for i in range(arr.shape[0]):
                v = fun_callable(list(arr[i, :]), 1)
                if hasattr(v, "__iter__"):
                    out[i] = float(np.mean(v))
                else:
                    out[i] = float(v)
            return out
    return f_true

def estimate_y_delta(f_true, region, delta=0.1, grid_per_dim=200, mc_samples=200000, dim_threshold=3, seed=123):
    lo = np.array([r[0] for r in region], dtype=float)
    hi = np.array([r[1] for r in region], dtype=float)
    d = len(region)
    if d <= dim_threshold:
        # grid
        counts = [grid_per_dim] * d
        axes = [np.linspace(lo[i], hi[i], num=counts[i]) for i in range(d)]
        mesh = np.meshgrid(*axes, indexing='xy')
        pts = np.stack([m.ravel() for m in mesh], axis=1)
        vals = f_true(pts)
        vals = np.asarray(vals).ravel()
        return float(np.quantile(vals, delta))
    else:
        rng = np.random.RandomState(seed)
        pts = rng.uniform(low=lo, high=hi, size=(int(mc_samples), d))
        vals = f_true(pts)
        vals = np.asarray(vals).ravel()
        return float(np.quantile(vals, delta))

# ---------- utility: distinct sample counting ----------
def _concat_all_samples(l_subr, coord_cols=None):
    import pandas as pd
    dfs = []
    for s in l_subr:
        if hasattr(s, 'pd_sample_record') and not s.pd_sample_record.empty:
            dfs.append(s.pd_sample_record.copy())
    if len(dfs) == 0:
        return pd.DataFrame(), []
    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    if coord_cols is None:
        coord_cols = [c for c in df_all.columns if str(c).lower().startswith('x')]
    return df_all, coord_cols



def count_distinct_samples_global(l_subr, coord_cols=None):
    import pandas as pd
    df_all, coord_cols = _concat_all_samples(l_subr, coord_cols)
    if df_all.empty:
        return 0, df_all
    if len(coord_cols) == 0:
        coord_cols = list(df_all.columns)
    df_unique = df_all.drop_duplicates(subset=coord_cols)
    N_total = len(df_unique)
    return int(N_total), df_unique

# ---------- main class ----------
class Part_X:

    def __init__(self, region, method, function_callable, budget, params=None, y_delta=None, args=None):
        # region: list of (low,high) tuples OR string to eval (we will handle)
        # method: 'BO' or 'uniform'
        # function_callable: callable fun(X, r=1)
        self.region = region
        self.method = method
        self.function_callable = function_callable
        self.budget = budget
        self.params = params
        self.y_delta = y_delta
        self.args = args  # keep CLI args if needed

        # convenient wrapper returning scalar mean for a point X (list/array)
        import numpy as _np
        self.function = lambda X, r=1: float(_np.mean(self.function_callable(list(X), r)))

        # placeholder for subregions if you want to reference externally after run
        self.l_subr = None

    def test_function(self, X, r=1):
        # kept for compatibility: returns scalar mean
        return self.function(X, r)

    def __exe__(self):
        start = time.time()

        # region may be string from cli; handle both
        if isinstance(self.region, str):
            region = eval(self.region)
        else:
            region = self.region
        dim = len(region)
        region_vol = vol(region, dim)

        # If y_delta not provided, try to read from params module if present
        if self.y_delta is None and self.params is not None and hasattr(self.params, 'y_delta'):
            self.y_delta = getattr(self.params, 'y_delta')

        # prepare loop vars as your original code
        budget_cum = getattr(self.params, 'initial_budget_per_region', 20) if self.params is not None else 20
        total_budget = 0

        tree = {
            'iter_0': {
                'parent_NULL': {
                    'r1_L0': region
                }
            }
        }
        mother_allocation = {'r1_L0': budget_cum}
        dim_index = {'r1_L0': 0}
        part_number = {'r1_L0': 2}
        region_counter = 0

        theta_plus = {}
        theta_minus = {}
        vMinus_cum = 0
        vPlus_cum = 0
        sample_all = {'iter_1': {}}
        rob_all = {'iter_1': {}}
        score_scaled = {}
        score_unscaled = {}

        remain_volume = {}
        minus_volume = {}
        plus_volume = {}
        budget_cum_dict = {}
        last_iteration = 0

        # stats collection for CSV
        iter_stats = []
        # header keys: cumulative_budget, ratio_M, ratio_P, ratio_C, ratio_misclassified
        csv_outfile = getattr(self.args, 'out_csv', 'partx_iter_stats.csv') if self.args is not None else 'partx_iter_stats.csv'
        # backup existing file
        if os.path.exists(csv_outfile):
            os.rename(csv_outfile, csv_outfile + ".bak")

        # initial undefined
        theta_undefined = tree['iter_0']['parent_NULL']
        und_v = region_vol

        # main loop (kept largely your original)
        max_iter = getattr(self.params, 'max_iterations', 10) if self.params is not None else 10
        for iteration in range(max_iter):
            score_iter = {}
            avg_mu_iter = {}
            avg_sigma_iter = {}
            theta_minus_iter = {}
            theta_plus_iter = {}

            if not (budget_cum < self.budget and und_v > 0.005 * region_vol):
                logging.info("Stopping criteria met: budget limit or low undefined volume.")
                break

            logging.info(f"Starting iteration {iteration+1}, cumulative budget: {budget_cum}")
            iter_key = 'iter_' + str(iteration + 1)

            partitioner = Partitioning(tree, mother_allocation, dim_index, part_number,
                                        dim, region_counter, list(theta_undefined.keys()), min_samples_per_child=1)
            tree, child_allocation, dim_index, region_counter = partitioner.partition()

            part_subregions, parent_iter = extract_regions_and_parents_iter(tree, iter_key)

            theta_minus_iter = {}
            theta_plus_iter = {}
            theta_undefined = {}
            sample_all[iter_key] = {}
            rob_all[iter_key] = {}
            sample_sign = {}

            # loop each subregion
            for key in part_subregions.keys():
                subregion = part_subregions[key]

                if self.method == 'BO' and child_allocation[key] >1:
                    n_bo, n_unif = allo__b_uni__(child_allocation[key])
                else:
                    n_unif = child_allocation[key]

                sample_uni = uniform_sampling(subregion, dim, n_unif)
                robustness_uni = robustness_values(sample_uni, lambda x: self.test_function(x))
                sample_all, rob_all = merge_parent_samples_to_child(sample_all, rob_all, iteration, key,
                                                                    parent_iter[key], subregion, sample_uni,
                                                                    robustness_uni)

                if self.method == 'BO' and child_allocation[key] >1:
                    __exe_BO_ = Bayesian_Optimizer(sample_all[iter_key][key], rob_all[iter_key][key],
                                                    self.test_function, subregion, n_bo)
                    subr_sample, subr_robust = __exe_BO_.Bayesian_optimization()
                    sample_all[iter_key][key] = subr_sample
                    rob_all[iter_key][key] = subr_robust
                else:
                    subr_sample = sample_all[iter_key][key]
                    subr_robust = rob_all[iter_key][key]

                sample_sign[key] = check_all_same_sign(subr_robust)
                
                exe_gp = GP_model(subr_sample, subr_robust,
                                dim, subregion, 64)
                avg_mu, avg_sigma, score, CI_lower, CI_upper = exe_gp.confidence_interval()
                avg_mu_iter[key] = avg_mu
                avg_sigma_iter[key] = avg_sigma
                score_iter[key] = score

                theta_minus_iter, theta_plus_iter, theta_undefined = region_classify(
                    subregion, CI_lower, CI_upper, key, theta_undefined,
                    theta_minus_iter, theta_plus_iter)

            # update undefined volume
            und_v = undefined_vol(theta_undefined, dim)
            remain_und_v = und_v / region_vol
            logging.info(f"Iteration {iteration+1} completed. Remaining undefined volume proportion: {remain_und_v:.4f}")
            score_unscaled[iter_key] = score_iter
            score_scaled[iter_key] = score_scale(avg_mu_iter, score_iter)
            part_number, total_subregions = score_based_partition(score_unscaled[iter_key], iteration, sample_sign)
            
            base = getattr(self.params, 'initial_budget_per_region', 20) if self.params is not None else 20

            total_budget = round(
                base
                * (total_subregions)    #**0.7            # 次线性，而不是线性
                #* (dim ** 0.75)                    
                #* (1 / (iteration + 1)) ** 0.25    
                #* ( 0.5+remain_und_v)                  
                )
            # total_budget = round(getattr(self.params, 'initial_budget_per_region', 20) * total_subregions * (1 / (iteration + 1)) ** (1 / 4)) \
            #     if self.params is not None else round(20 * total_subregions *(dim/2)* (1 / (iteration + 1)) ** (1 / 4))
            if budget_cum + total_budget > self.budget:
                logging.info('last iteration exhausted budget, adjusting to remaining budget')
                last_iteration = 1
                total_budget = self.budget - budget_cum

            budget_cum += total_budget
            budget_cum_dict[iter_key] = budget_cum
            logging.info(f"Total subregions: {total_subregions}, allocated budget for this iteration: {total_budget}, cumulative budget: {budget_cum}")

            mother_allocation_unscaled = sample_allo_mother(score_iter, total_budget, last_iteration)
            mother_allocation = sample_allo_mother(score_unscaled[iter_key], total_budget, last_iteration)

            vMinus_cum += undefined_vol(theta_minus_iter, dim)
            vPlus_cum += undefined_vol(theta_plus_iter, dim)
            remain_volume[iter_key] = und_v / region_vol
            minus_volume[iter_key] = vMinus_cum / region_vol
            plus_volume[iter_key] = vPlus_cum / region_vol

            theta_plus[iter_key] = theta_plus_iter
            theta_minus[iter_key] = theta_minus_iter

            # plotting optional (kept as in your original)
            if dim == 2:
                try:
                    part_plot(theta_minus, theta_plus, theta_undefined, region, self.test_function, self.method,
                              sample_all, rob_all, iteration + 1, save_dir="partition_plots")
                except Exception as e:
                    logging.warning("part_plot failed: " + str(e))

            # === statistics to record for this iteration ===
            # distinct cumulative budget
            N_total, df_unique = count_distinct_samples_global([s for s in (getattr(self, 'l_subr', []) + [])] + list(part_subregions.values())) if hasattr(self, 'l_subr') and self.l_subr is not None else count_distinct_samples_global(list(part_subregions.values()))
            # fallback: use budget_cum as cumulative if counting fails
            if N_total == 0:
                N_total = int(budget_cum)

            # compute volume ratios (M,P,C) using f_volumn of active subregions
            #V_total = region_vol
            #V_M = sum(s.f_volumn for s in (list(part_subregions.values()) if part_subregions is not None else []) if getattr(s, 's_label', 'C') == 'M')
            #V_P = sum(s.f_volumn for s in (list(part_subregions.values()) if part_subregions is not None else []) if getattr(s, 's_label', 'C') == 'P')
            #V_C = sum(s.f_volumn for s in (list(part_subregions.values()) if part_subregions is not None else []) if getattr(s, 's_label', 'C') == 'C')
            # normalize
            ratio_M = vMinus_cum / region_vol
            ratio_P = vPlus_cum / region_vol
            ratio_C = und_v / region_vol

            # misclassification estimate: use your compute_misclassification_mc_iter utility if available
            # that function signature in your codebase: compute_misclassification_mc_iter(theta_minus, theta_plus, region, f_str_or_callable, n_samples=50000, seed=42)
            try:
                # if params module provided and user passed function module as string, compute_misclassification accepts function string too;
                # Here we prefer to pass a callable wrapper; compute_misclassification_mc_iter in your repo might accept function string, adjust if needed.
                mis = compute_misclassification_mc_iter(theta_minus, theta_plus, region, self.function_callable, n_samples=getattr(self.args, 'mis_mc_samples', 50000), seed=42)
                ratio_mis = float(mis.get('ratio_misclassified', 0.0))
                ratio_mis_prune = float(mis.get('ratio_incorrect_prune', 0.0))
                ratio_mis_maintain = float(mis.get('ratio_incorrect_maintain', 0.0))
            except Exception as e:
                logging.warning(f"compute_misclassification_mc_iter failed: {e}. Falling back to ratio_mis=0.")
                ratio_mis = 0.0
                ratio_mis_prune = 0.0
                ratio_mis_maintain = 0.0

            iter_stats.append({
                'iter': int(iteration + 1),
                'cumulative_budget': int(N_total),
                'ratio_M': float(ratio_M),
                'ratio_P': float(ratio_P),
                'ratio_C': float(ratio_C),
                'ratio_misclassified': float(ratio_mis),
                'ratio_mis_prune': float(ratio_mis_prune),
                'ratio_mis_maintain': float(ratio_mis_maintain)
            })

            # print concise
            logging.info(f"[iter {iteration+1}] budget={N_total}, M={ratio_M:.4f}, P={ratio_P:.4f}, C={ratio_C:.4f}, mis={ratio_mis:.6g}")

            # update for next iter
            budget_cum_dict[iter_key] = budget_cum
            # update tree/subregions lists
            # append new subregions to self.l_subr for global tracking (optional)
            if self.l_subr is None:
                self.l_subr = list(part_subregions.values())
            else:
                self.l_subr += list(part_subregions.values())

        # end loop

        # write CSV
        keys = ['iter', 'cumulative_budget', 'ratio_M', 'ratio_P', 'ratio_C', 'ratio_misclassified', 'ratio_mis_prune', 'ratio_mis_maintain']
        try:
            with open(csv_outfile, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                for row in iter_stats:
                    writer.writerow(row)
            logging.info(f"Wrote iteration stats to {csv_outfile}")
        except Exception as e:
            logging.error(f"Failed to write CSV {csv_outfile}: {e}")

        end = time.time()
        logging.info(f"Total run time: {end - start:.2f} seconds")
        logging.info(f"iteration cumulative budget used: {budget_cum_dict}")
        logging.info(f"Remaining volume proportions - Undefined: {remain_volume}, Minus: {minus_volume}, Plus: {plus_volume}")
        return theta_minus, theta_plus, theta_undefined, budget_cum, sample_all, rob_all, tree

# ---------- CLI entry ----------
def main():
    parser = argparse.ArgumentParser(description="Part_X: level-set classification (modified)")
    parser.add_argument("-pm", "--params_module", type=str, default=None, help="params module name (python module, e.g. partx_params)")
    parser.add_argument("-fm", "--function_module", type=str, required=True, help="function module name (python module, e.g. testfunction_himm)")
    parser.add_argument("-r", "--region", type=str, default=None, help="region as string to eval, e.g. '[(-2,2),(-2,2)]' (overrides params module region)")
    parser.add_argument("-m", "--method", type=str, default=None, help="sampling method override ('BO' or 'uniform')")
    parser.add_argument("-b", "--budget", type=int, default=None, help="total budget override")
    parser.add_argument("--delta", type=float, default=None, help="target delta (e.g. 0.1). If provided, will estimate y_delta and inject into params")
    parser.add_argument("--grid_res", type=int, default=200, help="grid per-dim for low-d quantile")
    parser.add_argument("--mc_samples", type=int, default=200000, help="MC samples for high-d quantile")
    parser.add_argument("--mis_mc_samples", type=int, default=50000, help="MC samples for misclassification estimate per iter")
    parser.add_argument("--out_csv", type=str, default="partx_iter_stats.csv", help="output CSV file for iteration stats")
    args = parser.parse_args()

    # load params if provided
    params = None
    if args.params_module is not None:
        params = load_params_module(args.params_module)
        logging.info(f"Loaded params module {args.params_module}")
    # load function callable
    fun_callable = load_function_callable(args.function_module)
    logging.info(f"Loaded function from module {args.function_module}")

    # determine region/method/budget: priority CLI -> params module -> defaults in code
    if args.region is not None:
        region = eval(args.region)
    elif params is not None and hasattr(params, 'region'):
        region = getattr(params, 'region')
    else:
        raise ValueError("Region must be provided either via -r or in params module as 'region'")

    method = args.method if args.method is not None else (getattr(params, 'method', 'BO') if params is not None else 'BO')
    budget = args.budget if args.budget is not None else (getattr(params, 'budget', 6000) if params is not None else 6000)

    # if delta provided, estimate y_delta and inject into params (if any)
    y_delta = None
    if args.delta is not None:
        #f_true = make_vectorized_wrapper(fun_callable)
        logging.info(f"Estimating y_delta for delta={args.delta} ...")
        #y_delta = estimate_y_delta(f_true, region, delta=args.delta, grid_per_dim=args.grid_res, mc_samples=args.mc_samples)
        #logging.info(f"Estimated y_delta = {y_delta}")
        if params is not None:
            setattr(params, 'y_delta', y_delta)
            setattr(params, 'f_delta', args.delta)

    # instantiate and run
    px = Part_X(region, method, fun_callable, budget, params=params, y_delta=y_delta, args=args)
    theta_minus, theta_plus, theta_undefined, budget_cum, sample_all, rob_all, tree = px.__exe__()

    # final misclassification summary (optional, reuse compute_misclassification_mc_iter)
    try:
        mis = compute_misclassification_mc_iter(theta_minus, theta_plus, region, fun_callable, n_samples=args.mis_mc_samples, seed=42)
        logging.info(f"Final misclassification estimate: {mis.get('V_misclassified',0.0):.6f} (prop {mis.get('ratio_misclassified',0.0):.6f})")
    except Exception as e:
        logging.warning("Final misclassification estimate failed: " + str(e))

    # final plot (kept for compatibility)
    try:
        if len(region) == 2:
            part_plot(theta_minus, theta_plus, theta_undefined, region, px.test_function, method,
                      sample_all, rob_all, 'final', save_dir="final_partition_plots")
    except Exception as e:
        logging.warning("Final part_plot failed: " + str(e))

if __name__ == "__main__":
    main()
# import sys
# import numpy as np
# import argparse
# import logging
# import warnings
# import math 
# import copy
# import time
# from treelib import Node, Tree
# from Graphing.partition_plot import part_plot
# from Graphing.score_boxplot import plot_one_iteration_boxplot
# from Graphing.allocation_comparison import plot_allocation_bar 
# from Graphing.score_comparison import plot_score_comparison_bar
# from Functional.__tools__ import vol, undefined_vol , extract_regions_and_parents_iter, check_all_same_sign
# from partitioning_algorithm.partitioning_algorithm import Partitioning
# from Sampling_Method.Uniform_random import uniform_sampling, robustness_values, sobol_sampling
# from Model_construction.GP_Model import GP_model
# from Classify_Method.classification import region_classify
# from Sampling_Method.Bayesian_optimization.Bayesian_optimizer import Bayesian_Optimizer
# from Sampling_Method.allocate_bo_uni import allo_new, allo__b_uni__
# from Sampling_Method.Merge_Filter_sample import merge_parent_samples_to_child
# from Graphing.partition_plot import part_plot
# from score_construction.sample_budget_allocation import sample_allo_mother
# from score_construction.score_scaling import score_scale
# from score_construction.dynamic_partition import score_based_partition
# from Functional.miscfy_cal import compute_misclassification_mc_iter, _flatten_theta_regions

# # === Logging Configuration ===
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(levelname)s] %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#     ]
# )

# class Part_X:

#     def __init__(self, region, method, function_callable, budget, params=None, y_delta=None):
#         self.region = region
#         self.method = method
#         self.function_callable = function_callable
#         self.budget = budget
#         self.params = params
#         self.y_delta = y_delta
    
    
#     def test_function(self, X, r=1):
#         """
#         X: list/array-like; r: replication
#         """
#         v = self.function_callable(X, r)
#         # make sure returns scalar or list; we want scalar mean value here
#         if hasattr(v, "__iter__"):
#             import numpy as np
#             return float(np.mean(v))
#         else:
#             return float(v)    
#     #def test_function(self, X):
#         #return eval(self.function)

#     def __exe__(self):
#         start = time.time()
#         region = eval(self.region)
#         dim = len(region)
#         budget_cum = 20
#         total_budget = 0

#         tree = {
#             'iter_0': {
#                 'parent_NULL': {
#                     'r1_L0': region
#                 }
#             }
#         }
#         mother_allocation = {'r1_L0': 20}############
#         dim_index = {'r1_L0': 0}
#         part_number = {'r1_L0': 2}
#         region_counter = 0
#         region_vol = vol(region, dim)

#         theta_plus = {}
#         theta_minus = {}
#         vMinus_cum = 0
#         vPlus_cum = 0
#         sample_all = {'iter_1' : {}}
#         rob_all = {'iter_1' : {}}
#         score_scaled = {}
#         score_unscaled = {}

#         remain_volume = {}
#         minus_volume = {}
#         plus_volume = {}
#         budget_cum_dict = {}
#         last_iteration = 0
#         for iteration in range(10):##########
#             score_iter = {}
#             avg_mu_iter = {}
#             avg_sigma_iter = {}
#             theta_minus_iter = {}
#             theta_plus_iter = {}

#             if iteration == 0:
#                 theta_undefined = tree['iter_0']['parent_NULL']
#                 und_v = region_vol
                
#             if budget_cum < self.budget and \
#                 und_v > 0.01 * region_vol:
                    
#                 logging.info(f"Starting iteration {iteration+1}, cumulative budget: {budget_cum}")
#                 iter_key = 'iter_' + str(iteration + 1)
                
#                 partitioner = Partitioning(tree, mother_allocation, dim_index, part_number, 
#                                            dim, region_counter, list(theta_undefined.keys()))
#                 tree, child_allocation, dim_index, region_counter = partitioner.partition()
                
#                 part_subregions, parent_iter = extract_regions_and_parents_iter(tree, iter_key)
                
                
#                 theta_minus_iter = {}
#                 theta_plus_iter = {}
#                 theta_undefined = {}
#                 sample_all[iter_key] = {}
#                 rob_all[iter_key] = {}
#                 sample_sign = {}
#                 #print('subregions'+ iter_key, part_subregions)
#                 for key in part_subregions.keys():
#                     subregion = part_subregions[key]

#                     if self.method == 'BO':
#                         n_bo, n_unif = allo__b_uni__(child_allocation[key])
#                     else:
#                         n_unif = child_allocation[key]
#                     #print('uniform_sample_length', n_unif)
#                     sample_uni = uniform_sampling(subregion, dim, n_unif)
#                     robustness_uni = robustness_values(sample_uni, lambda x: self.test_function(x))
#                     sample_all, rob_all = merge_parent_samples_to_child(sample_all, rob_all, iteration, key,
#                         parent_iter[key], subregion, sample_uni, robustness_uni)
#                     #print('uniform sampling', sample_all[iter_key][key])

#                     if self.method == 'BO':
#                         #print('bo sample length:', n_bo)
#                         __exe_BO_ = Bayesian_Optimizer(sample_all[iter_key][key], rob_all[iter_key][key], 
#                                                         self.test_function, subregion, n_bo)
#                         subr_sample, subr_robust = __exe_BO_.Bayesian_optimization()
#                         sample_all[iter_key][key] = subr_sample
#                         rob_all[iter_key][key] = subr_robust
#                     else:
#                         subr_sample = sample_all[iter_key][key]
#                         subr_robust = rob_all[iter_key][key]
#                     #print('bo_data_length', sample_all[iter_key][key])
                    
#                     #sample_all, rob_all = merge_parent_samples_to_child(sample_all, rob_all, iteration, key,
#                         #parent_iter[key], subregion, subr_sample, subr_robust)
#                     #print('sample_all:', sample_all)
#                     #print('rob_all:', rob_all)
#                     #print('sample_all_2_model', subr_sample)
#                     sample_sign[key] = check_all_same_sign(subr_robust)
                    
#                     exe_gp = GP_model(subr_sample, subr_robust,
#                                       dim, subregion, 64)#sample_all['iter_' + str(iteration+1)][key],rob_all['iter_' + str(iteration+1)][key],
#                     avg_mu, avg_sigma, score, CI_lower, CI_upper = exe_gp.confidence_interval()
#                     #print('Posterior Region:', CI_lower, CI_upper)
#                     avg_mu_iter[key] = avg_mu
#                     avg_sigma_iter[key] = avg_sigma
#                     score_iter[key] = score
#                     #print('Subregion'+str(key)+'score :', score)
#                     theta_minus_iter, theta_plus_iter, theta_undefined = region_classify(
#                         subregion, CI_lower, CI_upper, key, theta_undefined,
#                         theta_minus_iter, theta_plus_iter)

#                 und_v = undefined_vol(theta_undefined, dim) 
#                 remain_und_v = und_v/region_vol
#                 logging.info(f"Iteration {iteration+1} completed. Remaining undefined volume proportion: {remain_und_v:.4f}")   
#                 score_unscaled[iter_key] = score_iter
#                 score_scaled[iter_key] = score_scale(avg_mu_iter, score_iter)
#                 part_number, total_subregions = score_based_partition(score_unscaled[iter_key], iteration, sample_sign)
#                 total_budget =  round(20*total_subregions*(1/(iteration+1))**(1/4))############
#                 if budget_cum + total_budget > self.budget:
#                     logging.info(f'last iteration')
#                     last_iteration = 1
#                     total_budget = self.budget - budget_cum
#                 #print('total_budget:', total_budget)
#                 budget_cum += total_budget
#                 budget_cum_dict[iter_key] = budget_cum
#                 logging.info(f"Total subregions: {total_subregions}, allocated budget for this iteration: {total_budget}, cumulative budget: {budget_cum}")
#                 mother_allocation_unscaled  = sample_allo_mother(score_iter,  total_budget, last_iteration) 
#                 mother_allocation = sample_allo_mother(score_unscaled[iter_key], total_budget, last_iteration)
#                 #print('remain_vol',  und_v/region_vol)
#                 vMinus_cum += undefined_vol(theta_minus_iter, dim) 
#                 logging.info(f"theta_minus_volume proportion: {vMinus_cum/region_vol:.4f}")   
#                 vPlus_cum += undefined_vol(theta_plus_iter, dim)
#                 logging.info(f"theta_plus_volume proportion: {vPlus_cum/region_vol:.4f}") 
#                 remain_volume[iter_key] = und_v/region_vol
#                 minus_volume[iter_key] = vMinus_cum/region_vol
#                 plus_volume[iter_key] = vPlus_cum/region_vol
        
#                 theta_plus[iter_key] = theta_plus_iter
#                 theta_minus[iter_key] = theta_minus_iter
#                 #print('avg_mu_____________',avg_mu_iter)
#                 plot_one_iteration_boxplot(avg_mu_iter, avg_sigma_iter, 
#                                            iteration+1, save_dir="iteration_plots")
#                 plot_score_comparison_bar(score_iter, score_scaled[iter_key],
#                                        iteration+1, save_dir="score_compare_plots")
#                 plot_allocation_bar( mother_allocation_unscaled, mother_allocation,
#                                     iteration+1, save_dir="allocation_plots")
#                 #print('sample_length:', sample_all['iter_' + str(iteration + 1)])
#                 if dim == 2:
#                     part_plot(theta_minus, theta_plus, theta_undefined, region, self.test_function, self.method,
#                     sample_all, rob_all, iteration+1, save_dir="partition_plots")
#                 #print('theta_minus', theta_minus_iter.keys())
#                 #print('theta_plus', theta_plus_iter.keys())
#                 #print('theta_undefined in iteration'+str([iter_key]), theta_undefined.keys())
#                 #print('parent', parent_iter)·················
#             else:
#                 logging.info("Stopping criteria met: budget limit or low undefined volume.")
#                 break
#         #print('theta_plus', theta_plus)   
#         #print('theta_minus', theta_minus) 
#         #logging.info(f"Final unscaled scores: {score_unscaled}")
#         #logging.info(f"theta_undefined: {theta_undefined}")
#         #logging.info(f"theta_undefined: {theta_undefined}")
        
       
#         logging.info(f"iteration cumulative budget used: {budget_cum_dict}")
#         logging.info(f"Remaining undefined volume proportion: {und_v/region_vol:.4f}")
#         end = time.time()
#         logging.info(f"Total run time: {end - start:.2f} seconds")
#         logging.info(f"Remaining volume proportions - Undefined: {remain_volume}, Minus: {minus_volume}, Plus: {plus_volume}")
#         return theta_minus, theta_plus, theta_undefined, budget_cum, sample_all, rob_all, tree

# if __name__ == "__main__":
#     arguments_parser = argparse.ArgumentParser(description="level-set classification")
#     arguments_parser.add_argument("-r", "--region", type=str, help="region needed to be classified, as [[,], [,], [,], ...]")
#     arguments_parser.add_argument("-m", "--method", type=str, help="sampling method, 'BO' or 'uniform_sampling' ")
#     arguments_parser.add_argument("-f", "--function", type=str, help=" target black-box function as 'X[1]+X[0]...' ")

#     args = arguments_parser.parse_args()
#     bart = Part_X(args.region, args.method, args.function, budget = 10000)
#     logging.info("Input region: {}".format(args.region))
#     region = eval(args.region)
#     #test_function = eval(args.function) 
#     theta_minus, theta_plus, theta_undefined, budget_cum, sample_all, rob_all, tree = bart.__exe__()
#     mis = compute_misclassification_mc_iter(theta_minus, theta_plus, region, args.function, n_samples=50000, seed=42)
#     logging.info(f"Misclassification estimate: {mis['V_misclassified']:.4f} (proportion: {mis['ratio_misclassified']:.4f})")
    
#     part_plot(theta_minus, theta_plus, theta_undefined, region, args.function, args.method,
#             sample_all, rob_all, 'final', save_dir="final_partition_plots")
#     logging.info("---- Process end ----")
                    

