import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern as M, RBF as R, ConstantKernel as C 
from Sampling_Method.Uniform_random import uniform_sampling
from Functional.__tools__ import vol
from sklearn.gaussian_process.kernels import Matern as M, RBF as R, ConstantKernel as C 

class GP_model:
    
    def __init__(self, X:np.array, Y: np.array, dim: int, subregion, undefined_vol: int, N_gp: int = 50, fal_num:int = 10):
        self.X = X
        self.Y = Y
        self.N_gp = N_gp
        self.dim = dim
        self.subregion = subregion
        self.undefined_vol = undefined_vol
        self.fal_num = fal_num
       

    def confidence_interval(selfdd):
        kernel = C(1.0, (1e-3, 1e3)) * R(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        gp.fit(self.X, self.Y)
        N_gp_fal = int(self.fal_num*vol([self.subregion],self.dim)/self.undefined_vol) + self.N_gp
        sample_gp = uniform_sampling(self.subregion, N_gp_fal, self.dim)
        y_pred_st, sigma_st = gp.predict(sample_gp, return_std=True)  
        idxs = np.random.randint(0, N_gp_fal, self.N_gp)##
        score = sum(y_pred_st[idxs])
        CI_low = min(y_pred_st[idxs]-1.96*sigma_st[idxs])
        CI_upper = max(y_pred_st[idxs]+1.96*sigma_st[idxs])
        return score, CI_low, CI_upper