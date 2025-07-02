import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import qmc
from typing import Tuple, List
class GP_model:
    
    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 dim: int,
                 subregion: List[List[float]],
                 N_gp: int = 15):
        '''
        Generate the micro Gaussian Process model on a subregion

        Input: 
        X (np.array): sampling points in the subregion
        Y (np.array): corresponding robustness values
        dim (int): subregion dimension
        subregion (dict: list) : target subregion
        undefined_vol (int): undefined subregions total volumn
        
        

        Returns:
        model_lower(np.array): lower bound of confidence intervals of the subregion
        model_upper(np.array): lower bound of confidence intervals of the subregion
        score (int) : score for the subregion
          
        Raises:
            ValueError: if input shapes do not match or invalid subregion
        '''
        assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray), "X and Y must be numpy arrays"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        assert X.shape[1] == dim, f"X must have shape (n_samples, {dim})"
        assert len(subregion) == 2 and all(len(bounds) == dim for bounds in subregion), \
            "Subregion must be a list of two lists of length equal to dimension"

        self.X = X
        self.Y = Y
        self.N_gp = N_gp
        self.dim = dim
        self.subregion = subregion
        if N_gp <= 0:
            raise ValueError("Number of GP test points (N_gp) must be greater than 0")
       
       

    def confidence_interval(self) -> Tuple[float, float, float, float, float]:
        """
        Fit a GP model and estimate the confidence interval over the subregion.

        Returns:
            avg_mu (float): Mean predicted value over the region
            avg_sigma (float): Mean standard deviation over the region
            score (float): avg_mu / avg_sigma as uncertainty-based score
            CI_low (float): Lower bound of 95% confidence interval
            CI_upper (float): Upper bound of 95% confidence interval

        Raises:
            RuntimeError: if GP model fails or prediction errors occur
        """
        try:
            kernel = C(1.0, (1e-3, 1e3)) * R(10, (1e-2, 1e2))
            gp = GaussianProcessRegressor()#kernel=kernel, n_restarts_optimizer = 7)
            gp.fit(self.X, self.Y)
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            sobol_unit = sampler.random(n=self.N_gp)
            bounds = np.array(self.subregion)
            sobol_scaled = qmc.scale(sobol_unit, bounds[0], bounds[1])
            #N_gp_fal = int(self.fal_num*vol(self.subregion,self.dim)/self.undefined_vol) +\
                #self.N_gp
            #sample_gp = uniform_sampling(self.subregion,self.dim, N_gp_fal)
            y_pred, sigma = gp.predict(sobol_scaled, return_std=True)
            # print('y_pred', y_pred_st)
            # print('sigma', sigma_st) 
            #idxs = np.random.randint(0, N_gp_fal, self.N_gp)##
            #avg_statis = [sum(y_pred_st)/15, sum(sigma_st)/15]
            #y_pred_s = y_pred_st[idxs]
            #sigma_s = sigma_st[idxs]
            avg_mu = np.mean(y_pred)
            avg_sigma = np.mean(sigma)
            score = avg_mu / (avg_sigma + 1e-8)
            # pred_min = min(y_pred_s)#find maximum and minimum values of predicted values
            # pred_max = max(y_pred_s)
            # sigma_ss = max(sigma_s)
            CI_low = min(y_pred - 1.96*sigma)
            CI_upper = max(y_pred + 1.96*sigma)
            # CI_low = pred_min - 1.96*sigma_ss#[i_s]
            # CI_upper = pred_max + 1.96*sigma_ss#[i_ss]
            return avg_mu, avg_sigma, score, CI_low, CI_upper
         
        except Exception as e:
            raise RuntimeError(f"Failed to compute GP confidence interval: {e}")