
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from numpy import argmax
from warnings import catch_warnings, simplefilter
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc



class Bayesian_Optimizer:
    def __init__(self, X: np.ndarray, Y: np.ndarray, target_fun: callable, subregion: list, n_bo: int):
        assert X.shape[0] == len(Y), "Number of samples in X must match length of Y"
        assert X.shape[1] == len(subregion), "Dimension of X must match subregion dimensions"
        self.X = X
        self.Y = list(Y)
        self.target_fun = target_fun  # should be a callable function
        self.subregion = subregion
        self.n_bo = n_bo

    def test_function(self, X):
        try:
            M = X
            return eval(self.target_fun)
        except Exception as e:
            raise ValueError(f"Error evaluating target function: {e}")

    def surrogate(self, Xsamples, model):
        with catch_warnings():
            simplefilter("ignore")
            return model.predict(Xsamples, return_std=True)

    def EI(self, mean, std, y_min, xi=0.1):
        std = np.maximum(std, 1e-9)
        a = (y_min - mean - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)
    
   

    # def ls_acquisition(self, Xsamples: np.ndarray, model, tau: float = 0.0, delta: float = 0.05):
    #     mu, std = self.surrogate(Xsamples, model)
    #     #std = np.maximum(std, 1e-8) 
        
    #     lower = (tau - delta - mu) / std
    #     upper = (tau + delta - mu) / std
    #     prob = norm.cdf(upper) - norm.cdf(lower)
        
    #     return prob
    
    def ls_acquisition(self, Xsamples, model, tau=0.0):
        mu, std = self.surrogate(Xsamples, model)
        #print(mu, std)
        std = np.maximum(std, 1e-8)
        score = norm.pdf(mu, loc=tau, scale=std)  # 
        return score
    
    def acquisition(self, Xsamples: np.ndarray, model):
        # Use y_closest_to_zero instead of minimum
        scaler_Y = StandardScaler().fit(np.array(self.Y).reshape(-1, 1))
        y_target = min(self.Y, key=lambda y: abs(y))
        y_target_scaled = scaler_Y.transform(np.array([[y_target]])).ravel()[0]
        mu, std = self.surrogate(Xsamples, model)
        return self.EI(mu, std, y_target_scaled, xi=0.1)



    def opt_acquisition(self, model, n_b: int, i_dim):
        
        sampler = qmc.Sobol(d=i_dim, scramble=True)
        sobol_unit = sampler.random(n=n_b)  

        bounds = np.array(self.subregion)  # shape: (dim, 2)
        l_bounds = bounds[:, 0]
        u_bounds = bounds[:, 1]
        sobol_scaled = qmc.scale(sobol_unit, l_bounds, u_bounds)

        scores = self.ls_acquisition(sobol_scaled, model, tau= 0.0)
        
        ix = argmax(scores)
        return np.array(sobol_scaled[ix])

    def Bayesian_optimization(self):
        i_dim = len(self.subregion[0])
        n_b = 64  # candidate points for acquisition maximization

        for j in range(self.n_bo):
            # Standardize X and Y
            scaler_X = StandardScaler().fit(self.X)
            scaler_Y = StandardScaler().fit(np.array(self.Y).reshape(-1, 1))
            X_scaled = scaler_X.transform(self.X)
            Y_scaled = scaler_Y.transform(np.array(self.Y).reshape(-1, 1)).ravel()

            # Define GP model with proper kernel
            kernel = ConstantKernel(1.0, (1e-3, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
                #kernel=kernel,
                #normalize_y=True,
                #n_restarts_optimizer=10
            #)
            model.fit(X_scaled, Y_scaled)

            # BO Sampling
            bo_x = self.opt_acquisition(model, n_b, i_dim)
            bo_y = self.test_function(bo_x)

            # Update dataset
            self.X = np.vstack((self.X, bo_x.reshape(1, -1)))
            self.Y.append(bo_y)

        return np.array(self.X), np.array(self.Y)
    