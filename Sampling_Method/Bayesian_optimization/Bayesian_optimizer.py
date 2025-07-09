


import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from numpy import argmax
from warnings import catch_warnings, simplefilter
from sklearn.preprocessing import StandardScaler
from Sampling_Method.Uniform_random import uniform_sampling


class Bayesian_Optimizer:

    def __init__(self, X: np.array, Y: np.array, target_fun: str, subregion: list, n_bo: int):
        assert X.shape[0] == len(Y), "Number of samples in X must match length of Y"
        assert X.shape[1] == len(subregion), "Dimension of X must match subregion dimensions"
        self.X = X
        self.Y = Y
        self.target = target_fun
        self.subregion = subregion
        self.n_bo = n_bo

    def test_function(self, X):
        try:
            M = X
            return eval(self.target)
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

    def acquisition(self, Xsamples: np.array, model):
        best = min(self.Y)
        mu, std = self.surrogate(Xsamples, model)
        return self.EI(mu, std, best, xi=0.1)

    def opt_acquisition(self, model, n_b: int, i_dim):
        sbo = uniform_sampling(self.subregion, i_dim, n_b)
        scores = self.acquisition(sbo, model)
        ix = argmax(scores)
        return np.array(sbo)[ix]

    def Bayesian_optimization(self):
        i_dim = len(self.subregion[0])
        n_b = 50

        for j in range(self.n_bo):
            # --- Standardize X and Y
            scaler_X = StandardScaler().fit(self.X)
            scaler_Y = StandardScaler().fit(np.array(self.Y).reshape(-1, 1))
            X_scaled = scaler_X.transform(self.X)
            Y_scaled = scaler_Y.transform(np.array(self.Y).reshape(-1, 1)).ravel()

            # --- Define GP model
            kernel = ConstantKernel(1.0, (1e-3, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10
            )
            model.fit(X_scaled, Y_scaled)

            # --- BO Sampling
            bo_x = self.opt_acquisition(model, n_b, i_dim)
            bo_y = self.test_function(bo_x)

            self.X = np.vstack((self.X, bo_x.reshape(1, -1)))
            self.Y.append(bo_y)

        return self.X, self.Y

              
        
#b_o = Bayesian_Optimizer(s:np.array, Y:np.array, ' (M[0]**2+M[1]-11)**2+(M[0]+ M[1]**2-7)**2 -90', list: sub_r)