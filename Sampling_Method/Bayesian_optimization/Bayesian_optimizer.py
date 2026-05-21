import numpy as np
from numpy import argmax
from warnings import catch_warnings, simplefilter
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

class Bayesian_Optimizer:
    """
    Level-set–oriented Bayesian Optimization.
    - Focus sampling near the level set f(x) = tau (default tau=0).
    - Supports three acquisitions: BAND, STRADDLE, ENTROPY.
    - Standardizes X only (Y normalization handled by GP normalize_y=True).
    - Avoids duplicate/very-close samples via a minimum distance constraint.
    """
    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 target_fun: callable,
                 subregion: list,
                 n_bo: int,
                 acq_mode: str = "STRADDLE",
                 delta: float = None,
                 kappa: float = 2.0,
                 entropy_beta: float = 0.0,
                 min_dist: float = 1e-3,
                 n_candidates: int = 32,
                 tau: float = 0.0,
                 local_k: int = 120):
        assert X.shape[0] == len(Y)
        assert X.shape[1] == len(subregion)

        self.X = np.array(X, dtype=float)
        self.Y = list(map(float, Y))
        self.target_fun = target_fun
        self.subregion = np.array(subregion, dtype=float)
        self.n_bo = int(n_bo)

        self.acq_mode = acq_mode.upper()
        self.delta = delta
        self.kappa = float(kappa)
        self.entropy_beta = float(entropy_beta)
        self.min_dist = float(min_dist)
        self.n_candidates = int(n_candidates)
        self.tau = float(tau)
        self.local_k = int(local_k)

        self.model = None
        self._l = self.subregion[:, 0].astype(float)
        self._u = self.subregion[:, 1].astype(float)
        self._scale = np.maximum(self._u - self._l, 1e-12)
    
    def test_function(self, X):
        try:
            M = X
            return self.target_fun(M) if callable(self.target_fun) else eval(self.target_fun)
        except Exception as e:
            raise ValueError(f"Error evaluating target function: {e}")

    def _scale_X(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self._l) / self._scale

    def fit_surrogate(self):
        """
        Fast GP fit:
        - fixed bounds-based scaling
        - local subset only
        - no hyperparameter restarts
        - optimizer=None for speed
        """
        Xs_all = self._scale_X(self.X)
        Ys_all = np.asarray(self.Y, dtype=float)

        if len(Ys_all) > self.local_k:
            center = Xs_all[-1]
            dists = np.linalg.norm(Xs_all - center, axis=1)
            idx = np.argsort(dists)[:self.local_k]
            Xs = Xs_all[idx]
            Ys = Ys_all[idx]
        else:
            Xs = Xs_all
            Ys = Ys_all

        d = Xs.shape[1]

        # fixed kernel parameters = much faster
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=0.25 * np.ones(d),
            length_scale_bounds="fixed",
            nu=2.5,
            
        )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=1e-4,
            optimizer=None,
            copy_X_train=False, 
            n_restarts_optimizer= 10)

        with catch_warnings():
            simplefilter("ignore")
            self.model.fit(Xs, Ys)

    def surrogate(self, Xsamples: np.ndarray):
        Xs = self._scale_X(Xsamples)
        with catch_warnings():
            simplefilter("ignore")
            mu, std = self.model.predict(Xs, return_std=True)
        std = np.maximum(std, 1e-12)
        return mu, std

    def acq_band_prob(self, mu, std):
        delta = self.delta if self.delta is not None else np.median(std)
        lo = (self.tau - delta - mu) / std
        hi = (self.tau + delta - mu) / std
        return norm.cdf(hi) - norm.cdf(lo)

    def acq_straddle(self, mu, std):
        return self.kappa * std - np.abs(mu - self.tau)

    def acq_entropy(self, mu, std):
        z = (mu - self.tau) / std
        p = norm.cdf(z)
        eps = 1e-12
        H = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
        if self.entropy_beta > 0:
            H = H * (std ** self.entropy_beta)
        return H

    def sample_candidates(self, n: int):
        d = self.X.shape[1]
        sampler = qmc.Sobol(d=d, scramble=True)
        U = sampler.random(n)
        l, u = self.subregion[:, 0], self.subregion[:, 1]
        return qmc.scale(U, l, u)

    def pick_next(self):
        cand = self.sample_candidates(self.n_candidates)
        mu, std = self.surrogate(cand)

        if self.acq_mode == "BAND":
            scores = self.acq_band_prob(mu, std)
        elif self.acq_mode == "STRADDLE":
            scores = self.acq_straddle(mu, std)
        elif self.acq_mode == "ENTROPY":
            scores = self.acq_entropy(mu, std)
        else:
            raise ValueError("acq_mode must be one of: 'BAND', 'STRADDLE', 'ENTROPY'.")

        if self.min_dist > 0 and len(self.X) > 0:
            for i, x in enumerate(cand):
                if np.min(np.linalg.norm(self.X - x, axis=1)) < self.min_dist:
                    scores[i] = -np.inf

        ix = argmax(scores)
        return cand[ix]

    def Bayesian_optimization(self):
        d = len(self.subregion)
        for _ in range(self.n_bo):
            self.fit_surrogate()
            x_next = self.pick_next()
            y_next = self.test_function(x_next)
            self.X = np.vstack((self.X, x_next.reshape(1, d)))
            self.Y.append(float(y_next))
        return np.array(self.X), np.array(self.Y)




# import numpy as np
# from numpy import argmax
# from warnings import catch_warnings, simplefilter
# from scipy.stats import norm, qmc
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Matern, ConstantKernel
# from sklearn.preprocessing import StandardScaler

# class Bayesian_Optimizer:
#     """
#     Level-set–oriented Bayesian Optimization.
#     - Focus sampling near the level set f(x) = tau (default tau=0).
#     - Supports three acquisitions: BAND, STRADDLE, ENTROPY.
#     - Standardizes X only (Y normalization handled by GP normalize_y=True).
#     - Avoids duplicate/very-close samples via a minimum distance constraint.
#     """

#     def __init__(self,
#                  X: np.ndarray,
#                  Y: np.ndarray,
#                  target_fun: callable,
#                  subregion: list,
#                  n_bo: int,
#                  acq_mode: str = "BAND",     # "BAND" | "STRADDLE" | "ENTROPY"
#                  delta: float = None,         # BAND bandwidth; None = adaptive
#                  kappa: float = 2.0,          # STRADDLE exploration weight
#                  entropy_beta: float = 0.0,   # ENTROPY: multiply H by sigma^beta
#                  min_dist: float = 1e-3,      # min L2 distance to existing points
#                  n_candidates: int = 16,    # number of Sobol candidate points
#                  tau: float = 0.0):           # target level set
#         assert X.shape[0] == len(Y), "Number of samples in X must match length of Y"
#         assert X.shape[1] == len(subregion), "Dimension of X must match subregion dimensions"

#         self.X = np.array(X, dtype=float)
#         self.Y = list(map(float, Y))
#         self.target_fun = target_fun          # keep original structure (used by test_function)
#         self.subregion = np.array(subregion, dtype=float)  # [[l1,u1],[l2,u2],...]
#         self.n_bo = int(n_bo)

#         self.acq_mode = acq_mode.upper()
#         self.delta = delta
#         self.kappa = float(kappa)
#         self.entropy_beta = float(entropy_beta)
#         self.min_dist = float(min_dist)
#         self.n_candidates = int(n_candidates)
#         self.tau = float(tau)

#         self.scaler_X = None
#         self.model = None
        

#     # ---- test function: keep your original structure (eval on self.target_fun) ----
#     def test_function(self, X):
#         """
#         Evaluate target function at X by using the original structure.
#         self.target_fun is expected to be an expression-string using 'M' or a callable.
#         If it's a callable, you may adapt: return self.target_fun(X).
#         """
#         try:
#             M = X
#             return self.target_fun(M) if callable(self.target_fun) else eval(self.target_fun)
#         except Exception as e:
#             raise ValueError(f"Error evaluating target function: {e}")

#     # ---- GP surrogate ----
#     def fit_surrogate(self, local_k=300):
#         """Fit a GP surrogate on standardized X. Y normalization is handled internally."""
#         self.scaler_X = StandardScaler().fit(self.X)
#         Xs_all = self.scaler_X.transform(self.X)
#         Ys_all = np.asarray(self.Y, dtype=float)

#         if len(Ys_all) > local_k:
#             center = Xs_all[-1]   # 也可以换成子region中心
#             dists = np.linalg.norm(Xs_all - center, axis=1)
#             idx = np.argsort(dists)[:local_k]
#             Xs = Xs_all[idx]
#             Ys = Ys_all[idx]
#         else:
#             Xs = Xs_all
#             Ys = Ys_all


#         # self.scaler_X = StandardScaler().fit(self.X)
#         # Xs = self.scaler_X.transform(self.X)

#         #d = self.X.shape[1]
#         d = Xs.shape[1]
#         kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), nu=2.5)

#         #ConstantKernel(1.0, (1e-3, 1e4)) * Matern(length_scale=np.ones(d),
#                                                            #length_scale_bounds=(1e-2, 1e3),
#                                                            #nu=2.5)
#         self.model = GaussianProcessRegressor(kernel=kernel,
#                                               normalize_y=True,
#                                               alpha=1e-4,                # numerical jitter
#                                               n_restarts_optimizer=0)
#         with catch_warnings():
#             simplefilter("ignore")
#             self.model.fit(Xs, Ys)
#         # with catch_warnings():
#         #     simplefilter("ignore")
#         #     self.model.fit(Xs, np.array(self.Y))

#     def surrogate(self, Xsamples: np.ndarray):
#         """Predict GP mean/std at Xsamples (Xsamples in original scale; we standardize inside)."""
#         Xs = self.scaler_X.transform(Xsamples)
#         with catch_warnings():
#             simplefilter("ignore")
#             mu, std = self.model.predict(Xs, return_std=True)
#         std = np.maximum(std, 1e-12)
#         return mu, std

#     # ---- Level-set acquisition functions ----
#     def acq_band_prob(self, mu, std):
#         """
#         Band Probability (LSE-style):
#         P(|f - tau| <= delta) = Phi((tau+delta-mu)/std) - Phi((tau-delta-mu)/std).
#         If delta is None, use an adaptive bandwidth based on median(std).
#         """
#         delta = self.delta if self.delta is not None else np.median(std)
#         lo = (self.tau - delta - mu) / std
#         hi = (self.tau + delta - mu) / std
#         return norm.cdf(hi) - norm.cdf(lo)

#     def acq_straddle(self, mu, std):
#         """
#         Straddle heuristic:
#         score = kappa * std - |mu - tau|.
#         Encourages high uncertainty while keeping the mean near the level set.
#         """
#         return self.kappa * std - np.abs(mu - self.tau)

#     def acq_entropy(self, mu, std):
#         """
#         Sign entropy around the level set:
#         p = P(f > tau) = Phi((mu - tau) / std)
#         H = -p log p - (1-p) log (1-p)
#         Optionally multiply by std^beta to emphasize exploration.
#         """
#         z = (mu - self.tau) / std
#         p = norm.cdf(z)
#         eps = 1e-12
#         H = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
#         if self.entropy_beta > 0:
#             H = H * (std ** self.entropy_beta)
#         return H

#     # ---- Candidate generation and selection ----
#     def sample_candidates(self, n: int):
#         """Generate Sobol candidates in the original scale of subregion bounds."""
#         d = self.X.shape[1]
#         sampler = qmc.Sobol(d=d, scramble=True)
#         U = sampler.random(n)
#         l, u = self.subregion[:, 0], self.subregion[:, 1]
#         return qmc.scale(U, l, u)

#     def pick_next(self):
#         """Pick the next point by maximizing the chosen level-set acquisition over candidates."""
#         cand = self.sample_candidates(self.n_candidates)
#         mu, std = self.surrogate(cand)

#         if self.acq_mode == "BAND":
#             scores = self.acq_band_prob(mu, std)
#         elif self.acq_mode == "STRADDLE":
#             scores = self.acq_straddle(mu, std)
#         elif self.acq_mode == "ENTROPY":
#             scores = self.acq_entropy(mu, std)
#         else:
#             raise ValueError("acq_mode must be one of: 'BAND', 'STRADDLE', 'ENTROPY'.")

#         # Simple min-distance constraint to avoid duplicates or near-duplicates
#         if self.min_dist > 0:
#             for i, x in enumerate(cand):
#                 if np.min(np.linalg.norm(self.X - x, axis=1)) < self.min_dist:
#                     scores[i] = -np.inf

#         ix = argmax(scores)
#         return cand[ix]

#     # ---- Main BO loop ----
#     def Bayesian_optimization(self):
#         """
#         Run n_bo BO iterations:
#         - Fit GP on current data (with X standardized)
#         - Pick next point focusing on the level set f(x) = tau
#         - Evaluate target function using the original test_function structure
#         - Append (x, y) to the dataset
#         """
#         d = len(self.subregion)  # correct dimension
#         for _ in range(self.n_bo):
#             self.fit_surrogate()
#             x_next = self.pick_next()
#             y_next = self.test_function(x_next)  # keep original evaluation style
#             self.X = np.vstack((self.X, x_next.reshape(1, d)))
#             self.Y.append(y_next)
#         return np.array(self.X), np.array(self.Y)
    
# import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Matern, ConstantKernel
# from scipy.stats import norm
# from numpy import argmax
# from warnings import catch_warnings, simplefilter
# from sklearn.preprocessing import StandardScaler
# from scipy.stats import qmc



# class Bayesian_Optimizer:
#     def __init__(self, X: np.ndarray, Y: np.ndarray, target_fun: callable, subregion: list, n_bo: int):
#         assert X.shape[0] == len(Y), "Number of samples in X must match length of Y"
#         assert X.shape[1] == len(subregion), "Dimension of X must match subregion dimensions"
#         self.X = X
#         self.Y = list(Y)
#         self.target_fun = target_fun  # should be a callable function
#         self.subregion = subregion
#         self.n_bo = n_bo

#     def test_function(self, X):
#         try:
#             M = X
#             return eval(self.target_fun)
#         except Exception as e:
#             raise ValueError(f"Error evaluating target function: {e}")

#     def surrogate(self, Xsamples, model):
#         with catch_warnings():
#             simplefilter("ignore")
#             return model.predict(Xsamples, return_std=True)

#     def EI(self, mean, std, y_min, xi=0.1):
#         std = np.maximum(std, 1e-9)
#         a = (y_min - mean - xi)
#         z = a / std
#         return a * norm.cdf(z) + std * norm.pdf(z)
    
   

#     # def ls_acquisition(self, Xsamples: np.ndarray, model, tau: float = 0.0, delta: float = 0.05):
#     #     mu, std = self.surrogate(Xsamples, model)
#     #     #std = np.maximum(std, 1e-8) 
        
#     #     lower = (tau - delta - mu) / std
#     #     upper = (tau + delta - mu) / std
#     #     prob = norm.cdf(upper) - norm.cdf(lower)
        
#     #     return prob
    
#     def ls_acquisition(self, Xsamples, model, tau=0.0):
#         mu, std = self.surrogate(Xsamples, model)
#         #print(mu, std)
#         std = np.maximum(std, 1e-8)
#         score = norm.pdf(mu, loc=tau, scale=std)  # 
#         return score
    
#     def acquisition(self, Xsamples: np.ndarray, model):
#         # Use y_closest_to_zero instead of minimum
#         scaler_Y = StandardScaler().fit(np.array(self.Y).reshape(-1, 1))
#         y_target = min(self.Y, key=lambda y: abs(y))
#         y_target_scaled = scaler_Y.transform(np.array([[y_target]])).ravel()[0]
#         mu, std = self.surrogate(Xsamples, model)
#         return self.EI(mu, std, y_target_scaled, xi=0.1)



#     def opt_acquisition(self, model, n_b: int, i_dim):
        
#         sampler = qmc.Sobol(d=i_dim, scramble=True)
#         sobol_unit = sampler.random(n=n_b)  

#         bounds = np.array(self.subregion)  # shape: (dim, 2)
#         l_bounds = bounds[:, 0]
#         u_bounds = bounds[:, 1]
#         sobol_scaled = qmc.scale(sobol_unit, l_bounds, u_bounds)

#         scores = self.ls_acquisition(sobol_scaled, model, tau= 0.0)
        
#         ix = argmax(scores)
#         return np.array(sobol_scaled[ix])

#     def Bayesian_optimization(self):
#         i_dim = len(self.subregion[0])
#         n_b = 64  # candidate points for acquisition maximization

#         for j in range(self.n_bo):
#             # Standardize X and Y
#             scaler_X = StandardScaler().fit(self.X)
#             scaler_Y = StandardScaler().fit(np.array(self.Y).reshape(-1, 1))
#             X_scaled = scaler_X.transform(self.X)
#             Y_scaled = scaler_Y.transform(np.array(self.Y).reshape(-1, 1)).ravel()

#             # Define GP model with proper kernel
#             kernel = ConstantKernel(1.0, (1e-3, 1e4)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
#             model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
#                 #kernel=kernel,
#                 #normalize_y=True,
#                 #n_restarts_optimizer=10
#             #)
#             model.fit(X_scaled, Y_scaled)

#             # BO Sampling
#             bo_x = self.opt_acquisition(model, n_b, i_dim)
#             bo_y = self.test_function(bo_x)

#             # Update dataset
#             self.X = np.vstack((self.X, bo_x.reshape(1, -1)))
#             self.Y.append(bo_y)

#         return np.array(self.X), np.array(self.Y)
    