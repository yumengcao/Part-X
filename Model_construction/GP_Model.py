import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as R, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
from typing import Tuple, List
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
class GP_model:
    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 dim: int,
                 subregion: List[Tuple[float, float]],
                 N_gp: int = 32):
        """
        Generate the micro Gaussian Process model on a subregion.
        """
        assert isinstance(X, np.ndarray) and isinstance(Y, np.ndarray), "X and Y must be numpy arrays"
        assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples"
        assert X.shape[1] == dim, f"X must have shape (n_samples, {dim})"
        assert all(isinstance(b, tuple) and len(b) == 2 for b in subregion), \
            "subregion must be a list of (low, high) tuples"

        self.dim = dim
        self.N_gp = N_gp
        self.subregion = subregion

        # Standardize X and Y
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.X_scaled = self.scaler_X.fit_transform(X)
        self.Y_scaled = self.scaler_Y.fit_transform(Y.reshape(-1, 1)).ravel()

        # Define kernel
        kernel = C(1.0, (1e-2, 1e4)) * R(length_scale=np.ones(self.dim), length_scale_bounds=(1e-1, 1e3))

        self.gp = GaussianProcessRegressor( kernel=kernel,n_restarts_optimizer=5,normalize_y=False )

        self.gp.fit(self.X_scaled, self.Y_scaled)

    def confidence_interval(self) -> Tuple[float, float, float, float, float]:
        """
        Predict with GP and return mean, std, score, and CI over the subregion.
        """
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            sobol_unit = sampler.random(n=self.N_gp)
            bounds = np.array(self.subregion)
            l_bounds = bounds[:, 0]
            u_bounds = bounds[:, 1]

            if not np.all(l_bounds < u_bounds):
                raise ValueError(f"Inconsistent bounds! l_bounds: {l_bounds}, u_bounds: {u_bounds}")

            sobol_scaled = qmc.scale(sobol_unit, l_bounds, u_bounds)
            sobol_scaled_std = self.scaler_X.transform(sobol_scaled)

            y_pred_std, sigma_std = self.gp.predict(sobol_scaled_std, return_std=True)
            y_pred = self.scaler_Y.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()
            sigma = sigma_std * self.scaler_Y.scale_[0]

            avg_mu = np.mean(y_pred)
            avg_sigma = np.mean(sigma)
            score = np.abs(avg_mu) / (avg_sigma + 1e-2)
            if score<10:
                print('score:', score)
                print('avg_mu:', avg_mu)
                print('avg_sigma:', avg_sigma)
            CI_low = np.min(y_pred - 1.96 * sigma)
            CI_upper = np.max(y_pred + 1.96 * sigma)

            return avg_mu, avg_sigma, score, CI_low, CI_upper

        except Exception as e:
            raise RuntimeError(f"Failed to compute GP confidence interval: {e}")
