from warnings import catch_warnings
from warnings import simplefilter
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from numpy import argmax


class Bayesian_Optimizer:
    
    def __init__(self, X, Y, target_fun:str, subregion):
        self.X = X
        self.Y = Y
        self.target = target_fun
        self.subregion = subregion
        
 
   
    def test_function(self, M):
        return eval(self.target)

    def surrogate(self, Xsamples, model):
        '''
        predict for the Predicted values of BO
        Parameters:
            model
            X(np.array)
        

        Returns:
            predicted values

        '''
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return model.predict(Xsamples, return_std=True)
        
    def PI(self, mean, std, best):
        z = (best - mean + 0.5)/std#- xi)/std
        return norm.cdf(z)
    
    def EI(self, mean, std, y_max):
        a = (y_max - mean - 2)#- xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)
    
    
    def acquisition(self, Xsamples: np.array, model):
        '''
        calculate the best surrogate score found so far
        Parameters:
            model; the GP models
            X(np.array): sample points 
            Xsample(np.array): new sample points for BO
        

        Returns:
            sample probabiility of each sample points

        '''
        # calculate the best surrogate score found so far
        
        best = min(self.Y)
        # calculate mean and stdev via surrogate function
        mu, std = self.surrogate(Xsamples, model)
        # calculate the probability of improvement
        probs = self.EI(mu, std, best)
        return probs
    
    def opt_acquisition(self, model, n_b:int, i_dim):
        '''
        get the sample points
        Parameters:
            X(np.array): sample points 
            y(np.array): corresponding rebustness values
            model: the GP models 
            n_b(int): the number of sample points to construct the robustness values
            i_dim(int): dimesion
            sub_r(list): subregions
        

        Returns:
            min_bo(np.array): the new sample points by BO

        '''
        # random search, generate random samples
        
        sample_emp = np.empty(shape = (i_dim, n_b))
        for k in range(i_dim):
            sample_emp[k] = np.random.uniform(self.subregion[k][0],self.subregion[k][1],n_b)
        sbo = sample_emp.T
        # calculate the acquisition function for each sample
        scores = self.acquisition(sbo, model)
        # locate the index of the largest scores
        ix = argmax(scores)
        bo_x = np.array(sbo)[ix]
        return bo_x

    def Bayesian_optimization(self):
        '''
        calculate the best surrogate score found so far
        Parameters:
            s(np.array): sample points
            Y(list): robustness values
            n_bo(int): number of Bayesian Optimization sampling
            i_dim(int): dimension
            sub_r(list): subregions
            test_function: the test function
            n_b(int): number of points to construct GPs in BO (default = 100)


        Returns:
            S_BO(np.array): updated sample points
            Y(list): corresponding updated robuseness values

        '''
        i_dim = len(self.subregion[0])
        n_b = 50
        n_bo = 5
        for j in range(n_bo):
            model = GaussianProcessRegressor(
             kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=len(self.X),
            random_state= None,)
            model.fit(self.X, self.Y)
            # select the next point to sample
            bo_x = self.opt_acquisition(model, n_b, i_dim)
            # sample the point
            bo_y = self.test_function(bo_x)
            # add the data to the dataset
            self.X.append(bo_x)
            self.Y.append(bo_y)
        
#b_o = Bayesian_Optinizer(s:np.array, Y:np.array, ' (M[0]**2+M[1]-11)**2+(M[0]+ M[1]**2-7)**2 -90', list: sub_r)