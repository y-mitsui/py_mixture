'''
Created on 2017/06/12

'''
import numpy as np
from scipy.misc import logsumexp
import sys

class SupervisedBernoulliMixture:
    
    def __init__(self, n_components=2, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
    
    def fit(self, sample_X, sample_y):
        self.sample_X = sample_X
        self.sample_y = sample_y
        n_samples = sample_X.shape[0]
        n_dimentions = sample_X.shape[1]
        self.poi_params = np.random.rand(self.n_components, n_dimentions) 
        self.latent_z = np.random.rand(n_samples, self.n_components)
        self.weights = np.random.rand(self.n_components)
        self.weights /= self.weights.sum()
        
        for _ in range(self.n_iter):
            self.eStep()
            self.mStep()
        return self.latent_z
    
    def log_bernoulli(self, X, parameter):
        np.clip(parameter, 1e-10, 1 - 1e-10, out=parameter)
        pdf = X * np.log(parameter) + (1 - X) * np.log((1 - parameter))
        return pdf.sum()
     
    def eStep(self):
        self.latent_z = []
        log_weights = np.log(self.weights)
        for X in self.sample_X:
            weights_probs = [] 
            for k in range(self.n_components):
                a = log_weights[k]
                b = self.log_bernoulli(X, self.poi_params[k])
                weights_probs.append(a + b)
            tot_log_likelyhood = logsumexp(weights_probs)
            z = []
            for wp in weights_probs:  
                z.append(wp - tot_log_likelyhood)
            self.latent_z.append(z)
        self.latent_z = np.exp(np.array(self.latent_z))
    
    def mStep(self):
        new_poi_params = []
        for k in range(self.poi_params.shape[0]):
            tot_latent_z = self.latent_z[:, k].sum()
            row_poi_params = []
            for d in range(self.poi_params.shape[1]):
                row_poi_params.append((self.latent_z[:, k] * self.sample_X[:, d]).sum() / tot_latent_z)
            new_poi_params.append(row_poi_params)
            self.weights[k] = tot_latent_z / self.sample_X.shape[0]
        self.poi_params = np.array(new_poi_params)
        