'''
Created on 2017/06/12

'''
import numpy as np
from scipy.misc import logsumexp
import sys

class BernoulliMixture:
    
    def __init__(self, n_components=2, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
    
    def fit(self, sample_X):
        self.sample_X = sample_X
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
        pdf = X * np.log(parameter) + (1 - X) * (1 - np.log(parameter))
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
        
if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    n_sample = 20
    n_dim = 2
    n_components = 2
    
    weights = np.random.dirichlet([1] * 2)
    ber_params = np.random.uniform(1e-3, 1 - 1e-3, size=(n_components, n_dim))
    print "ber_params", ber_params
    latent_z = np.random.multinomial(1, weights, size=n_sample)
    sample_X = []
    for z in latent_z:
        print np.where(z==1)[0][0]
        X = np.random.binomial(1, ber_params[np.where(z==1)[0][0]])
        sample_X.append(X)
    #plt.hist(sample_X)
    #plt.show()
    sample_X = np.array(sample_X)
    poisson_mixture = BernoulliMixture(n_components, 100)
    poisson_mixture.fit(sample_X)
    print poisson_mixture.poi_params