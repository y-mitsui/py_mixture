'''
Created on 2017/06/12

'''
import numpy as np
from scipy.misc import logsumexp
import sys
from scipy import optimize

class SupervisedBernoulliMixture:
    
    def __init__(self, n_components=2, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
    
    def fit(self, sample_X, sample_y):
        self.sample_X = sample_X
        self.sample_y = sample_y
        self.n_class = int(np.max(sample_y)) + 1
        n_samples = sample_X.shape[0]
        n_dimentions = sample_X.shape[1]
        self.poi_params = np.random.rand(self.n_components, n_dimentions) 
        self.latent_z = np.random.rand(n_samples, self.n_components)
        self.weights = np.random.rand(self.n_components)
        self.weights /= self.weights.sum()
        self.supervived_params = np.random.rand(self.n_class, self.n_components)
        self.slack_params = np.random.rand(n_samples)
        
        for _ in range(self.n_iter):
            self.eStep()
            self.mStep()
        return self.latent_z
    
    def log_bernoulli(self, X, parameter):
        np.clip(parameter, 1e-10, 1 - 1e-10, out=parameter)
        pdf = X * np.log(parameter) + (1 - X) * np.log((1 - parameter))
        return pdf.sum()
    
    def eStep(self):
        new_z = []
        log_weights = np.log(self.weights)
        for d in range(self.sample_X.shape[0]):
            X = self.sample_X[d]
            weights_probs = []
            for k in range(self.n_components):
                a = log_weights[k]
                b = self.log_bernoulli(X, self.poi_params[k])
                components_sum = np.sum(np.exp(np.dot(self.supervived_params, self.latent_z[d])) * self.supervived_params[:, k])
                c = self.supervived_params[int(self.sample_y[d]), k] - 1 / self.slack_params[d] * components_sum
                weights_probs.append(a + b + c)
            tot_log_likelyhood = logsumexp(weights_probs)
            z = []
            for wp in weights_probs:
                z.append(wp - tot_log_likelyhood)
            
            new_z.append(z)
        self.latent_z = np.exp(np.array(new_z))
        print "self.latent_z", self.latent_z.shape
    
    def gradient(self, theta, *args):
        r = []
        for i in range(self.n_class):
            target_idx = self.sample_y == i
            tmp = np.exp(np.dot(self.supervived_params[i], self.latent_z[target_idx].T)) * self.latent_z[target_idx].T
            val = np.sum(self.latent_z[target_idx] - 1 / self.slack_params[target_idx].reshape(-1, 1) * tmp.T, 0)
            r.append(val)
        return np.array(r).flatten()
        
    def J(self, theta, *args):
        return 0
    
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
        init_theta = self.supervived_params.flatten()
        best_params = optimize.fmin_cg(self.J, init_theta, fprime=self.gradient)
        self.supervived_params = best_params.reshape(self.supervived_params.shape[0], self.supervived_params.shape[1])
        for d in range(self.slack_params.shape[0]):
            self.slack_params[d] = np.sum(np.exp(np.dot(self.supervived_params, self.latent_z[d])))
            
            
        