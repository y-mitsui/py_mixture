'''
Created on 2017/06/12

'''
import numpy as np
from scipy.misc import logsumexp
import sys
from scipy import optimize
import warnings

class SupervisedBernoulliMixture:
    
    def __init__(self, n_components=2, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
        self.n_flag = 0
    
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
        
        for i in range(self.n_iter):
            self.eStep()
            self.mStep()
            print "%d / %d"%(i, self.n_iter) 
        return self.latent_z
    
    def log_bernoulli(self, X, parameter):
        np.clip(parameter, 1e-10, 1 - 1e-10, out=parameter)
        pdf = X * np.log(parameter) + (1 - X) * np.log((1 - parameter))
        return pdf.sum()
    
    def eStep(self):
        new_z = []
        log_weights = np.log(self.weights)
        loglikelyhood = 0.0;
        for d in range(self.sample_X.shape[0]):
            X = self.sample_X[d]
            weights_probs = []
            for k in range(self.n_components):
                a = log_weights[k]
                b = self.log_bernoulli(X, self.poi_params[k])
                #components_sum = np.sum(np.exp(np.dot(self.supervived_params, self.latent_z[d].reshape(-1, 1))) * self.supervived_params[:, k].reshape(-1, 1))
                components_sum = np.sum(np.exp(self.supervived_params[:, k]))
                c = self.supervived_params[int(self.sample_y[d]), k] - 1 / self.slack_params[d] * components_sum
                weights_probs.append(a + b + 0.0 * c)
            tot_log_likelyhood = logsumexp(weights_probs)
            loglikelyhood += tot_log_likelyhood
            z = []
            for wp in weights_probs:
                z.append(wp - tot_log_likelyhood)
            
            new_z.append(z)
        print "loglikelyhood", loglikelyhood
        self.latent_z = np.exp(np.array(new_z))
    
    def gradient(self, theta, *args):
        s_params = theta.reshape(self.supervived_params.shape[0], self.supervived_params.shape[1])
        r = []
        for i in range(self.n_class):
            target_idx = self.sample_y == i
            #print "np.dot(s_params[i], self.latent_z[target_idx].T)", np.dot(s_params[i], self.latent_z[target_idx].T)
            #tmp = np.exp(np.dot(s_params[i], self.latent_z[target_idx].T)).reshape(1, -1) * self.latent_z[target_idx].T
            tmp = self.latent_z[target_idx] * np.exp(s_params[i, :]).reshape(1, -1)
            val = np.sum(self.latent_z[target_idx] - 1 / self.slack_params[target_idx].reshape(-1, 1) * tmp, 0)
            r.append(val)
        #print np.array(r)
        #print ""
        self.n_flag += 1
        return -np.array(r).flatten()
        
    def J(self, theta, *args):
        s_params = theta.reshape(self.supervived_params.shape[0], self.supervived_params.shape[1])
        r = 0.0
        for d in range(self.sample_y.shape[0]):
            tmp = np.sum(np.exp(np.dot(s_params, self.latent_z[d].reshape(-1, 1))))
            tmp3 = np.sum(self.latent_z[d] * tmp)
            tmp2 = np.dot(s_params[int(self.sample_y[d])], self.latent_z[d])
            r += tmp2 - 1 / self.slack_params[d] * tmp3 - np.log(self.slack_params[d]) + 1
            
        if r == float('inf') or r == float('-inf') or r != r:
            print s_params
            print self.latent_z[d]
            print np.dot(s_params, self.latent_z[d])
            sys.exit(1)
        #print "r", r
        return -r
    
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
        
        init_theta = np.random.rand(self.supervived_params.shape[0] * self.supervived_params.shape[1])
        min_max = [(-15, 15)] * init_theta.shape[0]
        #best_params = optimize.fmin_cg(self.J, init_theta, fprime=self.gradient, gtol=1e-5)
        #best_params = optimize.minimize(self.J, init_theta, tol=1e-4, method='L-BFGS-B', options={"maxiter":20}, bounds=min_max, jac=self.gradient)
        #print "best_params.fun", best_params.fun, best_params.nit
        best_params = optimize.minimize(self.J, init_theta, tol=1e-3, method='SLSQP', options={"maxiter":30}, bounds=min_max, jac=self.gradient)
        #best_params = optimize.minimize(self.J, init_theta, tol=1e-3, method='CG', options={"maxiter":150}, jac=self.gradient)
        #print "best_params.fun", best_params.fun, best_params.nit
        #best_params = optimize.differential_evolution(self.J, min_max, maxiter=50)
        #print "best_params.fun", best_params.fun, best_params.nit
        self.supervived_params = best_params.x.reshape(self.supervived_params.shape[0], self.supervived_params.shape[1])
        self.score(self.latent_z, self.sample_y)
        for d in range(self.slack_params.shape[0]):
            #self.slack_params[d] = np.sum(np.exp(np.dot(self.supervived_params, self.latent_z[d])))
            components_sum = np.sum(np.exp(np.dot(self.supervived_params, self.latent_z[d].reshape(-1, 1) )))
            self.slack_params[d] = np.sum(self.latent_z[d] * components_sum)
    
    
    def score(self, sample_X, sample_y):
        #print self.supervived_params
        result = []
        for i in range(self.n_class):
            a = np.dot(self.supervived_params[i, :].reshape(1, -1), sample_X.T).flatten()
            b = np.sum(np.dot(self.supervived_params, sample_X.T), 0)
            result.append(a / b)
        diff = sample_y - np.argmax(np.array(result), 0)
        print float(diff[diff==0].shape[0]) / diff.shape[0]
    
    
        
