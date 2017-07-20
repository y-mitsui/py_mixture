'''
Created on 2017/06/12

'''
import numpy as np
from scipy.misc import logsumexp
import sys
from scipy.stats import norm
from sklearn.cluster import KMeans

class BernoulliNormalMixture:
    
    def __init__(self, n_components=2, n_iter=100, init_kmeans=False):
        self.n_components = n_components
        self.n_iter = n_iter
        self.init_kmeans = init_kmeans
    
    def fit(self, sample_bernoulli, sample_normal):
        self.sample_bernoulli = sample_bernoulli
        self.sample_normal = sample_normal
        n_samples = sample_bernoulli.shape[0]
        n_dimentions_bernoulli = sample_bernoulli.shape[1]
        n_dimentions_normal = sample_normal.shape[1]
        self.bernoulli_params = np.random.rand(self.n_components, n_dimentions_bernoulli)
        
        if self.init_kmeans:
            k_means = KMeans(self.n_components)
            k_means.fit(sample_normal)
            self.normal_means = k_means.cluster_centers_
        else:
            self.normal_means = np.random.randn(self.n_components, n_dimentions_normal) * 10
            
        self.normal_sigmas = np.random.uniform(1, 2, size=(n_components, n_dimentions_normal))
        self.latent_z = np.random.rand(n_samples, self.n_components)
        self.weights = np.random.rand(self.n_components)
        self.weights /= self.weights.sum()
        
        for i in range(self.n_iter):
            self.eStep()
            self.mStep()
            print "%d / %d"%(i, self.n_iter)
        return self.latent_z
    
    def log_bernoulli(self, X, parameter):
        np.clip(parameter, 1e-10, 1 - 1e-10, out=parameter)
        pdf = X * np.log(parameter) + (1 - X) * np.log((1 - parameter))
        return pdf.sum()
    
    def log_normal(self, X, means, sigmas):
        return norm.logpdf(X, means, np.sqrt(sigmas)).sum()
    
    def eStep(self):
        self.latent_z = []
        log_weights = np.log(self.weights)
        loglikelyfood = 0.
        print "normal_means", self.normal_means
        print "self.normal_sigmas", self.normal_sigmas
        for X_bernoulli, X_normal in zip(self.sample_bernoulli, self.sample_normal):
            weights_probs = [] 
            for k in range(self.n_components):
                a = log_weights[k]
                b1 = self.log_bernoulli(X_bernoulli, self.bernoulli_params[k])
                b2 = self.log_normal(X_normal, self.normal_means[k], self.normal_sigmas[k])
                weights_probs.append(a + b1 + b2)
            tot_log_likelyhood = logsumexp(weights_probs)
            loglikelyfood += tot_log_likelyhood
            z = []
            for wp in weights_probs:  
                z.append(wp - tot_log_likelyhood)
            self.latent_z.append(z)
        self.latent_z = np.exp(np.array(self.latent_z))
        print "loglikelyfood", loglikelyfood
        
    def mStep(self):
        new_ber_params = []
        new_norm_means = []
        new_norm_sigma = []
        for k in range(self.bernoulli_params.shape[0]):
            tot_latent_z = self.latent_z[:, k].sum()
            row_ber_params = []
            for d in range(self.bernoulli_params.shape[1]):
                row_ber_params.append((self.latent_z[:, k] * self.sample_bernoulli[:, d]).sum() / tot_latent_z)
            new_ber_params.append(row_ber_params)
            
            row_norm_means = []
            for d in range(self.bernoulli_params.shape[1]):
                row_norm_means.append((self.latent_z[:, k] * self.sample_normal[:, d]).sum() / tot_latent_z)
            new_norm_means.append(row_norm_means)
            
            row_norm_sigma = []
            for d in range(self.bernoulli_params.shape[1]):
                row_norm_sigma.append((self.latent_z[:, k] * (self.sample_normal[:, d] - row_norm_means[d]) ** 2).sum() / tot_latent_z)
            #print row_norm_sigma
            new_norm_sigma.append(np.maximum(row_norm_sigma, 1e-5))
            #sigma = (self.latent_z[:, k] * (self.sample_normal[:, 0] - np.array(row_norm_means)) ** 2).sum(axis = 0) / tot_latent_z
            #print sigma
            #sys.exit(1)
            
            self.weights[k] = tot_latent_z / self.sample_bernoulli.shape[0]
        self.bernoulli_params = np.array(new_ber_params)
        self.normal_means = np.array(new_norm_means)
        self.normal_sigmas = np.array(new_norm_sigma)
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n_sample = 200
    n_dim = 1
    n_components = 2
    #np.random.seed(12154)
    weights = np.random.dirichlet([1] * n_components)
    #weights = [0.1, 0.9]
    #ber_params = [[0.1], [0.9]]
    #norm_means = [[20], [-20]]
    ber_params = np.random.uniform(1e-3, 1 - 1e-3, size=(n_components, n_dim))
    norm_means = np.random.uniform(-20, 20, size=(n_components, n_dim))
    norm_sigmas = np.random.uniform(1, 2, size=(n_components, n_dim))
    latent_z = np.random.multinomial(1, weights, size=n_sample)
    sample_bernoulli = []
    sample_normal = []
    for z in latent_z:
        X_bernoulli = np.random.binomial(1, ber_params[np.where(z==1)[0][0]])
        X_normal = np.random.normal(norm_means[np.where(z==1)[0][0]], norm_sigmas[np.where(z==1)[0][0]])
        sample_bernoulli.append(X_bernoulli)
        sample_normal.append(X_normal)
    #plt.hist(sample_X)
    #plt.show()
    sample_bernoulli, sample_normal = np.array(sample_bernoulli), np.array(sample_normal)
    
    poisson_mixture = BernoulliNormalMixture(n_components, 50)
    latent_z = poisson_mixture.fit(sample_bernoulli, sample_normal)
    print latent_z
    
    for ber, norm, color in zip(sample_bernoulli, sample_normal, latent_z):
        plt.scatter(ber, norm, c=['r', 'b'][np.argmax(color)],alpha=0.1)
    plt.show()
