'''
Created on 2017/06/13

'''
from bernoulli_mixture_wrap import BernoulliMixtureWrap
import numpy as np

if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    n_sample = 1000
    n_dim = 1000
    n_components = 200
    
    weights = np.random.dirichlet([1] * 2)
    ber_params = np.random.uniform(1e-3, 1 - 1e-3, size=(n_components, n_dim))
    latent_z = np.random.multinomial(1, weights, size=n_sample)
    sample_X = []
    for z in latent_z:
        X = np.random.binomial(1, ber_params[np.where(z==1)[0][0]])
        sample_X.append(X)
    #plt.hist(sample_X)
    #plt.show()
    sample_X = np.array(sample_X)
    poisson_mixture = BernoulliMixtureWrap(n_components, 5)
    mean_latent = poisson_mixture.fit_transform(sample_X)
    print np.array(mean_latent)
