'''
Created on 2017/06/13

'''
from bernoulli_mixture_wrap import BernoulliMixtureWrap
import numpy as np

if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    n_sample = 1000000
    n_dim = 2
    n_components = 2
    
    np.random.seed(123)
    weights = np.random.dirichlet([1] * 2)
    ber_params = np.random.uniform(1e-3, 1 - 1e-3, size=(n_components, n_dim))
    latent_z = np.random.multinomial(1, weights, size=n_sample)
    print "weights", weights
    print "ber_params", ber_params
    sample_X = []
    for z in latent_z:
        X = np.random.binomial(1, ber_params[np.where(z==1)[0][0]])
        row = []
        for i in range(n_dim):
            if X[i] == 1:
                row.append(i)
        sample_X.append(row)
    #plt.hist(sample_X)
    #plt.show()
    poisson_mixture = BernoulliMixtureWrap(n_components, 200)
    mean_latent = poisson_mixture.fit_transform(sample_X, n_dim)
    print "weights", np.array(poisson_mixture.getWeights())
    print "ber_params", np.array(poisson_mixture.getMeans())
