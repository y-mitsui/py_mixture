'''
Created on 2017/07/20

@author: mitsuiyosuke
'''
from bernoulli_normal_mixture_wrap import BernoulliNormalMixtureWrap
import numpy as np

import matplotlib.pyplot as plt
n_sample = 200
n_dim = 1
n_poi_dim = 2
n_components = 2
#np.random.seed(12154)
weights = np.random.dirichlet([1] * n_components)
#weights = [0.1, 0.9]
#ber_params = [[0.1], [0.9]]
#norm_means = [[20], [-20]]
ber_params = np.random.uniform(1e-3, 1 - 1e-3, size=(n_components, n_dim))
poi_params = np.random.uniform(1e-3, 3, size=(n_components, n_dim))
norm_means = np.random.uniform(-4, 4, size=(n_components, n_dim))
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

poisson_mixture = BernoulliNormalMixtureWrap(n_components, 50)
latent_z = poisson_mixture.fit_transform(sample_bernoulli, sample_normal)

for ber, norm, color in zip(sample_bernoulli, sample_normal, latent_z):
    plt.scatter(ber, norm, c=['r', 'b'][np.argmax(color)],alpha=0.1)
plt.show()
