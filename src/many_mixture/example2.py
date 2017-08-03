'''
Created on 2017/07/20

@author: mitsuiyosuke
'''
from many_mixture_wrap import ManyMixtureWrap
import numpy as np
import sys
import matplotlib.pyplot as plt

n_sample = 20
n_poi_dim = 2
n_components = 2
np.random.seed(1541)
weights = np.random.dirichlet([1] * n_components)
#weights = [0.1, 0.9]
poi_params = np.random.uniform(1e-3, 2, size=(n_components, n_poi_dim))
latent_z = np.random.multinomial(1, weights, size=n_sample)
poisson_indexes = []
poisson_counts = []
sample_poisson = []
for z in latent_z:
    X_poisson = np.random.poisson(poi_params[np.where(z==1)[0][0]], size=n_poi_dim)
    indexes_row = []
    counts_row = []
    for j in range(n_poi_dim):
        if X_poisson[j] > 0:
            indexes_row.append(j)
            counts_row.append(X_poisson[j])
    poisson_indexes.append(indexes_row)
    poisson_counts.append(counts_row)
    sample_poisson.append(X_poisson)
    
poisson_indexes, poisson_counts = np.array(poisson_indexes), np.array(poisson_counts)
sample_poisson = np.array(sample_poisson)
print sample_poisson
print poisson_indexes
print poisson_counts

poisson_mixture = ManyMixtureWrap(n_components, 50)
latent_z = poisson_mixture.fit_transform(poisson_indexes, poisson_counts, sample_poisson)

for poisson, color in zip(sample_poisson, latent_z):
    plt.scatter(poisson[0], poisson[1], c=['r', 'b'][np.argmax(color)],alpha=0.1)
plt.show()
