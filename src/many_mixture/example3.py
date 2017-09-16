'''
Created on 2017/07/20

@author: mitsuiyosuke
'''
from many_mixture_wrap import ManyMixtureWrap
import numpy as np
import sys
import matplotlib.pyplot as plt

n_sample = 200
n_categorical_dim = 2
n_categorical_params = [3, 3]
n_components = 2
#np.random.seed(15421)
weights = np.random.dirichlet([1] * n_components)
#weights = [0.1, 0.9]

categorical_params = []
for k in range(n_components):
    categorical_param = []
    for n_categorical_param in n_categorical_params:
        categorical_param.append(np.random.dirichlet([1] * n_categorical_param))
    categorical_params.append(categorical_param)
    
latent_z = np.random.multinomial(1, weights, size=n_sample)

sample_categorical = []
for z in latent_z:
    param = categorical_params[np.where(z==1)[0][0]]
    row_sample = []
    for p in param:
        row_sample.append(np.random.multinomial(1, p, size=len(p))[0])
    sample_categorical.append(row_sample)

sample_categorical = np.array(sample_categorical)
print "sample_categorical", sample_categorical
print sample_categorical[0][0][0]
poisson_mixture = ManyMixtureWrap(n_components, 200)
latent_z = poisson_mixture.fit_transform(sample_categorical=sample_categorical)

for poisson, color in zip(sample_poisson, latent_z):
    plt.scatter(poisson[0], poisson[1], c=['r', 'b'][np.argmax(color)],alpha=0.5)
plt.show()
