'''
Created on 2017/07/20

@author: mitsuiyosuke
'''
from many_mixture_wrap import ManyMixtureWrap
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd

n_sample = 1000
n_categorical_params = [2, 2]
n_components = 2
weights = [0.5, 0.5]
categorical_params = [[[0.1, 0.45, 0.45], [0.2, 0.8]], [[0.45, 0.45, 0.1], [0.8, 0.2]]]

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
poisson_mixture = ManyMixtureWrap(n_components, 20)
latent_z = poisson_mixture.fit_transform(sample_categorical=sample_categorical)

print categorical_params
print poisson_mixture.getCategoricalParams()
print zip(sample_categorical[:100], np.argmax(latent_z, 1)[:100])
