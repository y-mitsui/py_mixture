'''
Created on 2017/06/15

'''
import numpy as np
from sklearn.datasets import fetch_mldata
from bernoulli_mixture_wrap import BernoulliMixtureWrap
from PIL import Image
import sys

n_components = 3

mnist = fetch_mldata('MNIST original', data_home="python/")
zero_index = mnist["data"] <= 128
mnist["data"][zero_index] = 0
mnist["data"][np.logical_not(zero_index)] = 1
target_index = np.logical_or(np.logical_or(mnist["target"] == 2, mnist["target"] == 3), mnist["target"] == 4)
data = mnist["data"][target_index]
#data = mnist["data"]
np.random.shuffle(data)
print "data.shape", data.shape
data = data[:1000]
n_dim = data.shape[1]

sample_X = []
for X in data:
    row = []
    for i in range(n_dim):
        if X[i] == 1:
            row.append(i)
    sample_X.append(row) 

poisson_mixture = BernoulliMixtureWrap(n_components, 100)
mean_latent = poisson_mixture.fit_transform(sample_X, n_dim)
print "ber_params", np.array(poisson_mixture.getMeans())

for i, mean in enumerate(poisson_mixture.getMeans()):
    mean = np.array(mean)
    pil_img = Image.fromarray(np.uint8(mean.reshape(28, 28) * 255))
    pil_img.save('result' + str(i) + '.jpg')
