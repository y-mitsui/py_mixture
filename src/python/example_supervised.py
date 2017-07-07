'''
Created on 2017/06/15

'''
import numpy as np
from sklearn.datasets import fetch_mldata
from supervised_bernoulli_mixture import SupervisedBernoulliMixture
from PIL import Image
import sys

n_components = 4

mnist = fetch_mldata('MNIST original', data_home=".")
zero_index = mnist["data"] <= 128
mnist["data"][zero_index] = 0
mnist["data"][np.logical_not(zero_index)] = 1
target_index = np.logical_or(np.logical_or(np.logical_or(mnist["target"] == 0, mnist["target"] == 1), mnist["target"] == 2), mnist["target"] == 3)
shuffle_idx = range(mnist["data"][target_index].shape[0])
np.random.shuffle(shuffle_idx)
data = mnist["data"][target_index][shuffle_idx[:5000]]
target = mnist["target"][target_index][shuffle_idx[:5000]]
#data = mnist["data"]
#target = mnist["target"]
print "data.shape", data.shape
data = data
target = target
n_dim = data.shape[1]

poisson_mixture = SupervisedBernoulliMixture(n_components, 200)
poisson_mixture.fit(data, target)
print poisson_mixture.poi_params
print "finish"
for i, mean in enumerate(poisson_mixture.poi_params):
    mean = np.array(mean)
    pil_img = Image.fromarray(np.uint8(mean.reshape(28, 28) * 255))
    pil_img.save('s_result' + str(i) + '.jpg')
