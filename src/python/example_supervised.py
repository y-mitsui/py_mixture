'''
Created on 2017/06/15

'''
import numpy as np
from sklearn.datasets import fetch_mldata
from supervised_bernoulli_mixture import SupervisedBernoulliMixture
from PIL import Image
import sys

n_components = 5
np.random.seed(1234)
mnist = fetch_mldata('MNIST original', data_home=".")
zero_index = mnist["data"] <= 128
mnist["data"][zero_index] = 0
mnist["data"][np.logical_not(zero_index)] = 1

target_index = None
for target in range(0, n_components):
    if target_index == None:
        target_index = mnist["target"] == target
    else:
        target_index = np.logical_or(target_index, mnist["target"] == target)
shuffle_idx = range(mnist["data"][target_index].shape[0])
np.random.shuffle(shuffle_idx)
data = mnist["data"][target_index]#[shuffle_idx[:500]]
target = mnist["target"][target_index]#[shuffle_idx[:500]]
#data = mnist["data"]
#target = mnist["target"]
print "data.shape", data.shape
data = data
target = target
n_dim = data.shape[1]

poisson_mixture = SupervisedBernoulliMixture(n_components, 20)
poisson_mixture.fit(data, target)
print poisson_mixture.poi_params
print "finish"
for i, mean in enumerate(poisson_mixture.poi_params):
    mean = np.array(mean)
    pil_img = Image.fromarray(np.uint8(mean.reshape(28, 28) * 255))
    pil_img.save('s_result' + str(i) + '.jpg')
