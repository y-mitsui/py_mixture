'''
Created on 2017/06/15

'''
import numpy as np
from sklearn.datasets import fetch_mldata
from bernoulli_mixture import BernoulliMixture
from PIL import Image
import sys
from sklearn.ensemble import RandomForestClassifier

n_components = 2

mnist = fetch_mldata('MNIST original', data_home=".")
zero_index = mnist["data"] <= 128
mnist["data"][zero_index] = 0
mnist["data"][np.logical_not(zero_index)] = 1

target_index = None
for target in range(0, 10):
    if target_index == None:
        target_index = mnist["target"] == target
    else:
        target_index = np.logical_or(target_index, mnist["target"] == target)

#data = mnist["data"][target_index]
shuffle_idx = range(mnist["data"][target_index].shape[0])
np.random.shuffle(shuffle_idx)
data = mnist["data"][target_index][shuffle_idx[:100]]
target = mnist["target"][target_index][shuffle_idx[:100]]
print "data.shape", data.shape
#data = data[:1000]

poisson_mixture = BernoulliMixture(n_components, 20)
sample_X = poisson_mixture.fit(data)
print poisson_mixture.poi_params
print "finish"
for i, mean in enumerate(poisson_mixture.poi_params):
    mean = np.array(mean)
    pil_img = Image.fromarray(np.uint8(mean.reshape(28, 28) * 255))
    pil_img.save('n_result' + str(i) + '.jpg')

random_forest = RandomForestClassifier()
random_forest.fit(sample_X, target)
print random_forest.score(sample_X, target)
