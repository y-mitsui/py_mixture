'''
Created on 2017/06/15

'''
import numpy as np
from sklearn.datasets import fetch_mldata
from bernoulli_mixture_wrap import BernoulliMixtureWrap
from PIL import Image
import time
import sys

n_components = 10

mnist = fetch_mldata('MNIST original', data_home="/home/yosuke/workspace/py_mixture/src/python/")
print "loaded"
zero_index = mnist["data"] <= 128
mnist["data"][zero_index] = 0
mnist["data"][np.logical_not(zero_index)] = 1
#target_index = np.logical_or(np.logical_or(mnist["target"] == 2, mnist["target"] == 3), mnist["target"] == 4)
#data = mnist["data"][target_index]
#np.random.shuffle(data)
#data = data[:200]
data = mnist["data"]
n_dim = data.shape[1]
print "data.shape", data.shape

poisson_mixture = BernoulliMixtureWrap(n_components, 50)
t1 = time.time()
poisson_mixture.fit_transform(data)
print "time:", time.time() - t1
print np.array(poisson_mixture.getMeans())
print "finish"
for i, mean in enumerate(poisson_mixture.getMeans()):
    mean = np.array(mean)
    pil_img = Image.fromarray(np.uint8(mean.reshape(28, 28) * 255))
    pil_img.save('result' + str(i) + '.jpg')
