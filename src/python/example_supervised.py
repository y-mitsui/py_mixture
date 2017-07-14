'''
Created on 2017/06/15
'''
import numpy as np
from sklearn.datasets import fetch_mldata
from supervised_bernoulli_mixture import SupervisedBernoulliMixture
from PIL import Image
import sys
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation

n_components = 2
np.random.seed(1234)
mnist = fetch_mldata('MNIST original', data_home=".")
zero_index = mnist["data"] <= 128
mnist["data"][zero_index] = 0
mnist["data"][np.logical_not(zero_index)] = 1

target_index = None
for target in range(5, 5 + n_components):
    if target_index == None:
        target_index = mnist["target"] == target
    else:
        target_index = np.logical_or(target_index, mnist["target"] == target)

shuffle_idx = range(mnist["data"][target_index].shape[0])
np.random.shuffle(shuffle_idx)
data = mnist["data"][target_index][shuffle_idx]
target = mnist["target"][target_index][shuffle_idx] - 5
#data = mnist["data"]
#target = mnist["target"]
print "data.shape", data.shape
n_dim = data.shape[1]

poisson_mixture = SupervisedBernoulliMixture(n_components, 50)
sample_X_low = poisson_mixture.fit_transform(data, target)
train_X = sample_X_low[:10000]
train_y = target[:10000]
test_X = sample_X_low[10000:]
test_y = target[10000:]

print poisson_mixture.poi_params
print "finish"
for i, mean in enumerate(poisson_mixture.poi_params):
    mean = np.array(mean)
    pil_img = Image.fromarray(np.uint8(mean.reshape(28, 28) * 255))
    pil_img.save('s_result' + str(i) + '.jpg')

def create_model():
    model = Sequential()
    model.add(Dense(n_components, input_dim=n_components, init='glorot_uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer="adadelta", 
                  metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=256, verbose=0)
model.fit(train_X, train_y)
print model.score(test_X, test_y)
