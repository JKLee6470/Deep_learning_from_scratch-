import sys, os
sys.path.append('D:\\DEEP_LEARNING_FROM_SCRATCH')
from mnist_data.mnist import *
import numpy as np

(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) #(60000,784)
print(t_train.shape) #(60000,10)

#mini batch를 위하여 무작위 추출. 
trainsize = x_train.shape[0]
batch_size = 10

#0에서 train_size의 숫자에서, batch_size의 숫자만큼을 선택한다. 
batch_mask = np.random.choice(trainsize, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(np.random.choice(60000,10))
print(x_batch.shape)
print(t_batch.shape)