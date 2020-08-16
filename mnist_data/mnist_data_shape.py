import sys, os
sys.path.append('D:\\DEEP_LEARNING_FROM_SCRATCH')
from mnist_data.mnist import load_mnist

(x_train, t_train),(x_test,t_test) = load_mnist(flatten= True, normalize=False)
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(x_test.shape)
