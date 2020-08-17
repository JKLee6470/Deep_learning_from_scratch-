import sys, os
sys.path.append('D:\\DEEP_LEARNING_FROM_SCRATCH')
from mnist_data.mnist import *
from loss_function.loss_functions import cross_entropy_error
from activation_function.softmax_func import softmax
from differentiation.numerical_gradient import numerical_gradient
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        #weight 초기화
        self.params = dict()
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(a1,W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)

        grads = dict()

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print('W1 shape:',net.params['W1'].shape)
print('b1 shape:',net.params['b1'].shape)
print('W2 shape:',net.params['W2'].shape)
print('b2 shape:',net.params['b2'].shape)

x = np.random.rand(100,784)
t = np.random.rand(100,10)
y = net.predict(x)

grads = net.numerical_gradient(x, t)

print("grad W1's shape", grads['W1'].shape)
print("grad b1's shape", grads['b1'].shape)
print("grad W2's shape", grads['W2'].shape)
print("grad b2's shape", grads['b2'].shape)