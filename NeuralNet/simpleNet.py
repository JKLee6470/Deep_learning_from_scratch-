import sys, os
sys.path.append('D:\\DEEP_LEARNING_FROM_SCRATCH')
from mnist_data.mnist import *
from loss_function.loss_functions import cross_entropy_error
from activation_function.softmax_func import softmax
from differentiation.numerical_gradient import numerical_gradient
import numpy as np


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self,x): 
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
        
        return loss

#SimpleNet을 활용한 몇가지의 실험
net = SimpleNet()

x = np.array([0.6,0.9])
p = net.predict(x)
t = np.array([0,0,1])

f = lambda W: net.loss(x,t)
dW = numerical_gradient(f,net.W)

print(dW)