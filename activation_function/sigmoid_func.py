import numpy as np
import matplotlib.pyplot as plt 

#sigmoid function (or other types of activation functions) are the difference between perceptron
#and neural network. 
def sigmoid(x):
    return 1/(1 + np.exp(-x))

x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.show()