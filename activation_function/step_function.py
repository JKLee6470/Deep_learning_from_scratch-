import numpy as np
import matplotlib.pyplot as plt

#activation function for perceptron
def step_function(x):
    y = x>0 #numpy array를 인수로 받아들인다. 

    return y.astype(np.int)

x = np.arange(-5.0,5.0,0.1)
y = step_function(x)
plt.plot(x,y)
plt.show()