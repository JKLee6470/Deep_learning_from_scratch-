import numpy as np


def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    #x와 형상이 같은 0으로 이루어진 배열을 형성한다. 

    for index in range(x.size):
        tmp_val = x[index]

        x[index] = tmp_val + h
        fxh1 = f(x)

        x[index] =  tmp_val - h
        fxh2 = f(x)

        grad[index] = (fxh1- fxh2) / (2*h)
        x[index] = tmp_val 

    return grad

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    
    return x

def function_2(x):
    return np.sum(x**2)

init_x = np.array([-3.0,4.0])

# a = gradient_descent(function_2, init_x=init_x, lr = 0.1, step_num=100)
# b = gradient_descent(function_2, init_x=init_x, lr = 10, step_num=100) # learning ratge가 클 때 
c = gradient_descent(function_2, init_x=init_x, lr = 1e-10, step_num=100) # learning rate가 작을 때
# print(a)
# print(b)
print(c)