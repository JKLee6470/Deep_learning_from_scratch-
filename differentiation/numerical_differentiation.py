import numpy as np
import matplotlib.pyplot as plt


#bad implementation of numerical differentiation
def bad_numerical_diff(f,x):
    h = 10e-50 # 10e-50은 너무나도 작은 숫자기에 rounding_error(반올림 오차)를 일으킨다
    return (f(x+h)-f(x)) / h #중심 차분하지 않아, 실제 기울기랑은 차이가 있다. 

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x
    
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x,y)
#plt.show()

print(numerical_diff(function_1, 5)) #0.19999999
print(numerical_diff(function_1, 10)) #0.29999999

def function_2(x):
    return x[0]**2 + x[1]**2
    #or return np.sum(x**2)

#x0 = 3, x1 = 4일 때 xo에 대한 편미분

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1,3.0))

#xo = 3, x1 = 4일 떄, x1에 대한 편미분을 구하여라 .
def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

print(numerical_diff(function_tmp2, 4.0))