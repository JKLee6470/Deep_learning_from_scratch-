#퍼셉트론은 다수의 입력을 받아서, 하나의 신호를 출력.
#이때, 신호는 0과 1의 값을 가질수 있다. 
#maginitude of the bias controls how easy the neuron will be activated.

import numpy as np

def AND(x1, x2):
    #w1,w2 = weight, theta = threshold value
    w1, w2, theta = 0.5,0.5,0.7
    tmp = x1*w1 + x1*w2
    if tmp <= theta:
        return 0

    elif tmp > theta:
        return 1


#threshold value였던 theta를 bias의 형태로 전환후 배열 계산 도입. 
def AND_numpy(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7

    tmp = np.sum(x*w) + b #x*w = elementwise multiplication. 
    if tmp <= 0:
        return 0
    else:
        return 1

print(AND(0,0)) #0
print(AND(0,1)) #0
print(AND(1,0)) #0
print(AND(1,1)) #1
print(AND_numpy(0,0)) #0
print(AND_numpy(0,1)) #0
print(AND_numpy(1,0)) #0
print(AND_numpy(1,1)) #1