import numpy as np

def AND_numpy(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7

    tmp = np.sum(x*w) + b #x*w = elementwise multiplication. 
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7

    tmp = np.sum(x*w) + b

    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):

    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.3

    tmp = np.sum(x*w) + b #elementwise multiplication
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND_numpy(s1,s2)

    return y

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))