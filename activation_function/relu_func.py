import numpy as np

def relu(x):
    return np.maximum(x)

#np.maximum으로 0 초과이면 x의 값이 그대로 나오며
#0 이하이면 0이 나오게 된다. 