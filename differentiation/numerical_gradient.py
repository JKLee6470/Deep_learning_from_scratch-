import numpy as np

def numerical_gradient_1D(f,x):
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


def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f, x): #N차원 배열의 gradient를 구할 수 있도록 바뀐것. 
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad

print(numerical_gradient(function_2, np.array([3.0,4.0])))
print(numerical_gradient(function_2, np.array([0.0,2.0])))
print(numerical_gradient(function_2, np.array([3.0,0.0])))

