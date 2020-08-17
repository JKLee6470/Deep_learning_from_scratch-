import numpy as np

#MSE(mean squared error) 구현

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

t = [0,0,1,0,0,0,0,0,0,0]

y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]

print(mean_squared_error(np.array(y), np.array(t)))

#Cross entropy error 구현
def cross_entropy_error_old(y,t):
    delta = 1e-7
    
    return -np.sum(t * np.log(y + delta))
    # delta를 더해주는 것은 무한대로 발산하게 하지 않는 방법.

#mini batch를 구현하는 cross_entropy_error
def cross_entropy_error_2d(y,t):
    if y.ndim == 1:
        #1차원 데이터를 2차원으로 바꾸는 것은, mini batch 데이터가 2차원이여서 
        #2차원으로 바꾸어 밑의 식에 공통적으로 적용할려고
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
t = [0,0,1,0,0,0,0,0,0,0]

y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]

print(cross_entropy_error(np.array(y), np.array(t)))

