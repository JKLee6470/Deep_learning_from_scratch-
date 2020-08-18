import numpy as np
import sys, os
sys.path.append('D:\\DEEP_LEARNING_FROM_SCRATCH')
from book_sample_codes.util import im2col

x = np.random.rand(10,1,28,28)
print(x[0].shape) # 첫번째 데이터에 접근하는 것. 
print(x[1].shape) # 두번째 데이터에 접근하는 것. 
print(x[0][0].shape) # 첫번째 데이터의 첫번째 공간 데이터에 접근 하는 것. 

#im2col
x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1,5,5,stride = 1, pad = 0)
print(col1.shape)

x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2, 5, 5, stride = 1, pad = 0)
print(col2.shape) #이전과는 달리 batch 내의 데이터 사이즈를 10배로 하여,
                  #행의 숫자가 10배로 증가한 거을 알 수 있다. 