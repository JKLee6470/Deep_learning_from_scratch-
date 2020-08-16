import numpy as np

#softmax 함수는 지수함수를 사용하기에, 매우 큰 값이 생성되어 
#overflowrk 발생하기 쉽다. 따라서 계산식의 위와 아래에, C를 곱해도 동일한 결과가 나오므로
#숫자를 조정해준다. 

def softmax(a):
    c = np.max(a) # to prevent overflow
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([0.1,0.3,0.7])
print(softmax(a))
print(softmax(a).sum()) #softmax의 원소들의 합은 1이 된다! 확률로 해석이 가능하다!
 