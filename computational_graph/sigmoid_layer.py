#sigmoid layer은 출력만으로 미분값을 하류로 전달해 줄수 있기에, 출력값을 저장하도록 한다!

import numpy as np

class Sigmoid:
    def __init__(self):
        selt.out = None
        #순전파의 출력을 인스턴스 변수 out에 보관했다가 backpropagation 때 다시 사용한다!   
    def forward(self, x):
        out = 1/ (1 + np.exp(-x))
        return out

    def backward(self, dout):
        dx = dout * self.out * (1-self.out)

        return dx
