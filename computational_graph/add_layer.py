
import sys, os
sys.path.append('D:\\DEEP_LEARNING_FROM_SCRATCH')
from computational_graph.mul_layer import MulLayer

class AddLayer:
    def __init__(self):
        pass #pass를 하는 이유: backpropagation에서 입력신호를 사용하지 않기 때문.

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

#계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

#forward propagation
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
apple_orange_price=  add_apple_orange_layer.forward(apple_price, orange_price)
total_price = mul_tax_layer.forward(apple_orange_price, tax)
print('total_price: ',total_price)

#back propagation
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
print('dapple: ', dapple)
print('dorange: ',dorange)
print('dapple_num: ',dapple_num)