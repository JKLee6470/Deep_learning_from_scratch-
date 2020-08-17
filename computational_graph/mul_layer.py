#곱셈 계층과 덧셈 계층의 구현

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
        #순전파시의 x 와 y의 값을 전역변수로 저장하는 것은, backpropagation시 x 와 y를 사용하기 때문!

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

apple = 100
apple_num = 2
tax = 1.1

#계층
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

#forward propagation
apple_price = mul_apple_layer.forward(apple, apple_num)
total_price = mul_tax_layer.forward(apple_price, tax)

print(total_price)

#back propagation
dprice = 1 #상류에서 오는 신호값.
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)