import math

class Node:

    def __init__(self):
        self.val = None
        self.links = None

    # Value
    def value(self):
        return self.val

    def reset(self):
        for l in self.links:
            l.reset()

    # Evalulate
    def forward(self, dict):
        raise NotImplementedError

    # Derivate function
    def backward(self):
        raise NotImplementedError

    def gradient(self, acc=1):
        for l in range(len(self.links)):
            self.links[l].gradient(acc * self.backward()[l])

    # Operations to build graph
    def __neg__(a):
        return Neg((a,))

    def __add__(a, b):
        return Add((a, b))
    
    def __sub__(a, b):
        return Sub((a, b))

    def __mul__(a, b):
        return Mul((a, b))

    def __div__(a, b):
        return Div((a, b))

class Const(Node):

    def __init__(self, const):
        self.val = const

    def reset(self):
        pass

    def forward(self, dict):
        return self.val

    def backward(self, dict):
        return (1,)

    def gradient(self, acc=1):
        pass

class Placeholder(Node):

    def __init__(self, symbol):
        self.val = 0
        self.grad = 0
        self.symbol = symbol

    def reset(self):
        self.grad = 0

    def forward(self, dict):
        self.val = dict[self.symbol]
        return self.val

    def backward(self):
        return (1,)

    def gradient(self, acc=1):
        self.grad += acc

class Tanh(Node):

    def __init__(self, links):
        self.val = 0
        self.links = links

    def forward(self, dict):
        self.val = math.tanh(self.links[0].forward(dict))
        return self.val

    def backward(self):
        return ((1 - math.tanh(self.links[0].value()) ** 2),)

class Relu(Node):

    def __init__(self, links):
        self.val = 0
        self.links = links

    def forward(self, dict):
        x = self.links[0].forward(dict)
        self.val = x if x > 0 else 0
        return self.val

    def backward(self):
        return (1 if self.val > 0 else 0, )

class Sigmoid(Node):

    def __init__(self, links):
        self.val = 0
        self.links = links

    def forward(self, dict):
        self.val = 1 / (1 + math.exp(-self.links[0].forward(dict)))
        return self.val

    def backward(self):
        return (((1 / (1 + math.exp(self.links[0].value()))) * (1 - (1 / (1 + math.exp(self.links[0].value()))))),)

class Neg(Node):

    def __init__(self, links):
        self.links = links

    def forward(self, dict):
        self.val = -self.links[0].forward(dict)
        return self.val

    def backward(self):
        return (-1,)

class Add(Node):

    def __init__(self, links):
        self.val = 0
        self.links = links

    def forward(self, dict):
        self.val = self.links[0].forward(dict) + self.links[1].forward(dict)
        return self.val

    def backward(self):
        return (1, 1)
    
class Sub(Node):

    def __init__(self, links):
        self.val = 0
        self.links = links

    def forward(self, dict):
        self.val = self.links[0].forward(dict) - self.links[1].forward(dict)
        return self.val

    def backward(self):
        return (1, -1)

class Mul(Node):

    def __init__(self, links):
        self.val = 0
        self.links = links

    def forward(self, dict):
        self.val = self.links[0].forward(dict) * self.links[1].forward(dict)
        return self.val

    def backward(self):
        return (self.links[1].value(), self.links[0].value())

class Div(Node):

    def __init__(self, links):
        self.val = 0
        self.links = links

    def forward(self, dict):
        self.val = self.links[0].forward(dict) / self.links[1].forward(dict)
        return self.val

    def backward(self):
        return (1 / self.links[1].value(), -(self.links[0].value() / (self.links[1].value() * self.links[1].value())))
