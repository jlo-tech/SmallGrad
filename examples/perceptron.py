import math
import random
from graph import *
import matplotlib.pyplot as plt

"""
    A simple example teaching a small ANN to learn the XOR function
"""

# Set seed for better testing
#random.seed(1)

# Network topology: 2 Input, 3 Hidden, 1 Output
def net(x0, x1, w0, w1, w2, w3, w4, w5, w6, w7, w8, b0, b1, b2):

    n1 = Sigmoid((((x0 * w0) + (x1 * w1)) + b2,))
    n2 = Sigmoid((((x0 * w2) + (x1 * w3)) + b1,))
    n3 = Sigmoid((((x0 * w4) + (x1 * w5)) + b,))
    
    n4 = Sigmoid((((n1 * w6) + (n2 * w6) + (n3 * w8)),))
    
    return n4

feed = {
    "X0": 0,
    "X1": 0,
    "Y":  0,
    "W0": random.randrange(-1, 1) * random.random(),
    "W1": random.randrange(-1, 1) * random.random(),
    "W2": random.randrange(-1, 1) * random.random(),
    "W3": random.randrange(-1, 1) * random.random(),
    "W4": random.randrange(-1, 1) * random.random(),
    "W5": random.randrange(-1, 1) * random.random(),
    "W6": random.randrange(-1, 1) * random.random(),
    "W7": random.randrange(-1, 1) * random.random(),
    "W8": random.randrange(-1, 1) * random.random(),
    "B" : random.randrange(-1, 1) * random.random(),
    "B1": random.randrange(-1, 1) * random.random(),
    "B2": random.randrange(-1, 1) * random.random(),
}

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

Y = [
    0,
    1,
    1,
    0,
]

x0 = Placeholder("X0")
x1 = Placeholder("X1")
y  = Placeholder("Y")

w0  = Placeholder("W0")
w1  = Placeholder("W1")
w2  = Placeholder("W2")
w3  = Placeholder("W3")
w4  = Placeholder("W4")
w5  = Placeholder("W5")
w6  = Placeholder("W6")
w7  = Placeholder("W7")
w8  = Placeholder("W8")
b   = Placeholder("B")
b1   = Placeholder("B1")
b2   = Placeholder("B2")

# Evaluation model
model = net(x0,x1,w0,w1,w2,w3,w4,w5,w6,w7,w8, b, b1, b2)

# MSE cost function for SGD
cost = (model - y) * (model - y)

err = []
lr = 0.2
epochs = 100000
for i in range(epochs):

    # Set current input data
    feed["X0"] = X[i%4][0]
    feed["X1"] = X[i%4][1]
    feed["Y"]  = Y[i%4]

    print("X0:", feed["X0"])
    print("X1:", feed["X1"])
    print("Y:", feed["Y"])

    # Reset gradients
    cost.reset()
    # Compute forward pass
    cost.forward(feed)
    # Get model prediction
    model.forward(feed)
    print("Out:", model.value())
    # Obtain error 
    err.append(cost.value())
    print("Err:", cost.value())
    print("\n")

    # Compute gradient
    cost.gradient()

    # Update weights
    feed["W0"]  -= (lr * w0.grad)
    feed["W1"]  -= (lr * w1.grad)
    feed["W2"]  -= (lr * w2.grad)
    feed["W3"]  -= (lr * w3.grad)
    feed["W4"]  -= (lr * w4.grad)
    feed["W5"]  -= (lr * w5.grad)
    feed["W6"]  -= (lr * w6.grad)
    feed["W7"]  -= (lr * w7.grad)
    feed["W8"]  -= (lr * w8.grad)
    feed["B"]  -= (lr * b.grad)
    feed["B1"]  -= (lr * b1.grad)
    feed["B2"]  -= (lr * b2.grad)

print("************* Result ****************")
feed["X0"] = X[0][0]
feed["X1"] = X[0][1]
model.forward(feed)
print("X: 0 0 | Y: 0 | Result computed by model ", model.value())
feed["X0"] = X[1][0]
feed["X1"] = X[1][1]
model.forward(feed)
print("X: 0 1 | Y: 1 | Result computed by model ", model.value())
feed["X0"] = X[2][0]
feed["X1"] = X[2][1]
model.forward(feed)
print("X: 1 0 | Y: 1 | Result computed by model ", model.value())
feed["X0"] = X[3][0]
feed["X1"] = X[3][1]
model.forward(feed)
print("X: 1 1 | Y: 0 | Result computed by model ", model.value())
print("*************************************")

# Plot error
plt.plot(err)
plt.show()
