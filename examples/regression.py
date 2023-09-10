import random
from graph import *
import matplotlib.pyplot as plt

"""
	A simple linear regression example
"""

# Input Data
X = [-1, 0, 1, 2]

# Output Data
Y = [
    -10,
    20,
    50,
    80,
]

# Network topology: 2 Input, 2 Hidden, 1 Output
def line(x, m, b):
    return m * x + b

# Dictionary holds actual values
feed = {
    "X": 0,
    "Y": 0,
    "M": random.random(),
    "B": random.random(),
}

# Variables of the model and cost function
x = Placeholder("X")
y = Placeholder("Y")
m = Placeholder("M")
b = Placeholder("B")

# Model instance
model = line(x, m, b)

# MSE cost function to perform SGD
cost = ((y - model) * (y - model))

err = []

lr = 0.001
epochs = 1000
for i in range(epochs):

    feed["X"] = X[i%len(X)]
    feed["Y"] = Y[i%len(X)]

    print("================================")
    print("X:", feed["X"])
    print("Y:", feed["Y"])

    # Compute prediction of the model
    model.forward(feed)

    print("R:", model.value())

    # Reset gradient values
    cost.reset()
    e = cost.forward(feed)
    err.append(e)
    print("Err: ", e)

    # Compute gradient
    cost.gradient()

    # Update gradients
    feed["M"] = feed["M"] - (lr * m.grad)
    feed["B"] = feed["B"] - (lr * b.grad)

    print("M Grad:", m.grad)
    print("B Grad:", b.grad)

    print("================================\n")

print("**************** Result ******************")
print("Perfect value for M: 30, post training value: ", feed["M"])
print("Perfect value for B: 20, post training value: ", feed["B"])
print("******************************************")

# Visualize convergence of the algorithm
plt.plot(range(len(err)), err)
plt.show()