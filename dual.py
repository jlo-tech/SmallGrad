class Dual(object):

    def __init__(self, v=0, dv=0):
        self.v = v      # Value
        self.dv = dv    # Derivate

    def __str__(self):
        return str(self.v) + ("+" if self.dv >= 0 else "-") + str(abs(self.dv))

    def __neg__(self):
        return Dual(-self.v, -self.dv)

    def __add__(self, other):
        return Dual(self.v + other.v, self.dv + other.dv)

    def __sub__(self, other):
        return Dual(self.v - other.v, self.dv - other.dv)

    def __mul__(self, other):
        return Dual((self.v * other.v), ((self.v * other.dv) + (self.dv * other.v)))

    def __div__(self, other):
        if other.v == 0:
            raise DivisionByZero
        return Dual(self.v / other.v, ((self.dv * other.v) - (self.v * other.dv)) / (other.v * other.v))


""" 
# Example of computing the gradient for a simple function using dual numbers:

def f(x):
    return (Dual(2, 0) * x) + Dual(3, 0)

res = f(Dual(2, 1)).v
grad = f(Dual(2, 1)).dv
print("Result:", res, "Gradient:", grad)
print(f(Dual(2, 1)))
"""