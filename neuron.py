# Importations
from settings import *

# Function definitions
def sigmoid(x):
    # Activation function
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

# Class definitions
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        # Weight inputs, add bias, then use an activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
