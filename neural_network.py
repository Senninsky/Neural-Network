# Importations
from settings import *
from neuron import Neuron

# Class definition
class NeuralNetwork:
    '''
    A neural network with:
     - 2 inputs
     - a hidden layer with 2 neurons (h1, h2)
     - an output layer with 1 neuron (o1)
    Each neuron has te same weights and bias:
     - w = [0, 1]
     - b = 0
    '''
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedForward(self, x):
        out_h1 = self.h1.feedForward(x)
        out_h2 = self.h2.feedForward(x)

        out_o1 = self.o1.feedForward(np.array([out_h1, out_h2]))

        return out_o1
    