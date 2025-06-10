# Importations
from settings import *
from neural_network import NeuralNetwork

# Define dataset
# Example
data_1 = np.array([
    [-2, 1],
    [25, 6],
    [17, 4],
    [-15, -6],
])

all_y_trues = np.array([
    1,
    0,
    0,
    1
])

# Xor
xor_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

xor_y_trues = np.array([
    0,
    1,
    1,
    0
])

# Train our neural network!
network = NeuralNetwork()
network.train(xor_data, xor_y_trues)