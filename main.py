# Importations
from settings import *
from loss_functions import mseLoss

# Example
yTrue = np.array([1, 0, 0, 1])
yPred = np.array([0, 0, 0, 0])
print(mseLoss(yTrue, yPred))