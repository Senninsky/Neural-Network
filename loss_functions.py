# Importations
from settings import *

# Fuction definitions
def mseLoss(yTrue, yPred):
    # y-true and y-pred are numpy arrays of the same length
    return ((yTrue - yPred) ** 2).mean()