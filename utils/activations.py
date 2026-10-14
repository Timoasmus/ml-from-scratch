import numpy as np

def sigmoid(x):
    # Gelijk aan 1 / (1 + e^-x), maar zonder overflow bij grote negatieve x.
    return np.exp(-np.logaddexp(0, -x))

def softmax(x):
    expX = np.exp(x - np.max(x, axis=1, keepdims=True))
    return expX / np.sum(expX, axis=1, keepdims=True)

def ReLU(x):
    return np.maximum(0, x)

def dReLU(X):
    return (X > 0).astype(float)