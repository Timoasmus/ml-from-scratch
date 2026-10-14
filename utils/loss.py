import numpy as np

def MAE(yh, y):
    return np.mean(np.abs(y - yh))

def MSE(yh, y):
    return np.mean((y - yh)**2)

def binary_cross_entropy(yh, y):
    eps = 1e-15
    yh = np.clip(yh, eps, 1 - eps)
    return -(y*np.log(yh) + (1 - y)*np.log(1 - yh))

def cross_entropy(yh, y):
    yh = np.clip(yh, 1e-15, 1-1e-15)
    cost = -np.mean(np.sum(y * np.log(yh), axis=1))
    return cost

def logistic_cost(yh, y):
    return np.mean(binary_cross_entropy(yh, y))

