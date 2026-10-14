import numpy as np
from .linear_regression import LinearRegression
from utils.activations import sigmoid

class LogisticRegression(LinearRegression):
    """
    Klasse voor het trainen en gebruiken van een logistisch regressiemodel.
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        return sigmoid(X @ self.w + self.b)
