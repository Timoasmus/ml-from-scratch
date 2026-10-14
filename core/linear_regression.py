import numpy as np
from .base import Model

class LinearRegression(Model):
    """
    Klasse voor het trainen en gebruiken van een lineair regressiemodel.
    """
    def __init__(self, f_in: int):
        """
        Initialiseert de parameters van het model.

        Parameters:
            f_in (int): Het aantal features van de inputdata.
        """
   
        self.w = np.random.randn(f_in, 1) * 0.01 
        self.b = 0.0
         
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X 
        return X @ self.w + self.b 

    def backward(self, yh: np.ndarray, y: np.ndarray) -> None:
        """
        Berekent de gradiënten van w en b. (yh - y) / m is de gradiënt van ½ MSE
        (en van logistic_cost bij LogisticRegression).
        """
        m = y.shape[0]

        dyh = (yh - y) / m 

        self.dw = self.X.T @ dyh 
        self.db = np.sum(dyh)
    
    def update_parameters(self, lr: float) -> None:
        self.w -= lr * self.dw 
        self.b -= lr * self.db 
    
    def save(self, file_path: str) -> None:
        np.savez(file_path, w=self.w, b=self.b)

    def load(self, file_path: str) -> None:
        with np.load(file_path) as params:
            self.w = params['w']
            self.b = params['b']


