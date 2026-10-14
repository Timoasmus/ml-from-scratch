import os
import numpy as np
from utils.activations import ReLU, dReLU

class Layer():
    def __init__(self, f_in: int, f_out: int):
   
        self.W = np.random.randn(f_in, f_out) * np.sqrt(2 / f_in)
        self.b = np.zeros((1, f_out))

        self.f_in = f_in 
        self.f_out = f_out

        self.final = False

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.W + self.b 
        self.X = X
        self.Z = Z 

        if self.final:
            return Z 
        
        A = ReLU(Z)
        return A 
    
    def backward(self, dA: np.ndarray) -> np.ndarray:
        if self.final:
            dZ = dA
        else:
            dZ = dReLU(self.Z) * dA 

        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)

        dA_prev = dZ @ self.W.T 
        return dA_prev
    
    def update_parameters(self, lr: float) -> None:
        self.W -= lr * self.dW 
        self.b -= lr * self.db 

    def save(self, file_path: str, i: int) -> None:
        os.makedirs(file_path, exist_ok=True)
        path = os.path.join(file_path, f'layer{i}.npz')
        np.savez(path, W=self.W, b=self.b)

    def load(self, file_path: str, i: int) -> None:
        path = os.path.join(file_path, f'layer{i}.npz')
        with np.load(path) as params:
            self.W = params['W']
            self.b = params['b']

    def __repr__(self):
        return (f"Dense Layer (f_in: {self.f_in}, f_out: {self.f_out}), " 
                f"{'No' if self.final else 'ReLU'} activation")