import numpy as np
from ..base import Model
from typing import List, Literal, Optional
from .layer import Layer
from utils.activations import sigmoid, softmax

class NeuralNetwork(Model):
    """
    Klasse voor het trainen en gebruiken van een neuraal netwerk, die intern
    alle lagen bijhoudt en orkestreert.
    """
    def __init__(self, f_in: int, 
                 act_func: Optional[Literal["sigmoid", "softmax"]] = None):
        """
        Initialiseert het neurale netwerk, met een lege lijst layers, het aantal
        input features van de input data en de juiste activatiefunctie.

        Parameters:
         - f_in (int): Aantal input features.
         - act_func (Literal["sigmoid", "softmax"] | None): activatiefunctie van
            de output layer. None betekent een lineaire output (regressie).
        """

        self.layers: List[Layer] = []
        self.f_prev = f_in

        func_mapping = {
            "sigmoid": sigmoid,
            "softmax": softmax,
        }
        if act_func is not None and act_func not in func_mapping:
            raise ValueError(f"Onbekende activatiefunctie: {act_func}. "
                             f"Kies uit {list(func_mapping)} of None.")
        self.act_func = func_mapping.get(act_func)

    def forward(self, X: np.ndarray) -> np.ndarray:
        for l in self.layers:
            X = l.forward(X)

        if self.act_func:
            return self.act_func(X)
        return X 
    
    def backward(self, yh: np.ndarray, y: np.ndarray) -> None:
        """
        Backpropagation door alle lagen.

        De gradiënt van de output is (yh - y) / m. Dit is de juiste gradiënt
        voor de volgende combinaties van activatie- en lossfunctie:
         - softmax + cross_entropy
         - sigmoid (1 output neuron) + logistic_cost
         - geen activatie (1 output neuron) + ½ MSE
        """
        self.layers[-1].final = True

        m = y.shape[0]
        dprev = (yh - y) / m
        for l in reversed(self.layers):
            dprev = l.backward(dprev)

    def update_parameters(self, lr: float) -> None:
        for l in self.layers:
            l.update_parameters(lr)
    
    def save(self, file_path: str) -> None:
        for i, l in enumerate(self.layers):
            l.save(file_path, i+1)

    def load(self, file_path: str) -> None:
        for i, l in enumerate(self.layers):
            l.load(file_path, i+1)

    def add(self, neurons: int | List[int]) -> None:
        if self.layers:
            self.layers[-1].final = False

        if isinstance(neurons, int):
            self._add_neurons(neurons)
        else:
            [self._add_neurons(n) for n in neurons]

        self.layers[-1].final = True

    def _add_neurons(self, neurons: int):
        self.layers.append(Layer(f_in=self.f_prev, 
                                          f_out=neurons))
        self.f_prev = neurons

    def __repr__(self):
        first_line = "="*25 + "Neural Network" + "="*25
        final_line = "="* 64

        name = self.act_func.__name__ if self.act_func else None
        activation_function = f"-- Activation function: {name}"

        body = "\n".join([f"-- {i + 1}: {l}" for i, l in enumerate(self.layers)])
        return "\n".join([first_line, body, activation_function, final_line])