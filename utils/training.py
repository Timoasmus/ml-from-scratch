import numpy as np
from core.base import Model
from typing import List, Tuple, Callable

def model_fit(model: Model, X: np.ndarray, y: np.ndarray, 
               cost_function: Callable, epochs: int, lr: float, 
               X_val: np.ndarray = None, y_val: np.ndarray = None, 
               batch_size: int = None, print_every: int = 10) -> Tuple[List, List]:
    """
    Traint een model een bepaald aantal epochs, met optionele validation set
    en batch grootte.

    Parameters:
     - model (Model): Het model dat getraind moet worden.
     - X (np.ndarray): Input data waarop het model getraind wordt.
     - y (np.ndarray): Output data waarop het model getraind wordt.
     - cost_function (Callable): Functie waarmee de cost van de output berekend
        moet worden. 
     - epochs (int): Aantal keer dat over de hele trainingsdata getraind wordt.
     - lr (float): learning rate
     - X_val (np.ndarray): Optionele input data van validation set.
     - y_val (np.ndarray): Optionele output data van validation set.
     - batch_size (int): Optionele grootte voor batch, als er per batch getraind
        moet worden. De data wordt elke epoch opnieuw geschud.
     - print_every (int): Om de hoeveel epochs de cost geprint wordt.
    """

    m = X.shape[0]
    train_costs = []
    val_costs = []

    for i in range(epochs):

        if batch_size:
            perm = np.random.permutation(m)
            X, y = X[perm], y[perm]

            costs = []
            for j in range(0, m, batch_size):
                current_batch_size = min(batch_size, m - j)
                Xb = X[j:j+current_batch_size, :]
                yb = y[j:j+current_batch_size, :]
                cost = iteration(model, Xb, yb, cost_function, lr)
                costs.append(cost)
            cost = np.mean(costs)
        
        else:
            cost = iteration(model, X, y, cost_function, lr)

        train_costs.append(cost)

        if X_val is not None:
            # Cost van validation set berekenen, als deze is meegegeven.
            val_cost = iteration(model, X_val, y_val, cost_function, update=False)
            val_costs.append(val_cost)

        if i % print_every == 0 or i == epochs - 1:
            line = f"Epoch {i:>4} | train cost: {cost:.4f}"
            if X_val is not None:
                line += f" | validation cost: {val_cost:.4f}"
            print(line)

    return train_costs, val_costs



def iteration(model: Model, X: np.ndarray, y: np.ndarray,
                cost_function: Callable, lr:float = None, update: bool = True) -> float:
    yh = model.forward(X)
    cost = cost_function(yh, y)

    if update:
        model.backward(yh, y)
        model.update_parameters(lr)
    
    return cost


