import numpy as np

class Model:
    """
    Basisklasse die de algemene structuur omschrijft voor ml-modellen.
    """
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Enkele forward pass van het model.
        """
        raise NotImplementedError

    def backward(self, yh: np.ndarray, y: np.ndarray) -> None:
        """
        Enkele backward pass van het model.
        """
        raise NotImplementedError

    def update_parameters(self, lr: float) -> None:
        """
        Werkt de parameters van het model bij.
        """
        raise NotImplementedError

    def save(self, file_path: str) -> None:
        """
        Slaat de gewichten van het model op.

        Parameters:
         - file_path (str): path waar het model opgeslagen moet worden.
        """
        raise NotImplementedError

    def load(self, file_path: str) -> None:
        """
        Laadt opgeslagen gewichten in het model.
        (De dimensies van het model en de opgeslagen gewichten moeten hiervoor
         exact overeenkomen).

        Parameters:
         - file_path (str): path waar de gewichten opgeslagen zijn.
        """
        raise NotImplementedError
