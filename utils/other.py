import numpy as np

def variance(x, centroid):
    """
    Som van de gekwadrateerde afstanden van de punten tot hun centroid
    (within-cluster sum of squares).
    """
    return np.sum((x - centroid)**2)