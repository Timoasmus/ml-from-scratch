import numpy as np
from utils.other import variance
from typing import List, Tuple

def cluster_points(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Stopt alle punten in een cluster, op basis van de afstand tot de centroids.
    """

    distances = np.linalg.norm(X[:, None] - centroids, axis=2)
    return np.argmin(distances, axis=1)
 

def kmeans_cluster(X: np.ndarray, K: int, centroids: np.ndarray,
                   max_iter: int = 300) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Stopt de punten in clusters totdat de centroids, berekend door het gemiddelde van
    een cluster, niet meer veranderen (of max_iter is bereikt).
    """

    for _ in range(max_iter):
        labels = cluster_points(X, centroids)
        new_centroids =  np.array([
            X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
            for k in range(K)
        ])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    clusters = [X[labels == k] for k in range(K)]
    return clusters, centroids



def kmeans_iteration(X: np.ndarray, K: int) -> Tuple[List, np.ndarray]:
    """
    Een enkele iteratie: initialiseert de centroids willekeurig.
    """

    indices = np.random.permutation(X.shape[0])[:K]
    centroids = X[indices]
    return kmeans_cluster(X, K, centroids)

    
def kmeans_fit(X: np.ndarray, K: int, iters: int) -> Tuple[List[np.ndarray], float]:
    """
    Voert k-means iters keer uit (elke keer met andere startcentroids) en kiest
    het resultaat met de laagste variatie binnen de clusters.

    Parameters:
        - X (np.ndarray) : De punten die geclusterd moeten worden.
        - K (int) : Het aantal clusters.
        - iters (int) : Het aantal runs.

    Returns:
        - De clusters van de beste run en de bijbehorende variatie.
    """

    clusterings = []
    variances = []

    for _ in range(iters):
        clusters, centroids = kmeans_iteration(X, K)
        var = sum([variance(clusters[k], centroids[k]) for k in range(K)])

        clusterings.append(clusters)
        variances.append(var)

    min_idx = np.argmin(np.array(variances))
    return clusterings[min_idx], variances[min_idx]
