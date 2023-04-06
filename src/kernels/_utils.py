from typing import Iterable

import numpy as np


def normalize_adjacency_matrix(A: np.ndarray) -> np.ndarray:
    D = np.diag(np.sum(A, axis=0))
    return np.linalg.inv(D) @ A


def nx_shortest_path_2_array(sp: Iterable, n_nodes: int) -> np.ndarray:
    """
    Convert a dictionary of shortest path lengths to a numpy array.

    Parameters
    ----------
    sp : dict
        A dictionary of shortest path lengths.

    Returns
    -------
    s : numpy.array
        A numpy array of shortest path lengths.
    """

    sp = dict(sp)
    arr = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if i in sp and j in sp[i]:
                arr[i, j] = sp[i][j]
                arr[j, i] = sp[j][i]
    return arr
