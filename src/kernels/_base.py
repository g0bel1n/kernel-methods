from abc import ABC, abstractmethod

import networkx as nx
import numpy as np


class GraphKernel(ABC):
    def __init__(self):
        pass

    def __call__(self, x1, x2):
        if isinstance(x1, nx.Graph) and isinstance(x2, nx.Graph):
            return self._kernel(x1, x2)
        elif isinstance(x1, list) and isinstance(x2, list):
            return np.array(
                [
                    [self._kernel(x1[i], x2[j]) for i in range(len(x1))]
                    for j in range(len(x2))
                ]
            )

    @abstractmethod
    def _kernel(self, G1: nx.Graph, G2: nx.Graph):
        pass
