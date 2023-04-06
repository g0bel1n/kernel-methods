import networkx as nx
import numpy as np

from ._base import GraphKernel


class VertexHistogramKernel(GraphKernel):

    def _kernel(self, G1: nx.Graph, G2: nx.Graph):
        # Compute the histogram of the vertex degrees for each graph
        h1 = np.histogram(list(dict(G1.degree).values()), bins=np.arange(0, 10), density=True)[0]
        h2 = np.histogram(list(dict(G2.degree).values()), bins=np.arange(0, 10), density=True)[0]

        # Compute the dot product of the histogram vectors
        return np.dot(h1, h2)
    

        
