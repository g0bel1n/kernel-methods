import networkx as nx
import numpy as np

from ._base import GraphKernel
from ._utils import nx_shortest_path_2_array


class ShortestPathKernel(GraphKernel):
    def _kernel(self, G1: nx.Graph, G2: nx.Graph) -> float:
        """
        Compute the kernel value (similarity) between two graphs.

        Parameters
        ----------
        g_1 : networkx.Graph
            First graph.
        g_2 : networkx.Graph
            Second graph.

        Returns
        -------
        k : The similarity value between g_1 and g_2.
        """
        # Compute the shortest path lengths between all pairs of nodes in each graph
        sp1 = nx.shortest_path_length(G1)
        sp2 = nx.shortest_path_length(G2)

        # Create an array of the shortest path lengths for each graph
        s1 = nx_shortest_path_2_array(sp1, len(G1.nodes))
        s2 = nx_shortest_path_2_array(sp2, len(G2.nodes))

        nbins = max(s1.max(), s2.max()) + 2

        # Compute the histogram of the shortest path lengths for each graph
        h1 = np.histogram(s1, bins=np.arange(nbins), density=True)[0]
        h2 = np.histogram(s2, bins=np.arange(nbins), density=True)[0]

        # Compute the dot product of the histogram vectors
        return np.dot(h1, h2)
