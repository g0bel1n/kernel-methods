from ._base import GraphKernel
import networkx as nx
import numpy as np


class WLKernel(GraphKernel):
    def __init__(self, num_iterations: int = 5, kernel_degree: int = 3):
        self.num_iterations = num_iterations
        self.kernel_degree = kernel_degree

    @staticmethod
    def get_initial_node_labels(G: nx.Graph) -> np.ndarray:
        labels = nx.get_node_attributes(G, "labels")
        return np.array(list(labels.values())).reshape(-1, 1)

    @staticmethod
    def wl_iteration(G: nx.Graph, labels: np.ndarray) -> np.ndarray:
        new_labels = {}
        for node in G.nodes():
            # get node label
            label = labels[node][0]

            # get labels of neighboring nodes
            neighbors = list(G.neighbors(node))
            neighbor_labels = sorted([labels[n][0] for n in neighbors])

            # compute new label
            new_label = hash(
                (label, *neighbor_labels)
            )  # hash the tuple instead of joining strings

            # add new label to dictionary
            new_labels[node] = new_label

        return np.array(list(new_labels.values())).reshape(-1, 1)

    def _kernel(self, G1: nx.Graph, G2: nx.Graph) -> float:
        # compute initial node labels
        node_labels1 = self.get_initial_node_labels(G1)
        node_labels2 = self.get_initial_node_labels(G2)

        # iterate over WL iterations
        for _ in range(self.num_iterations):
            node_labels1 = self.wl_iteration(G1, node_labels1)
            node_labels2 = self.wl_iteration(G2, node_labels2)

        # compute kernel value
        set1 = set(node_labels1.reshape(-1))
        set2 = set(node_labels2.reshape(-1))
        intersection = len(set1 & set2)
        return intersection / (len(node_labels1) ** self.kernel_degree)
