from typing import Optional

import networkx as nx
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import cg

from ._base import GraphKernel
from ._utils import normalize_adjacency_matrix
from .subkernels import BinaryNodeKernel, LinearEdgeKernel
from .subkernels._base import EdgeKernel, NodeKernel


class RandomWalkKernel(GraphKernel):
    def __init__(
        self,
        edge_kernel: Optional[EdgeKernel] = None,
        node_kernel: Optional[BinaryNodeKernel] = None,
        normalize=False,
        node_labels=True,
        c: float = 0.1,
    ):

        self.edge_kernel = edge_kernel or LinearEdgeKernel()
        self.node_kernel = node_kernel or BinaryNodeKernel()
        self.normalize = normalize
        self.node_labels = node_labels
        self.c = c

    def _kernel(self, G1: nx.Graph, G2: nx.Graph) -> float:
        p1, q1 = np.ones(len(G1.nodes)), np.ones(len(G1.nodes))
        p2, q2 = np.ones(len(G2.nodes)), np.ones(len(G2.nodes))
        p, q = np.kron(p1, p2), np.kron(q1, q2)

        #
        if self.node_labels:
            W = np.zeros((len(G1.nodes) * len(G2.nodes), len(G1.nodes) * len(G2.nodes)))
            for e1 in G1.edges(data=True):
                for e2 in G2.edges(data=True):
                    idx = (e1[0] * len(G2.nodes) + e2[0], e1[1] * len(G2.nodes) + e2[1])
                    W[idx] = self.edge_kernel(
                        e1[2]["labels"][0], e2[2]["labels"][0]
                    ) * (
                        self.node_kernel(
                            G1.nodes[e1[0]]["labels"][0], G2.nodes[e2[0]]["labels"][0]
                        )
                        * self.node_kernel(
                            G1.nodes[e1[1]]["labels"][0], G2.nodes[e2[1]]["labels"][0]
                        )
                        + self.node_kernel(
                            G1.nodes[e1[0]]["labels"][0], G2.nodes[e2[1]]["labels"][0]
                        )
                        * self.node_kernel(
                            G1.nodes[e1[1]]["labels"][0], G2.nodes[e2[0]]["labels"][0]
                        )
                    )
                    W[idx[::-1]] = W[idx]

                    idx2 = (
                        e1[0] * len(G2.nodes) + e2[1],
                        e1[1] * len(G2.nodes) + e2[0],
                    )
                    W[idx2] = W[idx]
                    W[idx2[::-1]] = W[idx]
        else:
            if self.normalize:
                A1, A2 = normalize_adjacency_matrix(
                    nx.adjacency_matrix(G1).todense()
                ), normalize_adjacency_matrix(nx.adjacency_matrix(G2).todense())
            else:
                A1, A2 = (
                    nx.adjacency_matrix(G1).todense(),
                    nx.adjacency_matrix(G2).todense(),
                )
            W = np.kron(A1.T, A2.T)

        x, _ = cg(identity(W.shape[0]) - self.c * W, p)

        return x @ q

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.edge_kernel.__name__}, {self.node_kernel.__name__}, {self.normalize}, {self.node_labels}, {self.c})"
