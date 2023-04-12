import networkx as nx
import numpy as np

from ._base import GraphKernel


class GraphHistogramKernel(GraphKernel):


    def _kernel(self, G1: nx.Graph, G2: nx.Graph, bins=10, alpha=0.5, beta=0.5, gamma=0.5, normalize=True):
        # Compute the edge histograms of the graphs
        hist1, bin_edges = np.histogram(list(nx.get_edge_attributes(G1, 'labels').values()), bins=bins, range=(0, 1))
        hist2, bin_edges = np.histogram(list(nx.get_edge_attributes(G2, 'labels').values()), bins=bins, range=(0, 1))
        
        # Compute the node histograms of the graphs
        nodes1 = np.array([d['labels'] for n, d in G1.nodes(data=True)])
        nodes2 = np.array([d['labels'] for n, d in G2.nodes(data=True)])
        hist_nodes1, _ = np.histogram(nodes1, bins=bins, range=(0, 1))
        hist_nodes2, _ = np.histogram(nodes2, bins=bins, range=(0, 1))
        
        # Compute the kernel value as a weighted sum of the edge and node histogram dot products
        norm_edge1 = np.linalg.norm(hist1)
        norm_edge2 = np.linalg.norm(hist2)
        dot_product_edge = np.dot(hist1/norm_edge1, hist2/norm_edge2)
        norm_node1 = np.linalg.norm(hist_nodes1)
        norm_node2 = np.linalg.norm(hist_nodes2)
        dot_product_node = np.dot(hist_nodes1/norm_node1, hist_nodes2/norm_node2)
        kernel_value = alpha * dot_product_edge + beta * dot_product_node
        
        # Apply the gamma parameter to the kernel value
        kernel_value **= gamma
        
        # Normalize the kernel value by the maximum possible value
        if normalize:
            max_value = alpha * min(norm_edge1, norm_edge2) + beta * min(norm_node1, norm_node2)
            max_value **= gamma
            kernel_value /= max_value
        
        return kernel_value

    