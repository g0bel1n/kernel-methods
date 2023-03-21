

#random walk kernel with networkx graphs

import networkx as nx
import numpy as np

class Walk:

    def __init__(self, path):
        self.path = path

    def __eq__(self, other):
        #recursively check if two walks are equal
        if len(self.path) != len(other.path):
            return False
        elif len(self.path) == 1:
            return self.path[0] == other.path[0]
        else:
            return self.path[0] == other.path[0] and self.path[1:] == other.path[1:]
        
    def __hash__(self):
        return hash(tuple(self.path))

class RandomWalkKernel:
    
    def __init__(self) -> None:
        pass


    def __call__(self, x1, x2):
        if isinstance(x1, nx.Graph) and isinstance(x2, nx.Graph):
            return self._random_walk_kernel(x1, x2)
        elif isinstance(x1, list) and isinstance(x2, list):
            return np.array([[self._random_walk_kernel(x1[i], x2[j]) for i in range(len(x1))] for j in range(len(x2))])
        
    @staticmethod
    def _random_walk(G : nx.Graph, len_max: int):
        """
        Random walk on graph G, starting from node 0, with maximum length len_max
        """
        idx= np.random.choice(np.arange(len(G.nodes)))
        path = [list(G.nodes)[idx]]
        for _ in range(len_max):
            if neighbors := list(G.neighbors(path[-1])):
                idx = np.random.choice(np.arange(len(neighbors)))
                path.append(neighbors[idx])
            else:
                break
        return Walk(path)
    

    @staticmethod
    def product_graph(G1, G2, edge_agg = np.sum):
        G = nx.Graph()
        for n1 in G1.nodes:
            for n2 in G2.nodes:
                if G1.nodes[n1]['labels'] == G2.nodes[n2]['labels']:
                    G.add_node((n1, n2), labels=G1.nodes[n1]['labels'])
        for e1 in G1.edges:
            for e2 in G2.edges:
                if (e1[0], e2[0]) in G.nodes and (e1[1], e2[1]) in G.nodes:
                    G.add_edge((e1[0], e2[0]), (e1[1], e2[1]), labels=[edge_agg((G1.edges[e1]['labels'],G1.edges[e1]['labels']))])
        return G

    def _random_walk_kernel(self,G1 : nx.Graph, G2 : nx.Graph, n_random_walks : int = 100, len_max : int = 20):
        """
        Random walk kernel between graphs G1 and G2
        """

        Gx = self.product_graph(G1, G2, edge_agg = np.min)
        if not (
            connected_sub_graphs := [
                Gx.subgraph(c) for c in nx.connected_components(Gx)
            ]
        ):
            return np.nan
        #keep largest connected component
        Gx = max(connected_sub_graphs, key=len)

        mapping = {node: i for i, node in enumerate(Gx.nodes)}
        Gx = nx.relabel_nodes(Gx, mapping)

        walks = [self._random_walk(Gx, len_max) for _ in range(n_random_walks)]

        transitions_matrix = np.zeros((len(Gx.nodes), len(Gx.nodes)))

        n_nodes_visited = 0
        for walk in walks:
            for i in range(len(walk.path)-1):
                transitions_matrix[walk.path[i]][walk.path[i+1]] += (Gx[walk.path[i]][walk.path[i+1]]['labels'][0]+1)**(Gx.nodes[walk.path[i]]['labels'][0]+1)
                transitions_matrix[walk.path[i+1]][walk.path[i]] += (Gx[walk.path[i]][walk.path[i+1]]['labels'][0]+1)**(Gx.nodes[walk.path[i+1]]['labels'][0]+1)
            n_nodes_visited += len(walk.path)

        #normalize transitions matrix to get a probability matrix, beware of 0s

        transitions_matrix = transitions_matrix / n_nodes_visited

        #normalize transitions matrix to get a probability matrix, beware of 0s
        np.fill_diagonal(transitions_matrix, 0)
        #cols= transitions_matrix.sum(axis=1)!=0
        #transitions_matrix[cols] /= transitions_matrix[cols].sum(axis=1)[:,None]
        initial_distribution = np.ones(len(Gx.nodes))/len(Gx.nodes)

        #print(transitions_matrix)
        #print(np.max(np.linalg.inv(np.eye(len(Gx.nodes))-transitions_matrix)))
        return initial_distribution @ np.linalg.inv(np.eye(len(Gx.nodes))-transitions_matrix) @ np.ones(len(Gx.nodes))
        #return transitions_matrix


    def copy(self):
        return RandomWalkKernel()
    