import numpy as np
from graph_cut import *
from itertools import combinations

def get_pairwises(adj):
    n = np.size(adj, 0)
    pairs = combinations([i for i in range(0, n)], 2)

    edge_relations = []
    for i, j in pairs:
        edge_relations.append([i, j, 0, adj[i, j], 0, 0])
        edge_relations.append([j, i, 0, adj[j, i], 0, 0])

    return np.asarray(edge_relations)


if __name__ == '__main__':

    g = GraphCut(3, 4)

    unaries = np.array([[4, 9],
                        [7, 7],
                        [8, 5]])

    adjacency_matrix = np.array([[0, 3, 0],
                                 [2, 0, 5],
                                 [0, 1, 0]])

    pairwises = get_pairwises(adjacency_matrix)

    g.set_unary(unaries)
    g.set_pairwise(pairwises)
    flow = g.minimize()
    labels = g.get_labeling()

    print("Maxflow flow:", flow)

    print("Label of the node 0:", int(labels[0]))
    print("Label of the node 1:", int(labels[1]))
    print("Label of the node 2:", int(labels[2]))


