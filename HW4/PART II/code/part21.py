import numpy as np
import matplotlib.pyplot as plt
import PIL
import scipy
import tkinter
import maxflow

if __name__ == "__main__":
    g = maxflow.Graph[int](2, 4)
    nodes = g.add_nodes(3)

    g.add_edge(nodes[0], nodes[1], 3, 2)
    g.add_edge(nodes[1], nodes[2], 5, 1)

    g.add_tedge(nodes[0], 9, 4)
    g.add_tedge(nodes[1], 7, 7)
    g.add_tedge(nodes[2], 5, 8)

    flow = g.maxflow()

    print("Maxflow flow:", flow)

    print("Segment of the node 0:", g.get_segment(nodes[0]))
    print("Segment of the node 1:", g.get_segment(nodes[1]))
    print("Segment of the node 2:", g.get_segment(nodes[2]))

