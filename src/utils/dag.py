from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from src.dag_learner.linear import notears_linear
from src.dag_learner.notears import NOTEARS, lit_NOTEARS


def sample_nodes_to_corrupt(A, k):
    list_parents = find_parents(A)
    list_len_parents = np.array([len(parents) for parents in list_parents])
    list_possible = np.where(list_len_parents > 0)[0]
    sampled_indices = np.random.choice(list_possible, size=k, replace=False)
    return sampled_indices


def plot_graph(A, base_node):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    color_map = ["grey" for i in range(len(A))]
    color_map[base_node] = "red"
    nx.draw_kamada_kawai(G, node_color=color_map, with_labels=True)
    plt.show()


def get_adult_DAG(name_features, list_edges=None):
    A = np.zeros((len(name_features), len(name_features)))

    if list_edges is not None:
        for edge in list_edges:
            A[edge[0]][edge[1]] = 1
        return A

    list_edges = [
        [8, 6],
        [8, 14],
        [8, 12],
        [8, 3],
        [8, 5],
        [0, 6],
        [0, 12],
        [0, 14],
        [0, 1],
        [0, 5],
        [0, 3],
        [0, 7],
        [9, 6],
        [9, 5],
        [9, 14],
        [9, 1],
        [9, 3],
        [9, 7],
        [13, 5],
        [13, 12],
        [13, 3],
        [13, 1],
        [13, 14],
        [13, 7],
        [5, 6],
        [5, 12],
        [5, 14],
        [5, 1],
        [5, 7],
        [5, 3],
        [3, 6],
        [3, 12],
        [3, 14],
        [3, 1],
        [3, 7],
        [6, 14],
        [12, 14],
        [1, 14],
        [7, 14],
    ]
    for edge in list_edges:
        if edge[1] != 14:
            A[edge[0]][edge[1]] = 1
    return A


def get_DAG(D, method):

    if method == "gt":
        # Use the ground-truth DAG which was used to generate the data
        return D.DAG
    
    elif method == "notears_linear":
        X_train, _ = (
            D.train.dataset[D.train.indices][0].numpy(),
            D.test.dataset[D.test.indices][0].numpy(),
        )
        lambda1 = 0.1
        loss_type = "l2"
        A = notears_linear(
            X_train,
            lambda1,
            loss_type,
            max_iter=100,
            h_tol=1e-8,
            rho_max=1e16,
            w_threshold=0.3,
        )

        return A

    elif method == "notears_mlp":
        # Non parametric NOTEARS
        nt_h_tol = 1e-10
        nt_rho_max = 1e18
        epochs = 5
        k = 3
        graph_type = "ER"
        d = D.dim
        s = D.s0
        n = D.N

        config = {
            "model": {
                "model": NOTEARS(
                    dim=d, nonlinear_dims=[10, 1], sem_type="mlp", lambda1=0.01
                ),
                "h_tol": nt_h_tol,
                "rho_max": nt_rho_max,
                "n": n,
                "s": s,
                "dim": d,
                "K": k,
                "dag_type": graph_type,
            },
            "train": {
                "max_epochs": epochs,
                "callbacks": [
                    EarlyStopping(monitor="h", stopping_threshold=nt_h_tol),
                    EarlyStopping(monitor="rho", stopping_threshold=nt_rho_max),
                ],
                
            },
        }

        model = lit_NOTEARS(**config["model"])
        # Train the DSF
        trainer = pl.Trainer(**config["train"])
        trainer.fit(model, datamodule=D)

        A = model.A(grad=False)
        return A
    
    
    elif method == "autoregressive":
        # Triangular matrix
        A = np.tril(np.ones(D.dim), -1)
        return A
    else:
        raise ValueError("method not recognized")


def find_markov_boundary(A):
    """Given an adjacency matrix, return the markov boundary for each node.

    Args:
        A (np.array): adjacency matrix (2D array)


    Returns:
        list_markov_boundary: a list of markov boundary (one for each node in the graph, in the order of A)
    """
    list_markov_boundary = []
    for node in range(len(A)):
        parents = find_parents_node(A, node)
        children = find_children_node(A, node)
        children_parents = [
            parent
            for child in children
            for parent in find_parents_node(A, child)
            if parent != node
        ]
        markov_boundary = list(parents) + list(children) + children_parents
        list_markov_boundary.append(np.unique(markov_boundary))
    return list_markov_boundary


def find_parents_node(A, node):
    """Helper function to find parents in a graph, given the adjacency matrix

    Args:
        A (np.array): adjacency matrix(2D array)
        node(int): index of the node
    """

    column = A.T[node]
    parents = np.nonzero(column)[0]
    return parents


def find_parents(A):
    list_parents = []
    for node in range(len(A)):
        list_parents.append(find_parents_node(A, node))
    return list_parents


def find_children_node(A, node):
    """Helper function to find children in a graph, given the adjacency matrix

    Args:
        A (np.array): adjacency matrix(2D array)
        node (int): index of the node
    """
    row = A[node]
    children = np.nonzero(row)[0]
    return children
