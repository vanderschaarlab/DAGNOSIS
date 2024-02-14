# adapted from Zheng et al. (2020) -> https://github.com/xunzheng/notears/blob/master/notears/utils.py

# stdlib
import copy

# third party
import igraph as ig
import numpy as np
from scipy.special import expit as sigmoid


def is_dag(W):
    """Check whether the adjacency matrix W represents a DAG."""
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def find_parent_size(DAG, feature):

    G = ig.Graph.Adjacency(DAG.tolist())
    parents = G.neighbors(feature, mode=ig.IN)
    pa_size = len(parents)
    return pa_size


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP
    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif graph_type == "custom":
        B = np.zeros((d, d))
        B[1:s0, 0] = 1
        return B

    elif graph_type == "chain":
        B = np.eye(d, d, k=1)
        return B

    else:
        raise ValueError("unknown graph type")
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def modify_single_sem(
    pa_size,
    corruption_type,
    feature_corruption,
    list_SEMs,
    list_parameters,
    mean_linear=5,
    std_linear=1,
    mean_mlp=2,
    std_mlp=1.0,
    sample_last_layer=True,
):
    # Given a list of SEMs, modify the SEM indexed at feature_corruption. This list will then be leveraged to generate corrupted data.
    new_SEM_list = copy.deepcopy(list_SEMs)
    list_parameters_corrupted = copy.deepcopy(list_parameters)

    if corruption_type == "gaussian_noise":
        # Add some gaussian noise to the parameters of the SEM

        if "W" in list_parameters[feature_corruption].keys():
            # Linear SEM
            W = list_parameters[feature_corruption]["W"]
            gaussian_noise = (np.random.random(size=W.shape) + mean_linear) * std_linear
            W_corrupted = W + gaussian_noise
            corrupted_SEM = lambda X, z: X @ W_corrupted + z  # noqa: E731
            parameter = {"W": W_corrupted}

        elif "W2" in list_parameters[feature_corruption].keys():
            # MLP SEM
            W2 = list_parameters[feature_corruption]["W2"]
            if sample_last_layer:
                coordinates = np.random.choice(range(W2.shape[0]), 5, replace=False)
                gaussian_noise = np.random.random(size=coordinates.shape)

                W2_corrupted = W2
                W2_corrupted[coordinates] = W2_corrupted[coordinates] + std_mlp * (
                    mean_mlp + gaussian_noise
                )
            else:
                gaussian_noise = np.random.random(size=W2.shape)
                W2_corrupted = W2 + std_mlp * (gaussian_noise + mean_mlp)

            W1 = list_parameters[feature_corruption]["W1"]
            corrupted_SEM = (
                lambda X, z: sigmoid(X @ W1) @ W2_corrupted + z
            )  # noqa: E731
            parameter = {"W1": W1, "W2": W2_corrupted}
        else:
            raise ValueError("Not linear or mlp")

    else:
        # Generate a new SEM
        corrupted_SEM, parameter = generate_SEM(corruption_type, pa_size)

    new_SEM_list[feature_corruption] = corrupted_SEM
    list_parameters_corrupted[feature_corruption] = parameter
    return new_SEM_list, list_parameters_corrupted


def generate_SEM(sem_type, pa_size):
    parameters = {}
    if pa_size == 0:
        return lambda X, z: z, {}
    if sem_type == "mlp":
        hidden = 100

        W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
        W1[np.random.rand(*W1.shape) < 0.5] *= -1
        W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
        W2[np.random.rand(hidden) < 0.5] *= -1

        parameters["W1"] = W1
        parameters["W2"] = W2

        SEM = lambda X, z: sigmoid(X @ W1) @ W2 + z  # noqa: E731

    elif sem_type == "mim":
        w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
        w1[np.random.rand(pa_size) < 0.5] *= -1
        w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
        w2[np.random.rand(pa_size) < 0.5] *= -1
        w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
        w3[np.random.rand(pa_size) < 0.5] *= -1

        parameters["w1"] = w1
        parameters["w2"] = w2
        parameters["w3"] = w3

        SEM = (
            lambda X, z: np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        )  # noqa: E731

    elif sem_type == "constant":
        C = np.random.uniform(low=-10, high=10)
        SEM = lambda X, z: C  # noqa: E731
        parameters["C"] = C

    elif sem_type == "linear":
        W = np.random.uniform(low=0.5, high=2.0, size=[pa_size])
        W[np.random.rand(*W.shape) < 0.5] *= -1  # flip the sign with probability 0.5
        SEM = lambda X, z: X @ W + z  # noqa: E731
        parameters["W"] = W

    else:
        raise ValueError("SEM type not in list")
    return SEM, parameters


def generate_list_SEM(B, sem_type):
    d = B.shape[0]
    G = ig.Graph.Adjacency(B.tolist())
    list_parameters = []
    list_SEM = []
    for j in range(d):
        parents = G.neighbors(j, mode=ig.IN)
        pa_size = len(parents)
        SEM, parameters = generate_SEM(sem_type, pa_size)
        list_SEM.append(SEM)
        list_parameters.append(parameters)
    return list_SEM, list_parameters


def simulate_sem_by_list(B, n, list_SEMs, noise_scale=None, noise_mean_list=None):
    """Simulates data given a list of SEMs

    Args:
        B (array): Adjacency matrix used to find the parents
        n (int): number of samples
        list_SEMs (list): list of SEMs
        noise_scale (float, optional): std for the noise. Defaults to None.
        noise_mean_list (float, optional): mean for the noise. Defaults to None.

    Returns:
        _type_: data
    """

    d = B.shape[0]
    noise_mean_list = np.zeros(d) if noise_mean_list is None else noise_mean_list
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:

        parents = G.neighbors(j, mode=ig.IN)
        X_corrupted = copy.deepcopy(X)
        z = np.random.normal(loc=noise_mean_list[j], scale=scale_vec[j], size=n)
        X[:, j] = list_SEMs[j](X_corrupted[:, parents], z)

    return X


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.
    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition
    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG
    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError("B_est should take value in {0,1,-1}")
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError("undirected edge should only appear once")
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError("B_est should take value in {0,1}")
        if not is_dag(B_est):
            raise ValueError("B_est should be a DAG")
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {"fdr": fdr, "tpr": tpr, "fpr": fpr, "shd": shd, "nnz": pred_size}
