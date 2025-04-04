import time
import random
import numpy as np
from scipy import signal


def timeit(f):
    """timer decorator"""

    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print("[func:%r] ran in %2.4fs" % (f"{f.__module__}.{f.__qualname__}", te - ts))
        return result

    return timed


def vertex_degrees(n: int, eps: int):
    """
    Returns the sequence of degrees of each vertex of our eps-connected n-vertex graph using convolution
    """
    g = np.ones(n)
    el = np.hstack(([1] * eps, [0], [1] * eps))
    return signal.convolve(g, el, "same")


def make_special_adjacency_matrix(n: int, eps: int):
    A = np.zeros((n, n))
    dr, dc = np.diag_indices(n)

    for o in range(1, eps + 1):
        upper_d = dc + o
        A[(dr[:-o], upper_d[:-o])] = 1

    i_lower = np.tril_indices(n, -1)
    A[i_lower] = A.T[i_lower]

    return A.astype(int)


def make_random_adjacency_matrix(s, density=0.5, return_st=True):
    """
    Notes
    -----
    - density (float between [0,1]) describes how dense the graph is; with 1 making the graph complete
    - there might be isolated vertices
    """
    assert 0 < density <= 1

    n_density = int(np.ceil(density * sum(range(1, s))))
    A = np.zeros((s, s))
    l_upper = np.array(list(zip(*np.triu_indices_from(A, 1))))
    random_idx = np.random.permutation(len(l_upper))[:n_density]
    random_l_upper = l_upper[random_idx]

    for i in random_l_upper:
        A[i[0], i[1]] = 1
    i_lower = np.tril_indices(s, -1)
    A[i_lower] = A.T[i_lower]

    if return_st:
        l_ones = np.argwhere(A == 1)
        # pick a random 1 location in A for start/end couple
        s, e = l_ones[np.random.randint(0, len(l_ones))]
        return A.astype(int), s, e

    else:
        return A.astype(int), None, None


class PathGenerator:
    def __init__(self, adjacency_matrix):
        self.n = adjacency_matrix.shape[0]
        self.adjacency = [[] for _ in range(self.n)]
        for i in range(self.n):
            self.adjacency[i] = np.where(adjacency_matrix[i] == 1)[0].tolist()

    def make_path(self, start_node: int, end_node: int):

        if start_node == end_node:
            return [start_node], 1.0

        visited = 0  # bitmask to track visited nodes
        path = []
        current_node = start_node
        g = 1.0  # likelihood of the path being a valid path

        while True:
            path.append(current_node)
            visited |= 1 << current_node

            if current_node == end_node:
                return path, g

            neighbours = self.adjacency[current_node]
            unvisited = []
            for neighbour in neighbours:
                if not (visited & (1 << neighbour)):
                    unvisited.append(neighbour)

            if not unvisited:
                return None, 0.0  # dead-end

            # randomly select next node
            idx = random.randint(0, len(unvisited) - 1)
            next_node = unvisited[idx]
            g *= 1.0 / len(unvisited)
            current_node = next_node

    @timeit
    def estimate_count_naive(self, start_node, end_node, n: int = 5000, n_bootstrap: int = 1000):
        """Estimates count of valid s-t paths naively.

        Args:
            start_node (int) : Index of start vertex
            end_node (int) : Index of end vertex
            n (int) : Maximum number of paths to try and make in a single pass
            n_bootstrap (int) : Number of times the procedure is repeated

        Reference:
        - Roberts, B. and Kroese, D., 2007. Estimating the number of st paths in a graph. Journal of Graph Algorithms and Applications, 11(1), pp.195-214.
        """

        x = 0

        for _ in range(n_bootstrap):

            _paths_list = []
            _paths_lengths = []
            _x = 0

            for _ in range(n):

                path, g = self.make_path(start_node, end_node)
                if path:
                    _paths_lengths.append(len(path))
                    _x += 1 / g  # eq.(1)
                    if path not in _paths_list:
                        _paths_list.append(path)

            x += _x / n

        return x / n_bootstrap
