import time
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
    Returns the sequence of degrees of each vertex of an eps-connected n-vertex graph using convolution
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
