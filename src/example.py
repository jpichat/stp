import numpy as np

import utils
from st_paths import PathsFinder, PathGenerator


if __name__ == "__main__":

    # 1. list all paths

    n = 12  # total number of vertices [0, 1, 2, ..., n-1]
    eps = 3  # max discrete distance allowed between a path vertex and its neighbour
    start = 6
    end = 5

    # make adjacency matrix
    # A = utils.make_special_adjacency_matrix(n, eps)
    A, start, end = utils.make_random_adjacency_matrix(12, density=0.7, return_st=True)

    # find paths
    finder = PathsFinder(A, start, end)
    paths = finder.get_paths()

    print(f"Found {len(paths)} ({start},{end})-paths in the graph with adjacency matrix:\n{A}")

    # 2. naive estimation of the number of s-t paths
    pg = PathGenerator(A)
    naive_estimation = pg.estimate_count_naive(start, end, n_bootstrap=500)

    print(
        f"Estimated {naive_estimation} paths"
        + f" ({np.round(np.abs(naive_estimation - len(paths)) / len(paths) * 100, 3)}% error)"
    )
