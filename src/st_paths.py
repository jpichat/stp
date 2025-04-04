import numpy as np

import utils


class PathsFinder:
    def __init__(self, adjacency_matrix, start: int = None, end: int = None):
        self.neighbours_bitmask = self._build_bitmask(adjacency_matrix)
        n = len(self.neighbours_bitmask)

        if start is None:
            start = n // 2

        if end is None:
            end = start + 1 if start < n - 1 else start - 1

        self.start = start
        self.end = end

    @staticmethod
    def _build_bitmask(adjacency_matrix):
        n = adjacency_matrix.shape[0]
        bitmask = [0] * n
        for i in range(n):
            neighbours = np.where(adjacency_matrix[i] == 1)[0]
            mask = 0
            for neighbour in neighbours:
                mask |= 1 << neighbour
            bitmask[i] = mask
        return bitmask

    @utils.timeit
    def get_paths(self):
        start = self.start
        end = self.end
        neighbours_bitmask = self.neighbours_bitmask

        if start == end:
            return [[start]]

        out = []
        path = [start]
        visited = 1 << start
        stack = [(start, neighbours_bitmask[start] & ~visited)]

        while stack:
            current_node, available = stack[-1]
            if available == 0:
                # backtrack when no neighbors are left
                stack.pop()
                if path:
                    removed = path.pop()
                    visited ^= 1 << removed
                continue

            # find least significant bit (next neighbor)
            lsb = available & -available
            neighbour = int(lsb).bit_length() - 1
            stack[-1] = (current_node, available ^ lsb)  # upd available neighbours

            if neighbour == end:
                out.append(path + [neighbour])
            else:
                if not (visited & (1 << neighbour)):
                    visited |= 1 << neighbour
                    path.append(neighbour)
                    new_available = neighbours_bitmask[neighbour] & ~visited
                    stack.append((neighbour, new_available))

        return out


if __name__ == "__main__":

    from utils import PathGenerator

    # 1. list all paths

    n = 12  # total number of vertices [0, 1, 2, ..., n-1]
    eps = 3  # max discrete distance allowed between a path vertex and its neighbour
    start_vertex = 6
    end_vertex = 5

    # make adjacency matrix
    A = utils.make_special_adjacency_matrix(n, eps)

    # find paths
    finder = PathsFinder(A, start_vertex, end_vertex)
    paths = finder.get_paths()

    print(f"Found {len(paths)} ({start_vertex},{end_vertex})-paths.")

    # 2. naive estimation of the number of s-t paths
    pg = PathGenerator(A)
    naive_estimation = pg.estimate_count_naive(start_vertex, end_vertex, n_bootstrap=500)

    print(
        f"Estimated {naive_estimation} paths"
        + f" ({np.round(np.abs(naive_estimation - len(paths)) / len(paths) * 100, 3)}% error)"
    )
