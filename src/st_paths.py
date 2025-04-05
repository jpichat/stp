import random
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

    @utils.timeit
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
