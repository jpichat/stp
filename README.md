## Computing $(s,t)$-paths of a graph


<p align="center" >
    <img src="figures/graph.png" width="350" >
</p>

### Problem definition
> we look for all the simple paths that connect vertices $s$ and $t$ (i.e., the $(s,t)$-paths) in an undirected graph, $G$.

A solution to the problem translates into either:
- listing those paths ([section 1](#h1)), in which case the count is a by-product
- deriving a method that estimates the count without having to go through all the possible paths ([section 2](#h2))

Note that the latter has little interest when one seeks the actual paths.

---

### 0. Run

you may simply run or look at the example script
```python
python src/example.py
```

### <a name="h1">1.</a> Getting the list of $(s,t)$-paths
1a. Start with creating a random adjacency matrix:
```python
from utils import make_random_adjacency_matrix

n = 12  # number of vertices
density = 0.7  # proportion of edges (higher means more edges)

A, start_node, end_node = make_random_adjacency_matrix(n, density=density, return_st=True)
```
1b. Or start with the arbitrary adjacency matrix of the graph used in the [paper](#c1), where:
    
- $\epsilon$ is a free parameter that controls the connectedness of the graph, i.e. the number of adjacent vertices to which one vertex is connected "above" and "below" (assuming the nodes set is a sequence of increasing integers $[0,1,...,n]$, such that "above" and "below" refer to greater or smaller values).
- In other words, $\epsilon$ refers to the maximum discrete distance between the current vertex and the "next, reachable" one. Note that the degrees of vertices at both ends must therefore be less than $\epsilon$.
```python
from utils import make_special_adjacency_matrix

n = 6  # total number of vertices
eps = 5  # number of jumps allowed

A = make_special_adjacency_matrix(n, eps)
```

2. then get the list of paths:
```python
from st_paths import PathsFinder

paths = PathsFinder(A, start_node, end_node).get_paths()
```

#### Example

Taking $n=6$ vertices, $\epsilon=2$, we get the following "special" adjacency matrix, $A$:

```python
[[0 1 1 0 0 0]
 [1 0 1 1 0 0]
 [1 1 0 1 1 0]
 [0 1 1 0 1 1]
 [0 0 1 1 0 1]
 [0 0 0 1 1 0]]
```
the list of all the $(3,2)$-paths is $(3 \rightarrow 1 \rightarrow 0 \rightarrow 2), (3 \rightarrow 1 \rightarrow 2), (3 \rightarrow 2), (3 \rightarrow 4 \rightarrow 2), (3 \rightarrow 5 \rightarrow 4 \rightarrow 2)$. 
> [!Note]
> The case $\epsilon=1$ is trivial and only returns the path $(3 \rightarrow 2)$.



#### Observations
- on vertices
   - the sequence of vertices degrees can be obtained:
      - by summing the elements of each row (or column) of $A$ (in the example above, that sequence is $S=[2,3,4,4,3,2]$).
      - by convolution:
         - $S=f*g$, where $f=[1,1,...,1]\in \mathbb{R}^n$ and $g=[1,..,1,0,1,..,1] \in \mathbb{R}^{2 \epsilon +1}$ (with $\epsilon$ 1s on each side of a central zero, which is used so that no vertex is self-connected).
            - one can verify that if $\epsilon \ge n-1$, then $S=[n-1,n-1,...,n-1]\in \mathbb{R}^n$ i.e., the graph is complete.
   - the vertices have degrees, $d$:
      - at least $\epsilon$ and at most $2\epsilon$, if $1 \le \epsilon \le \lfloor \frac{n}{2} \rfloor$ 
      - at least $\epsilon$ and at most $n-1$, if $\lfloor \frac{n}{2} \rfloor \lt \epsilon \lt n-1$
      - $\min_{d\in D}d=\max_{d \in D}d=n-1$, where $D$ is the degree set of $G$, otherwise.   
- on $\epsilon$
  - if $\epsilon=1$, $G$ is a path graph,
  - if $1 \lt \epsilon \lt n-2$, $G$ is $\epsilon$-connected,
  - if $\epsilon=n-1$, $G$ is a complete graph.
  - the number of diagonals of 1s above (resp. below) the main diagonal of $A$ is $\epsilon$ and it is bounded above by $n-1$, in which case the graph is complete (without self-connections).


> [!Note]
> Choosing $n=12$ and $\epsilon=n-1$ gives 9,864,101 different paths (and although complete, one can easily agree this is still a rather small graph).


### <a name="h2">2.</a> Estimating the count in random graphs

The problem of estimating the number of $(s,t)$-paths in random graphs is complicated (it is \#P-complete), as listing them all in order to get their exact count is computationally intensive for large graphs. This is where sequential Monte Carlo methods are useful. Very interesting works have been proposed in [[1]](#r1) and [[2]](#r2).

Algorithm 1 in [[1]](#r1) estimates that number, and can be accessed via:
```python
import utils
from st_paths import PathGenerator

n = 12  # total number of vertices [0, 1, 2, ..., n-1]
eps = 3  # max discrete distance allowed between a path vertex and its neighbour
start = 6
end = 5

# make adjacency matrix
A = utils.make_special_adjacency_matrix(n, eps)

# estimate count of paths
naive_estimation = PathGenerator(A).estimate_count_naive(start, end, n_bootstrap=500)
```

which gives:
```
[func:'__main__.PathsFinder.get_paths'] ran in 0.0016s
Found 764 (6,5)-paths.
[func:'utils.PathGenerator.estimate_count_naive'] ran in 14.0664s
Estimated 763.5292224000002 paths (0.062% error)
```

#### Observations
   - the estimated number of paths calculated is rather close to the actual number.
   - the distribution of generated paths lengths (see the histogram below, obtained for a different set of parameters) shows a clear bias toward shorter paths (as pointed out by the authors), since longer paths are more likely to reach "dead ends" along the way.
      - different valid $(s,t)$-paths may have the same length, which makes the histogram a bit tricky to interpret (as opposed to the direct path of length 2, here $(7,3)$, which is the only one of length 2).

<p align="center" >
    <img src="figures/histo_naive2.png" width="500" >
</p>

### References
- <a name="r1">[1]</a> Roberts, B. and Kroese, D., 2007. Estimating the number of st paths in a graph. Journal of Graph Algorithms and Applications, 11(1), pp.195-214.
- <a name="r2">[2]</a> Bax, E.T., 1994. Algorithms to count paths and cycles. Information Processing Letters, 52(5), pp.249-252.


### <a name="c1"></a>Citation
This code was used to resolve some problems addressed in the paper [(Pichat, 2015) A multipath approach to histology volume reconstruction](http://discovery.ucl.ac.uk/1468614/3/ISBI2015_tig.pdf). Please cite it if this happened to be useful to your work!
```
@inproceedings{pichat2015multi,
  title={A multi-path approach to histology volume reconstruction},
  author={Pichat, Jonas and Modat, Marc and Yousry, Tarek and Ourselin, Sebastien},
  booktitle={2015 IEEE 12th International Symposium on Biomedical Imaging (ISBI)},
  pages={1280--1283},
  year={2015},
  organization={IEEE}
}
```
