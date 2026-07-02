> [!ABSTRACT]
> 
> While **[[Breadth First Search (BFS)|BFS]]** finds the shortest path in unweighted graphs (least number of edges), it fails when edges have **varying costs**. **Dijkstra's Algorithm** is a [[Computer Science/Algorithms/Greedy Algorithms/index|Greedy Algorithm]] that solves the **Single-Source Shortest Path** problem for weighted graphs, provided all edge weights are **non-negative**.

---
# Why BFS Fails on Weighted Graphs

BFS assumes that every edge has a cost of $1$. In a weighted graph, a path with **more edges** might actually have a **lower total weight** than a direct edge.

- **BFS Path:** $A \to C$ (Total Weight: 30)
- **Shortest Weighted Path:** $A \to B \to C$ (Total Weight: $12 + 5 = 17$)

Dijkstra's Algorithm accounts for these costs by prioritizing paths with the smallest cumulative weight.

---
# The Greedy Strategy

Dijkstra's is a [[Computer Science/Algorithms/Greedy Algorithms/index|Greedy Algorithm]]. It makes the optimal choice at each step—picking the closest unvisited vertex and assumes that this choice will lead to the overall shortest path.

## The Core Logic:

1. Assign a **Distance** of infinity to all nodes, except the start node (which is 0).
2. Maintain a [[Priority Queue]] to store `(distance, vertex)` pairs.
3. Always "relax" the neighbor: If the path to a neighbor through the current node is shorter than its previously known distance, update its distance and add it to the PQ.
4. Once a node is "Done" (dequeued), its shortest path is guaranteed.

---
# Pseudocode Implementation

Dijkstra's uses a **Priority Queue** to efficiently find the next vertex with the minimum distance.

```pseudo
	\begin{algorithm}
	\caption{Dijkstra}
	\begin{algorithmic}
		\Procedure{Dijkstra}{$G, \ell, s$}
			\ForAll{$u \in V$}
				\State dist($u$) $:= \infty$
				\State prev($u$) $:=$ null
            \EndFor
            \State dist($s$) $:= 0$
            \State $H$ := makequeue($V$)
            \While{$H$ is not empty}
	            \State $u := $ deletemin($H$)
	            \ForAll{edges $(u, v) \in E$}
		            \If{dist($v$) > dist($u$) + $\ell(u, v)$}
			            \State dist($v$) $:=$ dist($u$) + $\ell(u, v)$
			            \State prev($v$) $:=$ $u$
			            \State decreasekey($H, v$)
                    \EndIf
                \EndFor
            \EndWhile
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!Important] Pick $v$ in $F$ carefully
> **Dijkstra's Algorithm** falls into a problem where vertices may re-enter $F$ more than once.
> 
> If we pick $v$ in $F$ carefully, we can avoid this!!!

---
# Proof of Correctness
## Claim: 
Let $d(v)$ be the length of the shortest path from $s$ to $v$. Then after every iteration, $dist(v) = d(v)$ for all vertices $v$ in $X$

## Proof:
### Base Case
The first vertex to move into $X$ is $s$ → $dist(s) = 0 = d(s)$

### Inductive Hypothesis:
After $k$ vertices have been moved into $X$, assume $dist(v) = d(v)$ for all vertices in $X$

### Inductive Step:
Suppose $u$ is the next vertex to move into $X \dots$
Want to show that $dist(u) = d(u)$

Suppose by contradiction that $dist(u) > d(u)$, implying that there exist a path $P$ such that $length(P) = d(u)$

![[Pasted image 20260701193040.png]]

$P$ goes from $s$ to $u$, so there is an edge that crosses the boundary of $X$
	$dist(w) = d(w)$ by **Inductive Hypothesis**
	$dist(y) \geq dist(u)$ by choice of $u$
	
Therefore 
$$
\begin{align*}
d(u) &= \underbrace{ len(P) }_{ s \to u } \geq \underbrace{ dist(w) + \ell(w,y) }_{ s \to y } \\
&= dist(y) \geq dist(u) > d(u)\\
&\therefore \boxed{d(u) > d(u)}
\end{align*}
$$
Which there exists a contradiction, suggesting the negation of our assumption must be true: "$dist(u) = d(u)$"

---
# Constraints: The Negative Weight Problem

Dijkstra’s Algorithm **does not work with negative edge weights**.
- **The Reason:** Dijkstra assumes that once a node is marked "Done," no future path can possibly be shorter.
- **The Failure:** A negative edge could "reduce" the cost of a path discovered later, breaking the greedy assumption. For graphs with negative weights, you must use the [[Bellman-Ford Algorithm]].

---
# Complexity Analysis

Total time will be: 

$$
	O(|V|(\text{decrease min}) + |E|(\text{decrease key}))
$$
> [!Important] 
> Different Implementations of [[Priority Queue]] will have different trade-offs between **costs of operations**. Might not be a single one that is optimal for all kings of graphs

## [[Priority Queue#Implementation Trade-offs]]

|       **Operation**       |      **Complexity**       |                           **Notes**                            |
|:-------------------------:|:-------------------------:|:--------------------------------------------------------------:|
|    **Initialization**     |        $O(\|V\|)$         | Setting $\text{dist}=\infty$, $\text{prev}=NULL$ for all nodes |
| **Priority Queue Insert** |      $O(\log \|E\|)$      |               Performed for each edge processed                |
| **Priority Queue Delete** |      $O(\log \|E\|)$      |                   Performed once per vertex                    |
|      **Total Time**       | **$O(\|E\| \log \|E\|)$** |         Dominant factor is *Priority Queue Operations*         |
|      **Total Space**      |  **$O(\|V\| + \|E\|)$**   |            Sorting Vertices and the adjacency list             |

> [!NOTE]
> 
> In very dense graphs, $|E|$ approaches $|V|^2$. In such cases, the $O(|E| \log |V|)$ complexity is nearly $O(|V|^2 \log |V|)$.

---
# Comparison of Shortest Path Algorithms

| **Algorithm** |      **Graph Type**      | **Guaranteed Shortest Path?** |
|:-------------:|:------------------------:|:-----------------------------:|
|    **BFS**    |        Unweighted        |      Yes (by edge count)      |
|    **DFS**    |           Any            |              No               |
| **Dijkstra**  | Weighted (Positive only) |   **Yes** (by total weight)   |
