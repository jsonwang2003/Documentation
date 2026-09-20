---
title: Graph Algorithms
description: Foundational graph definitions, representations, and the generic Graph Search algorithm.
tags:
  - CS/index
  - CS/algorithms
aliases:
  - Graph Algorithms
---

> [!abstract] Overview 
> Foundational definitions and the generic search procedure underlying every graph traversal algorithm in this vault. DFS, BFS, and Explore are each a specific choice of data structure for the frontier $F$ in the Graph Search algorithm.

---
## Foundational Concepts

> [!info] Definition: Graphs
> A graph is specified by nodes and edges: $G = (V, E)$ where $V$ are vertices and $E$ are edges. A directed edge $(x, y)$ is an edge from $x$ to $y$.

### Graph Representations:
- **Adjacency Matrix:** A $V \times V$ matrix $A$ where $A(i, j) = 1$ if $(i, j) \in E$. Uses $O(V^2)$ space but checks edges in $O(1)$ time.
- **Adjacency List:** For each node, a list of outgoing edges. Uses $O(E)$ space and easily iterates neighbors, but checks edges in $O(V)$ time.

![[Pasted image 20260404010858.png|559]]

---
## Core Concept / Shared Building Block

### Graph Search
**Graph Search** is a core foundational outline of most graph algorithms, serving as a template for DFS, BFS, Dijkstra's, etc. At each point, vertices are partitioned into $X$ (explored), $F$ (frontier), and $U$ (unreached).

Choosing $F$ = stack gives [[3. Depth First Search (DFS)|Depth First Search]] / [[1. Explore|Explore]]; choosing $F$ = queue gives [[6. Breadth First Search (BFS)|Breadth First Search]].

```pseudo
\begin{algorithm}
\caption{Graph Search}
\begin{algorithmic}
    \Procedure{GraphSearch}{$G, s$}
        \State $X$ = empty, $F$ = $\{ s \}$, $U = V - F$
        \While{$F$ is not empty}
            \State Pick $w$ in $F$
            \ForAll{$(w, y) \in E$}
                \If{$y \not\in X$ or $F$}
                    \State move $y$ from $U$ to $F$
                \EndIf
            \EndFor
            \State move $w$ from $F$ to $X$
        \EndWhile
        \Return X
    \EndProcedure
\end{algorithmic}
\end{algorithm}
```

> [!example] Runtime Analysis:
> Since each $v$ is added to $F$ at most once, each $v$ is also deleted from $F$ at most once: 
> 
> $$
>	O(\sum_{v\in V}(1 + (out)deg(v))) = O(\|V\| + \|E\|)
> $$


---
## Notes in This Section

### Graph Algorithm Notes

| Note                                                                            | Description                                                                                                                                                                              |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[1. Explore\|Explore]]                                                         | The low-level, single-component recursive implementation of Graph Search with $F$ = stack.                                                                                     |
| [[3. Depth First Search (DFS)\|Depth First Search]]                             | Graph Search with $F$ = stack (via recursion), run over all vertices to cover disconnected components.                                                                         |
| [[4. Strongly Connected Components (SCC)\|Strongly Connected Components (SCC)]] | Two-pass DFS that decomposes a directed graph into its strongly connected components.                                                                                          |
| [[6. Breadth First Search (BFS)\|Breadth First Search]]                         | Graph Search with $F$ = queue; guarantees shortest paths on unweighted graphs.                                                                                                 |
| [[7. Dijkstra's Algorithm\|Dijkstra's Algorithm]]                               | Graph Search with $F$ = priority queue keyed by cumulative distance; handles non-negative weighted graphs.                                                                     |
| [[8. Kruskal's Algorithm\|Kruskal's Algorithm]]                                 | Sorts all edges globally and uses [[Disjoint Sets & Up-Trees]] instead of a frontier-based search.                                                                             |
| [[9. Prim's Algorithm\|Prim's Algorithm]]                                       | Same loop shape as Dijkstra's, but $F$ is keyed by single-edge cost to build a minimum spanning tree.                                                                          |
| [[11. A-Star Search\|A* Search]]                                                | Graph Search with $F$ = priority queue keyed by cumulative distance $+$ distance away from goal; handles weighted graphs (non-negative) and is more efficient than Dijkstra's. |
| [[Graph Algorithms Summary]]                                                    | A quick, notesheet-style overview of Graph Algorithms.                                                                                                                         |

### Supplementary Concepts & Proofs
| Note                                                          | Description                                                                                           |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [[2. Edge Types\|Edge Types]]                                 | Definitions of Tree, Back, and Cross edges in graph traversal.                                        |
| [[5. Graph Algorithm Approaches\|Graph Algorithm Approaches]] | Two general strategies for deriving a new algorithm from an existing one: Modification and Reduction. |
| [[10. Cut Property\|Cut Property]]                            | MST lemma proving the lightest edge crossing any cut can always be added to an MST.                   |

---
## Related Categories

- [[Minimum Spanning Trees]]
- [[Levels of Algorithm Design]]

