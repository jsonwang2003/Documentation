---
tags:
  - DepthFirstSearch
---
> [!ABSTRACT]
> **Depth-First Search (DFS)** is a graph traversal algorithm that explores as far as possible along each branch before backtracking. While [[Breadth First Search (BFS)|BFS]] explores "wide," DFS explores "deep," making it highly memory-efficient for specific graph structures.

---
# The Core Logic: Go Deep, then Backtrack

DFS starts at a source node and dives into the first available neighbor it finds. It continues moving to unvisited neighbors until it hits a "dead end" (a node with no unvisited neighbors). At that point, it backtracks to the most recent node that still has unexplored paths.

---
# Full Graph DFS

Running DFS from a single source only visits vertices reachable from that source. To traverse the **entire graph** (including disconnected components), we loop over all vertices and launch a new exploration from any unvisited one.

```pseudo
	\begin{algorithm}
	\caption{Depth First Search}
	\begin{algorithmic}
		\Procedure{DFS}{$G$}
			\State cc = 0
			\State clock = 1
			\For{each vertex $v$}
				\State visited($v$) = \False
            \EndFor
            \For{each vertex $v$}
	            \If{not visited($v$)}
		            \State cc += 1
		            \State explore($G, v$)
                \EndIf
            \EndFor
	    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

- `cc` increments each time we discover a new connected component
- `clock` is a global counter used to timestamp when each vertex is first and last visited (see Pre/Post Timestamps below)
## The Explore Procedure
`explore` is the recursive heart of DFS. It visits a single vertex and recursively follows all unvisited neighbors. For more information, visit [[Explore]]

```pseudo
	\begin{algorithm}
	\caption{Explore}
	\begin{algorithmic}
		\Procedure{explore}{$G = (V, E), s$}
			\State visited($s$) = \True
			\State previsit($s$)
			\State component($s$) = cc
			\For{each edge $(s, u)$}
				\If{not visited($u$)}
					\State prev($u$) = s
					\State explore($G, u$)
                \EndIf
            \EndFor
            \State postvisit($s$)
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

- `component(s) = cc` labels every vertex with its connected component
- `prev(u) = s` records the discovery edge, building the **DFS output forest**
## Pre/Post Timestamps

A global `clock` ticks upward each time a vertex is visited or finished. This gives every vertex two numbers that encode useful structural information about the graph.

### Previsit

```pseudo
	\begin{algorithm}
	\caption{Previsit}
	\begin{algorithmic}
		\Procedure{previsit}{$v$}
			\State pre($v$) = clock
			\State clock += 1
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

- `pre(v)` — the clock value when DFS **first arrives** at `v`
### Postvisit

```pseudo
	\begin{algorithm}
	\caption{Postvisit}
	\begin{algorithmic}
		\Procedure{postvisit}{$v$}
			\State post($v$) = clock
			\State clock += 1
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

- `post(v)` — the clock value when DFS **finishes** all of `v`'s neighbors and leaves

> [!TIP]
> The pre/post interval of a vertex `u` is either **entirely contained within** or **entirely disjoint from** the interval of any other vertex `v`. If `u`'s interval is inside `v`'s, then `u` is a descendant of `v` in the DFS tree.

---
# DFS Output Forest

> [!INFO] Definition
> A **DFS output forest** is the forest structure given by the `prev` array after DFS has been performed on a graph.

Each call to `explore` grows one tree in the forest, also known as DFS Output Tree. The edge `prev(u) = s` means DFS discovered `u` by traveling along the edge `(s, u)`. When the graph is connected, the output is a single **DFS tree**; when disconnected, it is a **forest** of one tree per component.

---
# Space Complexity

DFS has time complexity $O(|V| + |E|)$.

Memory usage depends on the shape of the graph:
- **Wide, shallow graph** — DFS uses very little memory (proportional to the height of the DFS tree)
- **Narrow, deep graph** — DFS must store the entire long path in the call stack

| Shape         | DFS Memory                 | BFS Memory                |
| ------------- | -------------------------- | ------------------------- |
| Wide, shallow | $O(\text{Height})$ → small | $O(\text{Width})$ → large |
| Narrow, deep  | $O(\text{Height})$ → large | $O(\text{Width})$ → small |

---
# Important Drawback

DFS is **not guaranteed to find the shortest path.** If DFS picks a deep branch that eventually leads to the destination, it will take it — even if a shorter path existed closer to the start. Use [[Breadth First Search (BFS)]] when shortest path on an unweighted graph is needed.