---
tags:
  - BreadthFirstSearch
  - algorithm
  - graph-traversal
description: Graph traversal algorithm that finds shortest paths in unweighted graphs by exploring layer-by-layer via a FIFO queue; O(V+E) time.
aliases:
  - BFS
---

> [!ABSTRACT] 
> **Breadth-First Search (BFS)** is the primary algorithm for finding the **shortest path in an unweighted graph**. It explores a graph layer-by-layer, ensuring that it visits every node at distance $k$ before moving on to any node at distance $k+1$. While [[Depth First Search (DFS)|DFS]] explores "deep," BFS explores "wide."
> 
> - **Category:** Graph Traversal
> - **Input:** Graph $G=(V,E)$, source vertex $s$
> - **Output:** `dist(v)` for every vertex — the shortest number of edges from $s$ to $v$
> - **Paradigm:** Iterative, FIFO-queue-based frontier expansion
> - **Typical use cases:** shortest path on unweighted graphs, level-order traversal, "minimum number of hops" problems

---
# Core Logic: Layer-by-Layer Exploration

The intuition behind BFS is similar to a ripple in a pond. Starting from a source node, the search expands outward in concentric circles:

1. **Level 0:** The starting node $s$.
2. **Level 1:** All immediate neighbors of $s$.
3. **Level 2:** All neighbors of Level 1 nodes that haven't been visited yet.

> [!tip] Key Idea 
> BFS is structured as a **single iterative procedure** rather than a recursive one because the level-by-level expansion follows a **FIFO** order, which maps naturally to a [[Queues|queue]] rather than a call stack. The outer loop continuously dequeues the earliest-discovered vertex and enqueues its unvisited neighbors, ensuring every vertex at distance $d$ is processed before any vertex at distance $d+1$. This ordering is what guarantees **shortest paths** on **unweighted graphs**.

> [!Question] Why doesn't an "early out" improve worst-case time complexity? 
> Including an "early out" doesn't change the **worst-case** complexity because, in the worst case, the destination node is the very last node visited (or is unreachable), forcing the algorithm to traverse the entire graph anyway.

---
# Pseudocode (Mid-Level Implementation)

### Full Graph BFS

```pseudo
	\begin{algorithm}
	\caption{Breadth First Search}
	\begin{algorithmic}
		\Procedure{BFS}{$G, s$}
			\For{each vertex $u \in V$}
				\State dist($u$) = $\infty$
            \EndFor			
			\State dist($s$) = $0$
			\State $Q = [s]$
			\Comment{queue that just containing $s$}
			\While{$Q$ is not empty}
				\State $u$ = dequeue($Q$)
				\ForAll{edges $(u, v) \in E$}
					\If{dist($v$) = $\infty$}
						\State enqueue($Q, v$)
						\State dist($v$) = dist($u$) + 1
                    \EndIf
                \EndFor
            \EndWhile
	    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`Q`|[[Queues\|Queue]] (FIFO)|Holds the current **frontier** of discovered-but-unprocessed vertices|
|`dist`|Array (vertex → int)|Distance from the starting vertex $s$ to the given vertex; $\infty$ until discovered|
|`u, v`|Vertex|Current vertex being dequeued / candidate neighbor|

## Helper Functions / Operations Used

- **`dequeue(Q)`** — pops and returns the **earliest inserted** vertex; O(1) with an array/linked-list backed queue
- **`enqueue(Q, v)`** — inserts vertex $v$ into the queue $Q$; O(1) amortized
- **`dist(v)`** — the distance between the starting vertex $s$ and $v$; each vertex's `dist` is set **at most once**, which is what keeps the algorithm at $O(V+E)$

---
# Proof of Correctness

For each vertex $v$, we want to show that $dist(v)$ is the minimum distance of all paths from $s$ to $v$. Prove by induction on distance $d$.

**Claim:** For each distance value $d = 0, 1, 2, \dots$, there is a moment in the algorithm when:

1. All vertices at distance $\leq d$ from $s$ have their distance values correctly set.
2. All other vertices (distance $> d$ from $s$) have distances set to $\infty$.
3. The queue contains exactly the nodes at distance $d$.

**Base Case ($d = 0$):**

1. $dist(s) = 0$ is the correct distance value (the only vertex at distance $0$ from $s$ is $s$ itself).
2. All other vertices have distances set to $\infty$ (initialization step).
3. The queue contains only $s$, which is the only vertex at distance $0$.

**Inductive Step:** Let $k \geq 0$ be arbitrary. Assume the claim holds for $d = k$ — all vertices at distance $\leq k$ have been set correctly, and the queue contains exactly the vertices at distance $k$.

Suppose $v$ is the next vertex popped from the queue (so $dist(v) = k$), and let $u$ be a neighbor of $v$:

- If $dist(u) \neq \infty$, then by the inductive hypothesis $dist(u)$ has already been set correctly, and it is **not** updated again.
- If $dist(u) = \infty$, then $dist(u) \gets dist(v) + 1 = k+1$. This is correct: since $u$ was unreachable before going through $v$, and $dist(v) = k$ is the minimum distance from $s$ to $v$, the minimum distance from $s$ to $u$ must be $k+1$.

Therefore, after this step:

1. All new vertices added to the queue have distance $k+1$ and are set correctly.
2. All vertices at distance $k+1$ have been added to the queue.
3. The queue contains exactly the nodes at distance $k+1$.

This completes the induction — every vertex's `dist` value equals its true shortest distance from $s$.

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(V + E)$|Each vertex enters the queue **at most one time**, and each edge is examined at most twice (once from each endpoint)|
|Space|$O(V)$|`dist` array + queue holding the current frontier in the worst case|

> [!Info] Notice 
> In BFS, each vertex enters the queue at most one time — this is the assumption used when calculating the runtime for [[Computer Science/Algorithms/Graph Algorithms/index#Graph Search|Graph Search]] in general, and it's what keeps `dist(v)` from being set more than once per vertex.

## Implementation-Dependent Variations

Memory usage depends on the shape of the graph, since BFS must hold the entire current frontier in the queue at once:

|Shape|DFS Memory|BFS Memory|
|---|---|---|
|Wide, shallow|$O(\text{Height})$ → small|$O(\text{Width})$ → large|
|Narrow, deep|$O(\text{Height})$ → large|$O(\text{Width})$ → small|

- **Wide, shallow graph** — BFS's frontier (queue) can balloon to hold most of the graph at once → large memory
- **Narrow, deep graph** — BFS only ever holds a thin frontier → small memory (DFS is the one that struggles here instead, storing the entire long path)

## Best / Worst / Average Case

- **Best / Worst / Average case:** All $O(V+E)$ — a full BFS from $s$ must enqueue and dequeue every reachable vertex exactly once and scan every incident edge, so there's no meaningfully better/worse case unless searching for a specific target vertex with early exit (which, per the note above, doesn't change the worst case).

---

# Drawbacks / Constraints

- **Preconditions:** Requires accessible adjacency info for each vertex; assumes a **FIFO** queue implementation to preserve level-order correctness.
- **Only works for shortest distance on graphs where each edge has equal weight.** BFS's correctness proof relies on every edge contributing exactly $+1$ to distance.
- **Not suitable for:** Weighted graphs. One can attempt to force BFS to work by forming $G'$ — adding $w_e - 1$ new vertices between $u$ and $v$ for every edge $e = (u,v)$ — and running BFS on $G'$, but this is **impractical** when edge weights are large integers (the graph blows up in size).
- **Alternatives to consider:** Use [[Dijkstra's Algorithm]] for weighted graphs with non-negative weights; Bellman-Ford if negative weights are present.

---
# References / Links

- [[Depth First Search (DFS)]]
- [[Dijkstra's Algorithm]]
- [[Queues]]