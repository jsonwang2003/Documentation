---
tags:
  - algorithm
  - graph-traversal
  - DepthFirstSearch
aliases:
  - DFS
  - Depth First Search
description: Graph traversal algorithm that explores as far as possible along each branch before backtracking; O(V+E) time, memory-efficient on wide/shallow graphs.
---

> [!ABSTRACT] 
> **Depth-First Search (DFS)** is a graph traversal algorithm that explores as far as possible along each branch before backtracking. While [[Breadth First Search (BFS)|BFS]] explores "wide," DFS explores "deep," making it highly memory-efficient for specific graph structures.
> 
> - **Category:** Graph Traversal
> - **Input:** Graph $G=(V,E)$, optionally a source vertex $s$ (or run over all vertices for the full graph)
> - **Output:** Connected components, discovery/finish timestamps, DFS output forest
> - **Paradigm:** Backtracking, implemented via recursion (or an explicit stack)
> - **Typical use cases:** connected components, cycle detection, topological sort, timestamp-based structural analysis

---

# Core Logic: Go Deep, then Backtrack

DFS starts at a source node and dives into the first available neighbor it finds. It continues moving to unvisited neighbors until it hits a "dead end" (a node with no unvisited neighbors). At that point, it backtracks to the most recent node that still has unexplored paths.

> [!tip] Key Idea 
> Running DFS from a single source only visits vertices reachable from that source. To traverse the **entire graph** (including disconnected components), we loop over all vertices and launch a new exploration from any unvisited one.

---
# Pseudocode (Mid-Level Implementation)

### Full Graph DFS

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

### The Explore Procedure

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

### Pre/Post Timestamps

A global `clock` ticks upward each time a vertex is visited or finished. This gives every vertex two numbers that encode useful structural information about the graph.

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

> [!TIP] 
> The pre/post interval of a vertex `u` is either **entirely contained within** or **entirely disjoint from** the interval of any other vertex `v`. If `u`'s interval is inside `v`'s, then `u` is a descendant of `v` in the DFS tree.

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`visited`|Boolean array|Tracks which vertices have been discovered, to avoid re-visiting and infinite loops on cycles|
|`cc`|Integer counter|Labels the current connected component; incremented each time `explore` is launched from `DFS`'s outer loop|
|`clock`|Integer counter|Global "time" used to stamp `pre`/`post` values on each vertex|
|`prev`|Array (vertex → vertex)|Records the discovery edge for each vertex, building the DFS output forest|
|`pre`, `post`|Array (vertex → int)|Discovery / finish timestamps for each vertex|
|`s, u`|Vertex|Current vertex being explored / candidate neighbor|

## Helper Functions / Operations Used

- **`explore(G, v)`** — recursive procedure that fully visits `v` and everything reachable from it before returning; see [[Explore]]
- **`previsit(v)` / `postvisit(v)`** — timestamp hooks called on entry to and exit from a vertex; O(1) each
- **`component(s) = cc`** — labels every vertex with its connected component id
- **`prev(u) = s`** — records the discovery edge, building the **DFS output forest**

## DFS Output Forest
A **DFS output forest** is the forest structure given by the `prev` array after DFS has been performed on a graph. Each call to `explore` grows one tree in the forest. The edge `prev(u) = s` means DFS discovered `u` by traveling along the edge `(s, u)`. When the graph is connected, the output is a single **DFS tree**; when disconnected, it is a **forest** of one tree per component.

---
# Proof of Correctness

**Claim:** Upon termination of `DFS(G)`, every vertex has `visited = True`, every vertex is assigned exactly one `component` label, and vertices `u, v` share a `component` label if and only if they are connected in `G`.

**Loop Invariant (outer loop over vertices):** At the start of each iteration of the outer `for` loop in `DFS`, `visited` is `True` for exactly the vertices that have been fully explored so far, and these vertices form a union of complete connected components.

- **Initialization:** Before the first iteration, all vertices are unvisited and `cc = 0`, so the invariant holds vacuously.
- **Maintenance:** When the outer loop finds an unvisited vertex `v`, it increments `cc` and calls `explore(G, v)`. By the recursive structure of `explore`, this call terminates only after every vertex reachable from `v` has been marked visited and labeled with the current `cc` — i.e. `explore` never returns while an edge out of a visited-but-unfinished vertex still leads to an unvisited vertex, since each such edge is followed before `postvisit` executes. So the set of visited vertices grows by exactly one full connected component per outer iteration.
- **Termination:** The outer loop examines every vertex once, so it terminates after $|V|$ iterations, at which point every vertex is visited.

**Why every reachable vertex is found:** Within `explore(s)`, every edge $(s, u)$ is checked; if `u` is unvisited, `explore(u)` is called before `explore(s)` finishes (i.e. before `postvisit(s)`). By induction on the recursion, this guarantees every vertex reachable from `s` via any path is eventually visited during the same call to `explore(s)`, and thus assigned the same `component` value.

> [!Question] **Why components are correct:** 
> Two vertices get the same `cc` value only if one was discovered while exploring from the other (directly or transitively), which — by the previous paragraph — happens exactly when they are connected. Two vertices in different components can never share a `cc` value, since `cc` is incremented only when the outer loop starts a fresh, disconnected search.

---
# Time & Space Complexity Analysis

## General Case

DFS has time complexity $O(|V| + |E|)$: every vertex is previsited / postvisited exactly once ($O(V)$), and every edge is examined exactly once from each endpoint in the worst case ($O(E)$).

## Space — Implementation-Dependent Variations

Memory usage depends on the shape of the graph, since the recursion (or explicit stack) only needs to hold the current path from the root down to the active vertex:

|Shape|DFS Memory|BFS Memory|
|---|---|---|
|Wide, shallow|$O(\text{Height})$ → small|$O(\text{Width})$ → large|
|Narrow, deep|$O(\text{Height})$ → large|$O(\text{Width})$ → small|

- **Wide, shallow graph** — DFS uses very little memory (proportional to the height of the DFS tree)
- **Narrow, deep graph** — DFS must store the entire long path in the call stack (or explicit stack), which can be large

## Best / Worst / Average Case

- **Best / Worst / Average case:** All $O(V+E)$ — a full-graph `DFS` (via the outer loop) must visit every vertex and edge exactly once regardless of graph shape or vertex ordering, so there's no meaningfully better/worse case here unless searching for a specific target vertex with early exit.

---
# Drawbacks / Constraints

- **Preconditions:** Requires accessible adjacency info for each vertex ($G = (V, E)$); the "full graph" version requires being able to iterate over all vertices, not just those reachable from one source.
- **Fails / degrades when:** A narrow, deep graph causes the recursive call stack (or explicit stack) to grow very large — risk of stack overflow.
- **Important Drawback:** DFS is **not guaranteed to find the shortest path.** If DFS picks a deep branch that eventually leads to the destination, it will take it — even if a shorter path existed closer to the start.
- **Not suitable for:** Finding the shortest path on an unweighted graph — use [[Breadth First Search (BFS)]] instead.

---
# References / Links

- [[Breadth First Search (BFS)]]
- [[Explore]]