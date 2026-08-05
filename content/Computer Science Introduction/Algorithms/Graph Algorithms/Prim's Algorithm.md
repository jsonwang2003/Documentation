---
description: Greedy algorithm that builds a minimum spanning tree by repeatedly adding the cheapest edge connecting the tree to a new vertex.
tags:
  - algorithm
aliases:
  - Prim
  - Prim's
---

> [!abstract] 
> **What it is:** Prim's Algorithm is a [[Computer Science Introduction/Algorithms/Greedy Algorithms/index|Greedy Algorithm]] that builds a **[[Minimum Spanning Trees|Minimum Spanning Tree (MST)]]** for a connected, undirected, weighted graph by repeatedly growing a single tree one cheapest edge at a time.
> 
> - **Category:** Minimum Spanning Tree / Greedy Algorithm
> - **Input:** Undirected, connected graph $G$ with (not necessarily positive) edge weights $\ell$
> - **Output:** A list of edges forming a minimum spanning tree of $G$ (total weight is minimized among all spanning trees)
> - **Paradigm:** Greedy, priority-queue-driven frontier expansion
> - **Typical use cases:** network design (minimizing total cable/pipe/wire length), clustering, approximation algorithms that use an MST as a subroutine

---
# Core Logic (High-Level)

1. Put all vertices in $U$ (undiscovered).
2. Pick any vertex $s$ to start from.
3. Put $s$ in $X$ (the tree built so far).
4. Repeat until all vertices are in $X$:
    1. Find the minimum edge that has one vertex in $X$ and one vertex outside it
    2. Move that outside endpoint from $U$ into $X$.
    3. Add that edge to the output.

Naively, step 4.1 means scanning every edge crossing the boundary of $X$ each iteration. The `cost` array below avoids that: it caches the cheapest crossing edge per vertex, so step 4.1 becomes "pick the frontier vertex with the smallest `cost`" instead.

> [!tip] Key Idea 
> Instead of keeping track by looking at _edges_, put the **cost** information in the vertices themselves (`cost` array) — `cost(u)` is the cheapest edge weight seen so far connecting $u$ to the tree.
> 
> Update a vertex by putting the **lowest cost vertex** from $F$ into $X$, then re-check its neighbors: if a neighbor now has a cheaper bridge through this new tree vertex, update its `cost` (same "relax" idea as [[Dijkstra's Algorithm]], just comparing one edge weight instead of a cumulative path).

---
# Pseudocode (Mid-Level Implementation)

Similar to [[Dijkstra's Algorithm]], **Prim's** uses the value $cost(v)$ instead of $dist(v)$ — same loop, same priority queue, but the relaxation compares just the single edge weight $\ell(v,u)$ against `cost(u)`, not a cumulative path length.

```pseudo
	\begin{algorithm}
	\caption{ Prim's Algorithm }
	\begin{algorithmic}
		\Input $G$ undirected connecred graph with positive edge weights
		\Output $output$: a list of edges that describe a minimum spanning tree
		\Procedure{Prim's}{$G$}
			\State Pick a random vertex $s$
			\State Initialize $X$ = empty, $F = \{s\}$
			\State Initialize $cost(v) = \infty$ for all $v$
			\State Initialize $cost(s) = 0$
			\State Initialize $prev(s) = null$
			\State Initialize $output = $ empty
			\While{$F$ is not empty}
				\State Pick the $v \in F$ that has the $lowest$ $cost(v)$ value
				\For{each neighbor $u$ of $v$}
					\If{$cost(u) > \ell(v, u)$}
						\State Move $u$ to $F$
						\State Set $cost(u) = \ell(v, u)$
						\State Set $prev(u) = v$
                    \EndIf
                \EndFor
                \State Move $v$ from $F$ to $X$
                \State Move $(v, prev(v))$ into $output$
            \EndWhile
			\Return $output$
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`X`|Set|Vertices already included in the growing spanning tree|
|`F`|Set / [[Priority Queue]]|Frontier — vertices discovered (adjacent to `X`) but not yet added, keyed by `cost`|
|`U`|Set (implicit)|Vertices not yet discovered at all ($V - X - F$)|
|`cost`|Array (vertex → number)|Cheapest known edge weight connecting the vertex to the current tree $X$; starts at $\infty$ except $cost(s)=0$|
|`prev`|Array (vertex → vertex)|The tree-neighbor that offered the current best `cost`; used to reconstruct the actual MST edges|
|`output`|List of edges|Accumulates the edges $(v, prev(v))$ that make up the final MST|
|`v, u`|Vertex|Current minimum-cost vertex being finalized / candidate neighbor|

## Helper Functions / Operations Used

- **`ℓ(v, u)`** — the weight of the edge between `v` and `u`; O(1) lookup with an adjacency list/matrix
- **Pick $v \in F$ with lowest `cost(v)`** — a `deletemin` operation on the priority queue backing `F`
- **Relax a neighbor** — if $cost(u) > \ell(v,u)$, update `cost(u)` and `prev(u)`, and move/re-prioritize `u` in `F` (a `decreasekey`, or an insert if `u` was previously in `U`)

> [!note] Low-Level Implementation 
> The low-level implementation is essentially [[Dijkstra's Algorithm#Optional Low-Level Implementation|Dijkstra's low-level implementation]] with `dist` renamed to `cost` and the relaxation condition changed from $dist(u) > dist(v) + \ell(v,u)$ to $cost(u) > \ell(v,u)$ — i.e. compare against the single edge weight, not the cumulative path weight. Same `makequeue` / `deletemin` / `decreasekey` primitives apply, and the same array-vs-binary-heap trade-off discussion carries over directly.

---

# Proof of Correctness

**Claim:** Upon termination, the edges in `output` form a minimum spanning tree of $G$.

**Loop Invariant (Cut Property):** At the start of each iteration of the while loop, the edges already added to `output` form a subset of _some_ minimum spanning tree of $G$ — equivalently, $X$ can always be extended to a full MST using only edges not yet ruled out.

- **Initialization:** Before the first iteration, `output` is empty and $X = {s}$. The empty edge set is trivially a subset of any MST, so the invariant holds.
- **Maintenance:** Suppose the invariant holds — $X$'s edges so far are consistent with some MST $T$. Consider the cut $(X, V-X)$. By the [[Cut Property]], the minimum-weight edge crossing this cut is guaranteed to be in _some_ MST. Prim's always selects exactly this edge next — the vertex $v \in F$ with lowest `cost(v)` is, by construction, the endpoint of the cheapest edge crossing the cut $(X, V-X)$ (since `cost(v)` was set to the weight of the cheapest edge from $X$ to $v$ during relaxation). So adding $(v, prev(v))$ to `output` keeps the invariant true — either $T$ already contains this edge, or swapping it into $T$ (removing whatever edge $T$ used to connect $v$) produces another MST of equal or lower weight, since the swapped-in edge is minimum-weight across the cut. See [[Cut Property]] for the full proof of why this swap argument works.
- **Termination:** Each iteration moves exactly one vertex from $F$ (or $U$, via relaxation) into $X$, so after $|V|-1$ iterations all vertices are in $X$ and `output` contains exactly $|V|-1$ edges — the size of a spanning tree on a connected graph. The while loop then ends because $F$ becomes empty.

**Why it doesn't miss/duplicate vertices:** Each vertex is moved into $X$ exactly once (guarded implicitly by only picking $v \in F$, and $F$ only ever contains vertices not already in $X$); since $G$ is connected, every vertex is eventually discovered as a neighbor of some vertex already in $X$, so no vertex is permanently stuck in $U$.

**Conclusion:** By induction, when the loop terminates ($X = V$), `output` is a spanning tree consistent with an MST at every step, and since it spans all of $G$ with exactly $|V|-1$ minimum-cut-respecting edges, `output` **is** a minimum spanning tree. $\blacksquare$

---

# Time & Space Complexity Analysis

Basically has the same runtime as Dijkstra's, using the same $O(|V|(deletemin) + |E|(decreasekey))$ formula — the one difference is that Prim's requires a **connected input graph**, so $|E| = \Omega (|V|)$. That's why the $|V|$ term drops out of the totals below: it's always dominated by $|E|$, so the runtime is stated purely in terms of $|E|$ and $|V|$ together rather than as a sum of two competing terms.

- **Binary Heap** — $O(|E| \log|V|)$
- **Array** — $O(|V|^{2})$

## General Case

|       | Complexity                                               | Notes                                                                                                                                |
| ----- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Time  | $O(\|V\|(\text{deletemin}) + \|E\|(\text{decreasekey}))$ | Every vertex is finalized once (deletemin), every edge is examined at most twice (once from each endpoint) as a relaxation candidate |
| Space | $O(\|V\|)$                                               | `cost`, `prev` arrays + priority queue holding the frontier, plus `output` of size $                                                 |

## Implementation-Dependent Variations

| Data Structure Choice         | Impact on Time                                                                    | Impact on Space                                                               | Notes                                                              |
| ----------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Array as Priority Queue       | $O(\|V\|^2)$ total — `deletemin` is $O(\|V\|)$, `decreasekey` is $O(1)$           | $O(\|V\|)$                                                                    | Better for **dense** graphs where $\|E\| = \Theta(\|V\|^2)$        |
| Binary Heap as Priority Queue | $O(\|E\|\log\|V\|)$ total — `deletemin` and `decreasekey` are both $O(\log\|V\|)$ | $O(\|V\|)$ + supplemental "address book" array to locate vertices in the heap | Better for **sparse** graphs where $\|E\| = \Theta(\|V\|)$; note $ |
| Adjacency list vs matrix      | $O(\|V\|+\|E\|)$ vs $O(\|V\|^2)$ for scanning neighbors                           | $O(\|V\|+\|E\|)$ vs $O(\|V\|^2)$                                              | Matrix only worth it on already-dense graphs                       |

## Best / Worst / Average Case

- **Best case:** Still $O(|V|(\text{deletemin}) + |E|(\text{decreasekey}))$ — Prim's has no early-exit condition; it must process every vertex and consider every edge to guarantee the minimum spanning tree, regardless of graph shape.
- **Worst case:** Same order — dense graph maximizes both the number of `decreasekey` calls and, if using an array-backed PQ, the cost of each `deletemin`.
- **Average case:** Same asymptotic order; Prim's has no probabilistic behavior to average over.

---

# Drawbacks / Constraints

- **Preconditions:** $G$ must be **connected** and **undirected** — Prim's does not handle directed graphs (there is no directed-graph analogue of a spanning tree in the same sense) and will not produce a spanning structure if $G$ is disconnected (some vertices would remain in $U$ forever, stuck at $cost = \infty$).
- **Unlike Dijkstra's, negative edge weights are fine.** Prim's only ever compares single-edge weights ($cost(u) > \ell(v,u)$), never cumulative path weights, so the greedy cut-property argument still holds even with negative weights — there's no analogue of Dijkstra's "a later negative edge could undercut an already-finalized path" failure mode.
- **Not suitable for:** Finding shortest paths between vertices — an MST minimizes total tree weight, not pairwise path weight; use [[Dijkstra's Algorithm]] (non-negative weights) or Bellman-Ford (negative weights allowed) for shortest paths instead.
- **MST is not unique in general.** If multiple edges tie for minimum weight across a cut, different tie-breaking choices can produce different (but equally minimal-weight) spanning trees.
- **Alternatives to consider:** [[Kruskal's Algorithm]] — a different greedy MST algorithm that sorts all edges globally and uses union-find, often preferable for very sparse graphs or when edges are already sorted/streamed.

---
# References / Links

- [[Dijkstra's Algorithm]]
- [[Kruskal's Algorithm]]
- [[Priority Queue]]
- [[Cut Property]]