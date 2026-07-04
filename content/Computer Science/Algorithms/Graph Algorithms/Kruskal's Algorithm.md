---
description: Greedy MST algorithm that sorts all edges by weight and adds each one that doesn't create a cycle, using a disjoint-set structure to detect cycles.
tags:
  - algorithm
aliases:
  - Kruskal's
  - Kruskal's MST
---
> [!abstract]
> Kruskal's Algorithm is a [[Computer Science/Algorithms/Greedy Algorithms/index|Greedy Algorithm]] that builds a **Minimum Spanning Tree (MST)** by sorting all edges by weight and adding each one that doesn't create a cycle, using a disjoint-set (union-find) structure to check for cycles in $O(\alpha(V))$ per edge.
> 
> - **Category:** Minimum Spanning Tree / Greedy Algorithm
> - **Input:** Undirected, connected graph $G$ with edge weights $w$
> - **Output:** A set of edges $X$ forming a minimum spanning tree of $G$
> - **Paradigm:** Greedy, global edge sort + Disjoint Sets (Union-Find)
> - **Typical use cases:** network design on sparse graphs, MST as a subroutine (e.g. clustering, approximation algorithms), situations where the edge list is already sorted or streamed

---
# Core Logic (High-Level)

1. Start with a graph with only the vertices (no edges).
2. Repeatedly add the next lightest edge that does not form a cycle.

> [!tip] Key Idea 
> Sort every edge in the graph once, globally, by weight. Then walk the sorted list and greedily take an edge whenever its two endpoints are still in different components — this is checked with `find`, not by searching the graph. Skipping an edge that would form a cycle is safe because both endpoints are already connected by cheaper edges, so the skipped edge can never be the minimum crossing edge for any cut (see [[Cut Property]]).

---
# Pseudocode (Mid-Level Implementation)

```pseudo
	\begin{algorithm}
	\caption{ Kruskal }
	\begin{algorithmic}
		\INPUT Undirected graph $G$ with edge weights $w$
		\OUTPUT A set of edges $X$ that defines a minimum spanning tree
		\PROCEDURE{ Kruskal }{$G, w$}
			\ForAll{$v \in V$}
				\State $Makeset(v)$
            \EndFor
            \State $X = \{\}$
            \State Sort the set of edges $E$ in increasing order by weight
            \ForAll{edges $(u,v) \in E$ until $|X| = |V| - 1$}
	            \If{$find(u) \neq find(v)$}
		            \State $add(u,v)$ to $X$
		            \State $union(u,v)$
                \EndIf
            \EndFor
		\ENDPROCEDURE
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`X`|Set of edges|The growing MST — edges accepted so far|
|`E` (sorted)|Sorted list of edges|The full edge list, sorted once up front by weight, so the greedy scan just walks it left to right|
|`π`, `rank`|Disjoint Sets (Union-Find)|Tracks which component each vertex currently belongs to; used to detect cycles via `find`|
|`u, v`|Vertex|The two endpoints of the edge currently being considered|

Kruskal's cycle check relies on the [[Disjoint Sets & Up-Trees]] data structure — see that note for the full operations, proofs, path compression, and amortized analysis. The short version used here:

- **`Makeset(v)`** — puts `v` into its own singleton set
- **`Find(u)`** — returns the name (root) of the set containing `u`
- **`Union(u,v)`** — merges the sets containing `u` and `v`

## Helper Functions / Operations Used

- **`Makeset(v)`** ($O(1)$) — initializes `v` as its own singleton set/component
- **`find(u)`** ($O(k) = O(\log u)$) — walks parent pointers up to the root to identify `u`'s current component; used as the cycle check (`find(u) ≠ find(v)` means adding edge `(u,v)` can't create a cycle)
- **`union(u,v)`** ($O(find) = O(\log u)$) — merges the two components containing `u` and `v`; see [[Disjoint Sets & Up-Trees#Union Variants|Union Variants]] for the by-rank vs. by-size tie-breaking choice that keeps this fast

> [!note] Low-Level Implementation 
> Path compression on `find` is what pushes `find`/`union` down to amortized $O(\alpha(V))$ each — see [[Disjoint Sets & Up-Trees#Optimizing Find Path Compression|Path Compression]] and [[Disjoint Sets & Up-Trees#Amortized Cost Analysis|Amortized Cost Analysis]] for the implementation and the proof of why.

---
# Proof of Correctness

**Claim:** Upon termination, $X$ is a minimum spanning tree of $G$.

**Loop Invariant:** At the start of each iteration, $X$ is a subset of some minimum spanning tree of $G$.

- **Initialization:** $X$ starts empty, which is trivially a subset of any MST.
- **Maintenance:** Consider an edge $(u,v)$ with $find(u) \neq find(v)$. Let $C$ be $u$'s current component (as tracked by the disjoint-set structure). The cut $(C, V-C)$ separates $u$ from $v$, and $(u,v)$ crosses it. Because edges are processed in increasing weight order and no earlier (cheaper) edge crossing this cut has been added — otherwise `find(u)` and `find(v)` would already agree — $(u,v)$ is the cheapest edge crossing $(C, V-C)$ seen so far. By the [[Cut Property]], this edge belongs to some MST, so adding it keeps $X$ a subset of some MST.
- **Termination:** Each accepted edge merges two components into one via `union`, reducing the number of components by exactly one. Starting from $|V|$ singleton components, the loop stops once $|X| = |V|-1$ edges have been added, at which point (for a connected graph) all vertices are in a single component.

**Why it doesn't create cycles or miss vertices:** The `find(u) ≠ find(v)` check rejects any edge that would connect two vertices already in the same component — exactly the definition of a cycle-forming edge — so $X$ stays a forest at every step. Since $G$ is connected and every edge is eventually considered, the forest ends up spanning all of $V$.

---
	 
# Time & Space Complexity Analysis

$$
	\begin{align*}
	&|V| makeset + 2 |E|find + (|V| - 1)union + sort(|E|)\\
	=& |V|O(1) + 2|E|O(\log |V|) + (|V|-1)O(\log |V|) + O(|E|\log|E|)\\
	=& O(V + 2|E|\log|V| + |V|\log|V| + |E|\log|E|)\\
	=& \boxed{O(|E|\log |V|)}
	\end{align*}
$$
## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(\|E\| \log \|E\|)$|Dominated by sorting the edge list; the union-find operations that follow add only $O(\|E\| \alpha(\|V\|))$, which is effectively linear|
|Space|$O(V + E)$|Edge list plus the `π`/`rank` arrays for the disjoint-set structure|

Since $|E| \leq |V|^2$, $\log|E| = O(\log|V|)$, so this is often written as $O(|E|\log|V|)$.

## Implementation-Dependent Variations

|Data Structure Choice|Impact on Time|Impact on Space|Notes|
|---|---|---|---|
|Comparison sort (e.g. mergesort) for edges|$O(\|E\|\log\|E\|)$|$O(E)$|General-purpose; this is what dominates the overall runtime|
|Bucket/radix sort for edges|$O(\|E\|)$|$O(E)$|Only usable when weights are small bounded integers — drops the runtime to near-linear, dominated instead by the union-find term|
|Union-Find with union by rank only|$O(\log\|V\|)$ per `find`/`union`|$O(V)$|Still fine, but slightly worse than adding path compression|
|Union-Find with union by rank + path compression|$O(\alpha(\|V\|))$ amortized per `find`/`union`|$O(V)$|Effectively constant time in practice; standard choice|
|Union-Find with no optimization (plain linked structure)|$O(V)$ worst case per `find`|$O(V)$|Avoid — makes the union-find term dominate over the sort|

## Best / Worst / Average Case

- **Best / Worst / Average case:** All $O(|E|\log|E|)$ — the edge sort has to happen regardless of graph shape, and it dominates the union-find work either way. There's a mild early exit (`until |X| = |V|-1`) once the tree is complete, but it doesn't change the worst-case bound since the sort itself already touched every edge.

---

# Drawbacks / Constraints

- **Preconditions:** $G$ must be connected for the output to be a single spanning tree (otherwise the loop ends with $|X| < |V|-1$, having produced a _minimum spanning forest_ instead); requires the full edge list up front to sort it.
- **Like Prim's, negative edge weights are fine** — Kruskal's also only ever compares individual edge weights, never cumulative path weights, so the greedy cut-property argument still holds.
- **Not suitable for:** Very dense graphs, where the $O(|E|\log|E|)$ sort becomes expensive relative to Prim's array-based $O(|V|^2)$ — use [[Prim's Algorithm]] instead when $|E| = \Theta(|V|^2)$.
- **Alternatives to consider:** [[Prim's Algorithm]] for dense graphs or when growing a single connected tree incrementally is more natural (e.g. streaming vertices rather than a static edge list).

---

# References / Links

- [[Prim's Algorithm]]
- [[Cut Property]]
- [[Disjoint Sets & Up-Trees]]