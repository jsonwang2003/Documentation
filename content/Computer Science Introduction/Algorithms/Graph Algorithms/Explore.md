---
description: Low-level recursive implementation of Graph Search using a stack (FILO); the core recursive routine underlying DFS.
tags:
  - algorithm
  - graph-traversal
  - DepthFirstSearch
aliases:
  - explore
---
> [!Abstract] 
> **Explore** is the same algorithm as [[Computer Science Introduction/Algorithms/Graph Algorithms/index#Graph Search|Graph Search]] where $F$ is a [[Computer Science Introduction/Data Structures/Introductory Data Structures/Stack|Stack]]. The main difference is that **Graph Search** is a [[Levels of Algorithm Design#Mid Level Design|Mid-Level Implementation]] while **Explore** is a [[Levels of Algorithm Design#Low Level Design|Low-Level Implementation]] of **Graph Search**.
> 
> - **Category:** Graph Traversal (single-component reachability)
> - **Input:** Graph $G=(V,E)$, source vertex $s$
> - **Output:** `visited` array; optionally `prev`, `pre`, `post` for path/timestamp tracking
> - **Paradigm:** Recursion (implicit stack, FILO order)
> - **Typical use cases:** the recursive building block used inside [[Depth First Search (DFS)]]; single-source reachability queries

---

# Core Logic: Recursive Stack-Based Search

> [!Note] Explore vs. DFS 
> The explore algorithm uses the same underlying logic as [[Depth First Search (DFS)]] and undergoes a **FILO** scheme. The difference lies in **Explore** will not visit any nodes that are disconnected from the starting vertex; whereas **DFS** continues to search in disconnected vertices after fully exploring the current sub-graph.

1. Mark the current vertex `s` as visited.
2. For each edge `(s, u)` out of `s`, if `u` is unvisited, recursively call `explore` on `u`.
3. Once every neighbor has been checked (and their subtrees fully explored), the call returns — this is the "leaving the stack" moment.

> [!tip] Key Idea 
> `explore` **is** the recursion, and the call stack **is** the stack `F` from Graph Search — there's no separate data structure to manage, the language runtime does it for you.

---
# Pseudocode (Mid-Level Implementation)

### Graph Search (general form, for comparison)

```pseudo
	\begin{algorithm}
	\caption{Graph Search}
	\begin{algorithmic}
		\Procedure{GraphSearch}{$G, s$}
			\State $X$ = empty, $F$ = $\{ s \}$, $U = V - F$
			\While{$F$ is not empty}
				\State Pick $v$ in $F$
				\ForAll{neighbors $u$ of $v$}
					\If{$u \not\in X$ or $F$}
						\State move $u$ from $U$ to $F$
                    \EndIf
                \EndFor
                \State move $v$ from $F$ to $X$
            \EndWhile
            \Return X
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Explore (low-level implementation of Graph Search, $F$ = stack)

```pseudo
	\begin{algorithm}
	\caption{Explore}
	\begin{algorithmic}
		\Procedure{explore}{$G = (V, E), s$}
		\State visited($s$) = \True
		\For{each edge $(s, u)$}
			\If{not visited($u$)}
				\State explore($G, u$)
            \EndIf
        \EndFor
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

The output is an array `visited` such that `visited(u)` is true if and only if **$u$ is reachable from $s$**, for all vertices $u \in V$.

> [!Important] 
> This implementation only gives information about whether there is a path from $s$ to another vertex. However, sometimes it is helpful to know what those paths are.

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`visited`|Boolean array|Marks whether a vertex has been discovered, preventing infinite loops on cycles|
|(implicit)|Call stack|Plays the role of `F` from Graph Search — FILO order gives depth-first behavior|
|`prev`|Array (vertex → vertex)|_(path-tracking version)_ Records the discovery edge for each vertex|
|`pre`, `post`|Array (vertex → int)|_(path-tracking version)_ Timestamps for when a vertex enters/leaves the stack|
|`clock`|Integer counter|_(path-tracking version)_ Global tick used to generate `pre`/`post` values|

## Helper Functions / Operations Used

- **`explore(G, u)`** — recursive call; the "push" onto the stack happens implicitly via the function call
- **`previsit(v)`** — called when `v` enters the stack; sets `pre(v) = clock` and increments `clock`
- **`postvisit(v)`** — called when `v` leaves the stack; sets `post(v) = clock` and increments `clock`

## DFS Output Tree: Keep Track of Paths

We can include another array of information. Set `prev(u)` to be the **"parent"** of `u` in the DFS output tree. By also tracking when a node **enters** (`pre`) and **leaves** (`post`) the stack, we can know the connected structure of _directed graphs_.

```pseudo
	\begin{algorithm}
	\caption{Explore with Path Record}
	\begin{algorithmic}
	\Procedure{explore}{$G = (V, E), s$}
		\State visited($s$) = \True
		\State previsit($s$)
		\For{each edge $(s, u)$}
			\If{not visited($u$)}
				\State prev($u$) = $s$
				\State explore($G, u$)
            \EndIf
        \EndFor
        \State postvisit($s$)
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

```pseudo
	\begin{algorithm}
	\caption{previsit}
	\begin{algorithmic}
		\Procedure{previsit}{$v$} \Comment{when vertex $v$ enters the stack}
			\State pre($v$) = clock
			\State clock++
	    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

```pseudo
	\begin{algorithm}
	\caption{postvisit}
	\begin{algorithmic}
		\Procedure{postvisit}{$v$}
			\Comment{when vertex $v$ leaves the stack}
			\State post($v$) = clock
			\State clock++
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!INFO] DFS Output Tree 
> A **DFS output tree** is the tree structure formed by the edges in the `prev` array after **Explore** has been performed on a graph. When `explore` discovers an unvisited neighbor `u` from vertex `s`, it records `prev(u) = s` — "I reached `u` by traveling the edge `(s, u)`." Collecting all such discovery edges across the entire Explore gives the output tree — it contains only the edges Explore actually used to find new vertices. Any edge that led to an already-visited vertex is discarded.
> 
> - **Connected graph** → a single **DFS output tree**, rooted at the first vertex explored
> - **Disconnected graph** → a **[[Depth First Search (DFS)#DFS Output Forest|DFS Output Forest]]**, one tree per connected component, each rooted at the vertex that triggered `cc++`
> 
> **Example:** If DFS visits A → B → D → C, the output tree contains edges A–B, B–D, and D–C. Any edge encountered along the way that pointed to an already-visited vertex is not included.

---
# Proof of Correctness

**Claim:** Upon termination of `explore(G, s)`, `visited(u) = True` if and only if `u` is reachable from `s` (i.e. `visited` correctly computes the connected component containing `s`).

**($\Rightarrow$) Nothing unreachable is ever marked visited:** `explore` only ever recurses along an actual edge `(s, u) \in E`. By induction on the recursion depth, every call `explore(G, v)` is only ever reached by following a chain of real edges starting from `s`, so every vertex marked visited is reachable from `s`.

**($\Leftarrow$) Everything reachable is eventually marked visited:** Suppose for contradiction some vertex `u` reachable from `s` is never visited. Take a shortest path $s = v_0, v_1, \dots, v_k = u$. Let $v_i$ be the first vertex on this path that is never visited. Since $v_{i-1}$ _is_ visited (by minimality of $i$), and `explore(G, v_{i-1})` iterates over **every** edge $(v_{i-1}, v_i)$ out of $v_{i-1}$, it must check whether $v_i$ is visited — and since it isn't, `explore` recurses into it, marking it visited. This contradicts the assumption that $v_i$ is never visited.

**Termination:** Each vertex is marked visited at most once (the `if not visited(u)` guard prevents re-entering an already-visited vertex), so the recursion depth and total number of calls are both bounded by $|V|$, and the algorithm terminates in finite time on a finite graph.

> [!Note]
> This proof only establishes correctness for the connected component containing `s` — `explore` says nothing about vertices outside that component by design (see [[#Drawbacks / Constraints|Drawbacks]] below).

---
# Time & Space Complexity Analysis

## General Case

Let $C$ be the connected component containing `s` (i.e. the set of vertices reachable from `s`), with $|V_C|$ vertices and $|E_C|$ edges.

| |Complexity|Notes|
|---|---|---|
|Time|$O(\|V_C\| + \|E_C\|)$|Each reachable vertex is visited once; each of its outgoing edges is examined once|
|Space|$O(\|V_C\|)$ worst case|Recursion (call stack) depth can be as large as the component itself on a narrow/deep path|

Note this is **not** $O(|V| + |E|)$ over the whole graph — `explore` only touches the component reachable from `s`. To cover an entire (possibly disconnected) graph, you need the outer loop shown in [[Depth First Search (DFS)]], which restarts `explore` on every still-unvisited vertex.

## Implementation-Dependent Variations

|Data Structure Choice|Impact on Time|Impact on Space|Notes|
|---|---|---|---|
|Recursive (implicit stack) vs. explicit stack|Same asymptotically|Recursion adds call-stack overhead; risk of stack overflow on deep/skinny components|The pasted algorithm here is the recursive form|
|Adjacency list vs. matrix|$O(V_C+E_C)$ vs $O(V_C^2)$|$O(V_C+E_C)$ vs $O(V_C^2)$|Matrix wastes time/space scanning non-edges|
|`visited`: boolean array vs. hash set|$O(1)$ either way|$O(V)$ either way|Array needs vertices indexable by small integers|

## Best / Worst / Average Case

- **Best case:** `s` has no unvisited neighbors — $O(1)$ beyond the initial call.
- **Worst case:** Full connected component must be traversed — $O(|V_C| + |E_C|)$.
- **Average case:** Same order as worst case; no probabilistic behavior to average over.

---
# Drawbacks / Constraints

- **Only reaches one connected component.** `explore` only reaches the set of vertices reachable from `s` — it will never touch vertices in a different connected component, even if they exist in the same graph object.
- **To cover a disconnected graph**, you must restart `explore` on a vertex that has not yet been visited — this is exactly what the outer loop in [[Depth First Search (DFS)]] does.
- **Reachability only, by default.** The base version only tells you _whether_ a path exists (`visited(u)`), not _what_ the path is — use the path-record version (`prev`, `pre`, `post`) if you need the actual route or timing structure.
- **Recursion depth risk.** Since `explore` is naturally recursive, a very deep/narrow component can exhaust the call stack; an explicit-stack iterative rewrite avoids this.

> [!Abstraction] Definition
> Connected Undirected Graph An undirected graph $G$ is connected if **for every pair of vertices $(v, u)$** in $G$, there exists **a path from $v$ to $u$**.

---

# References / Links

- [[Depth First Search (DFS)]]
- [[Breadth First Search (BFS)]]
- [[Graph Reachability#Graph Search Algorithm|Graph Search]]