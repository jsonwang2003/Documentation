> [!abstract] 
> In complex social networks or connectivity problems — like tracking connections between people in a network — standard graph traversals (BFS/DFS) can be too slow for frequent queries. The **Disjoint Set** ADT provides a way to merge groups and check connectivity in **near-constant time**.
> 
> - **Category:** Tree-based ADT (Up-Tree / forest)
> - **Stores:** a partition of elements into disjoint subsets, each with a representative "name"
> - **Built on top of:** arrays (the whole forest can be stored as a single array of parent pointers)
> - **Typical use cases:** cycle detection in [[Kruskal's Algorithm]], connectivity queries in dynamic graphs, network/social-graph "are these two in the same group" checks

---

# Core Structure

The most efficient way to represent disjoint sets is an **Up-Tree**. Unlike a standard tree where parents point to children, in an Up-Tree **children point to their parents**.

- **Sentinel Nodes (Roots):** the "representative" or "name" of the set. A node with no parent (a self-pointer) is the root.
- **Array Representation:** the entire forest can be stored in a single array — `Array[i]` stores the index of the parent of $i$. If `Array[i] == -1` (or another sentinel value), node $i$ is a root.

> [!Example] 
> ![[Pasted image 20260301120131.png]]
> 
> ![[Pasted image 20260301120136.png]]

> [!tip] Key Idea 
> Attaching the _shorter_ (or _smaller_) tree under the _taller_ (or _larger_) one during `Union`, plus flattening paths during `Find`, is what keeps the whole structure close to constant-time per operation despite each individual tree technically being able to grow.

## Properties

- **Invariant(s):** the structure is always a forest of up-trees — every non-root node has exactly one parent, and following parent pointers from any node always reaches a root in finite steps (no cycles).
- **Shape guarantee:** with union by rank or union by size, worst-case tree height is $O(\log n)$; with path compression added, amortized cost per operation drops to $O(\alpha(n))$.
- **Space complexity:** $O(n)$ — a single array of size $n$ (one parent pointer per element), regardless of whether rank/size is tracked in a second array or packed into the same one.
- **What it does NOT guarantee:** no ordering among elements within a set; no way to "split" a set back apart once two sets are unioned; no way to enumerate all members of a set efficiently (only membership/representative queries).

## Why the Invariant Holds

**Claim 1 — Ranks correspond to heights:** if a vertex has rank $k$, the height of its tree is $k$.

_Proof (induction on $k$):_ a vertex by itself has rank $0$ and height $0$. Assume every rank-$k$ vertex roots a height-$k$ tree. A vertex reaches rank $k+1$ only when `Union` merges two equal-rank ($k$) roots, making one a child of the other — so the new root's tree height becomes $k+1$ (the old height-$k$ subtree, plus one edge).

**Claim 2 — Rank implies a minimum tree size:** a root of rank $k$ has at least $2^k$ vertices in its tree.

_Proof (induction on $k$):_ a root of rank $0$ has at least $2^0=1$ vertex. Assume a root of rank $k$ has at least $2^k$ vertices. A root of rank $k+1$ can only form by unioning two rank-$k$ roots, so it has at least $2^k + 2^k = 2^{k+1}$ vertices.

**Result — maximum height is $O(\log n)$:** with $n$ total vertices, a vertex of rank $\log n$ already has at least $n$ vertices in its tree by Claim 2 — there's no room for a higher rank. Combined with Claim 1, no tree can have height greater than $O(\log n)$ under union by rank (and the same bound holds under union by size, by a symmetric argument).

---

# Data Structure Operations

## `Makeset(x)`

Initializes `x` as its own singleton set — a root with no children.

- **Time complexity:** $O(1)$
- **Notes:** sets `π(x) = x` (self-pointer, marking it a root) and `rank(x) = 0`.

```pseudo
	\begin{algorithm}
	\caption{Makeset}
	\begin{algorithmic}
	\Procedure{Makeset}{$x$}
		\State $\pi(x) = x$
		\State $rank(x) = 0$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Find(x)`

Determines which set `x` belongs to by walking parent pointers up to the root. If $Find(u) = Find(v)$, then `u` and `v` are in the same set.

- **Time complexity:** $O(\log n)$ worst case with union by rank/size alone; amortized $O(\alpha(n))$ with path compression added
- **Notes:** returns the root, which acts as the set's "name" — this is the value to compare for connectivity checks.

```pseudo
	\begin{algorithm}
	\caption{Find}
	\begin{algorithmic}
	\Procedure{Find}{$x$}
		\While{$x \neq \pi(x)$}
			\State $x = \pi(x)$
        \EndWhile
        \Return $x$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Union(x, y)`

Merges the set containing `x` with the set containing `y` into a single set, by attaching one root under the other.

- **Time complexity:** $O(\log n)$ worst case with union by rank/size; amortized $O(\alpha(n))$ with path compression
- **Notes:** always attaches the _shorter_ (rank) or _smaller_ (size) tree under the taller/larger one — attaching arbitrarily (e.g. always $x$ under $y$) can degrade the structure to a straight chain, see [[#Common Pitfalls]].

```pseudo
	\begin{algorithm}
	\caption{Union}
	\begin{algorithmic}
	\Procedure{Union}{$x, y$}
		\State $r_x = find(x)$
		\State $r_y = find(y)$
		\If{$r_x = r_y$}
			\State $\pi(r_y) = r_x$
		\Else
			\State $\pi(r_x) = r_y$
			\If{$rank(r_x) = rank(r_y)$}
				\State $rank(r_y) = rank(r_x) + 1$
            \EndIf
        \EndIf
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Union Variants

### Union-by-Size

Always attach the root of the **smaller** tree (fewer nodes) to the root of the **larger** tree.

> [!Example] $Union(F, E)$ 
> ![[Pasted image 20260301120257.png]]
>  ![[Pasted image 20260301120354.png]]

- **Worst-case height:** $O(\log n)$
- **Benefit:** easy to track — just store the negative size in the sentinel's array slot (e.g. `-5` means it's a root of a set with 5 nodes), so no second array is strictly needed.

### Union-by-Height (Rank)

Always attach the **shorter** tree to the **taller** tree.

> [!Example] $Union(A, C)$ 
> ![[Pasted image 20260301120510.png]]
> 
> ![[Pasted image 20260301120517.png]]

- **Worst-case height:** $O(\log n)$
- **Drawback:** harder to maintain once Path Compression is also in use, since path compression changes heights during searches without updating `rank` (rank becomes an _upper bound_ on height rather than the exact height once compression is active).

> [!Question] If we used Union-by-Size instead of Union-by-Height on the example above, would the result be better, worse, or the same? 
> In the provided example, Union-by-Size would likely produce a tree with the same or better performance than Union-by-Height, but in practice, size is preferred anyway because it's easier to keep updated once Path Compression starts moving nodes around.

## Optimizing `Find`: Path Compression

Every `Find(u)` call walks from `u` up to the root. **Path Compression** says: once you've found the root, go back and reattach `u` — and every node on the path — **directly to the root**.

```python
def find(parent, u):
    if parent[u] != u:
        parent[u] = find(parent, parent[u])  # path compression
    return parent[u]
```

> [!Example] $Find(A)$
> ![[Pasted image 20260301120853.png]]
> 
> Sees the nodes $(B, F)$ along the traversal up
> 
> ![[Pasted image 20260301120937.png]]

- **Result:** the next `Find` call on any of those flattened nodes takes $O(1)$.
- **Self-adjustment:** this turns the Up-Tree into a **self-adjusting structure** — the more it's used, the flatter (and faster) it gets.

## Amortized Cost Analysis

A single `Find` is $O(h)$ where $h$ is tree height — with a poorly shaped tree that could be $O(n)$. **Amortized analysis** instead looks at the cost of a _sequence_ of $m$ operations: an expensive $O(n)$ `Find` is actually an **investment**, since path compression permanently flattens every node it touches, making all future `Find` calls on those nodes $O(1)$.

Three formal frameworks are used to prove the amortized cost is nearly constant:

- **Aggregate Method** — show the _total_ time $T(m)$ for a sequence of $m$ operations is small; amortized cost per operation is $T(m)/m$.
- **Accounting (Banker's) Method** — charge each cheap operation slightly more than it actually costs, banking the difference as credit; an expensive `Find` withdraws banked credit to pay for the extra path-compression work.
- **Potential (Physicist's) Method** — define a potential function $\Phi$ representing tree "messiness" (height); an expensive `Find` does work that sharply _decreases_ $\Phi$ (by flattening the tree), offsetting its high actual cost.

__Why $\log n$, not just $O(1)$:_* trees aren't perfectly flat immediately, so the true bound is $O(m \log^* n)$ rather than $O(m)$. $\log^* n$ (iterated logarithm) grows so slowly it's effectively constant for any physically realistic $n$:

|Total Elements ($n$)|$\log^* n$|
|---|---|
|2|1|
|4 ($2^2$)|2|
|16 ($2^4$)|3|
|65,536 ($2^{16}$)|4|
|$2^{65536}$ (more than atoms in the universe)|5|

**The result:** with union by size (or rank) _and_ path compression, the average cost per operation is $O(\alpha(n))$ — the **Inverse Ackermann function** — which never exceeds 5 for any dataset humanity could ever store. Without amortized analysis, you'd see the $O(n)$ worst case of a single `Find` and wrongly conclude the structure is inefficient; in reality, the more you use it, the faster it gets.

|Operation|Naive Implementation|With Union-by-Size/Rank + Path Compression|
|---|---|---|
|**Union**|$O(n)$|$O(\alpha(n)) \approx O(1)$|
|**Find**|$O(n)$|$O(\alpha(n)) \approx O(1)$|

**Summary:** for $m$ operations on $n$ elements, total worst-case time is $O(n + m\log^* n)$, i.e. $O(\log^* n) \approx O(\alpha(n))$ per operation on average — considered effectively $O(1)$ in practice.

---

# Common Pitfalls

- **Skipping union by rank/size entirely.** Naively attaching one root under another with no size/height check (e.g. always attaching `x` under `y`) can degrade the tree into a straight chain — height $O(n)$, making `Find` $O(n)$ per call. This is the single biggest way to lose all the structure's benefits.
- **Trusting `rank` as exact height once Path Compression is active.** Path compression flattens paths without updating `rank`/`size` bookkeeping for the nodes it moves, so `rank` becomes an _upper bound_ on height rather than the literal height — don't rely on it for anything beyond the union comparison it was designed for.
- **Conflating the two array-encoding conventions.** Union-by-size is often implemented by packing size directly into the root's array slot as a negative number (`-5` = root of a 5-element set), while union-by-rank typically uses a separate `rank` array — mixing these conventions in one implementation (e.g. writing a raw size into a slot that `Find` expects to be a parent pointer) silently corrupts the structure.
- **Forgetting `Find` returns the root, not a boolean.** A common off-by-one in usage: comparing `Find(u) == Find(v)` is correct for connectivity; comparing `u == v` or checking `π(u)` directly is not.

---

# Tradeoffs Compared to Other Data Structures

|Structure|Check connectivity|Merge two groups|Notes|
|---|---|---|---|
|**Disjoint Set (Union-Find)**|$O(\alpha(n))$ amortized|$O(\alpha(n))$ amortized|Best when you only need "same group?" queries and one-way merges — no need to ever split|
|BFS/DFS on adjacency list (re-run per query)|$O(V+E)$ per query|N/A — recompute from scratch|Fine for a handful of static queries; far too slow for many connectivity queries interleaved with merges|
|Hash Set per group (store members explicitly)|$O(1)$ membership within a known group|$O(\text{size of smaller set})$ to merge (must relabel every element)|Supports enumeration of a group's members directly, which Union-Find doesn't, at the cost of slower merges|
|Balanced BST (e.g. by group ID)|$O(\log n)$|$O(\log n)$ or worse (must rebuild)|Rarely used for this purpose — no advantage over Union-Find for pure connectivity, but supports ordered queries Union-Find can't|

> [!note] When to reach for this structure Use Disjoint Sets whenever the pattern is "repeatedly merge groups, and repeatedly ask whether two elements are in the same group" — and you never need to split a group back apart or enumerate its members. This is exactly the shape of Kruskal's cycle check: "are these two endpoints already connected?"

---

# Related Notes

**Algorithms that use this structure:**

- [[Kruskal's Algorithm]] — uses `Find` to detect whether an edge's endpoints are already connected (would create a cycle) and `Union` to merge components as MST edges are accepted

**Other structures built on top of this one:**

- {{none currently in this vault}}

**Structures this one is built from:**

- Plain arrays — the entire forest is stored as a single array of parent pointers (optionally packing rank/size into the same array for roots)