---
description: "A tree-based collection structure managing partitions of elements into disjoint subsets with near-constant time operations."
aliases:
  - Disjoint Set ADT
  - Union-Find Data Structure
  - Up-Tree Forest
  - Partition Manager
tags:
  - data-structures
  - graph-algorithms
  - amortized-analysis
  - partition-sets
---
# Abstract 
When tracking grouping properties across dynamic networks—such as monitoring unified components or tracking connections in real time—standard graph traversals like [[Depth First Search (DFS)|DFS]] or [[Breadth First Search (BFS)|BFS]] carry massive overhead costs if invoked repeatedly. The **Disjoint Set Abstract Data Type (ADT)** solves this constraint by maintaining isolated subsets under an optimization model that merges groups and checks path connectivity in near-constant execution time.

**Category:** Tree-based ADT (Up-Tree Forest)  
**Stores:** A mathematical partition of elements split into disjoint subsets, where each group is managed by a unique representative node.  
**Built on top of:** Plain standard sequential arrays.  
**Typical use cases:** Dynamic cycle tracking inside [[Minimum Spanning Trees|Kruskal's Algorithm]], image segmentation tracking, network component clustering.

---

## Core Structure
The absolute most space-efficient mechanism to realize a Disjoint Set ADT is the **Up-Tree**. Unlike a traditional standard tree layout where parent references point down to their child arrays, nodes inside an Up-Tree point *upward* to their parent targets.

*   **Sentinel Root Nodes:** The absolute top representative node of a subset acts as the "name" of that group. A node pointing to itself or tracking a negative size flag is identified instantly as a root.
*   **Array Allocation Trick:** Because every node points strictly to a single parent node, an entire Up-Tree forest can be packed inside a single 1D flat integer array (`parent[]`). The index maps the item ID, and the slot value tracks its parent index. If `parent[i] < 0`, node `i` is determined to be a root.

![[Pasted image 20260301120131.png]]
![[Pasted image 20260301120136.png]]

> [!TIP] Key Idea
> Attaching the shorter (or smaller) tree under the taller (or larger) one during Union, plus flattening paths during Find, is what keeps the whole structure close to constant-time per operation despite each individual tree technically being able to grow.

---

## Properties

*   **Invariant(s):** The structure remains a strict forest of Up-Trees. Every non-root entry references exactly one parent, and chasing those upward references from any node guarantees hitting a root sentinel in finite iterations without encountering infinite cyclic traps.
*   **Shape Guarantee:** Enforcing smart Union balances caps the worst-case tree height at $O(\log n)$. Interlocking this with explicit Path Compression drops the long-term amortized runtime per operation down to a near-constant $O(\alpha(n))$, where $\alpha$ is the Inverse Ackermann Function.
*   **Space Complexity:** Strict linear allocation $O(n)$ to store parent paths and structural tracking statistics.
*   **What it does NOT guarantee:** Does not preserve an internal sorted element sequence; cannot easily split or partition a single group back into isolated elements once a merge is committed; cannot list all items inside a specific set without reading the entire array tracking scope.

---

## Why the Invariant Holds

### Lemma 1: Ranks Match Heights
If an Up-Tree vertex maintains an independent rank value $k$, the actual maximum height of its structural branch under clean union balancing is exactly $k$.
*   *Proof by Induction:* A singleton item begins at rank 0 and height 0. Assume every rank-$k$ node bounds a maximum branch height of $k$. A root can only climb to rank $k+1$ if a `Union` command attempts to merge two roots of identical rank $k$. One root becomes a child of the other, incrementing the height of the newly formed root structure to exactly $k+1$.

### Lemma 2: Rank Sizes Grow Exponentially
An Up-Tree root vertex holding rank $k$ is guaranteed to contain at least $2^k$ total elements within its underlying tree partition.
*   *Proof by Induction:* A root node at rank 0 holds at least $2^0 = 1$ item. Assume a rank-$k$ root bounds at least $2^k$ entries. To scale a tree to rank $k+1$, we must merge two individual rank-$k$ component trees. Summing their independent boundaries yields: $2^k + 2^k = 2^{k+1}$ elements.

### Theorem: Maximum Tree Height is Logarithmic
Given a total dataset constraints layout of $n$ elements, any vertex reaching rank $\log_2 n$ would demand an explicit footprint size of at least $n$ elements according to Lemma 2. There is mathematically zero physical room left inside the array allocations to grow a rank higher than this value. Paired with Lemma 1, this caps the structural height bounds of an Up-Tree at $O(\log n)$ when using Rank Balancing.

---

## Data Structure Operations

### Makeset(x)
Instantiates an independent item $x$ as its own singleton group partition.

```pseudo
\begin{algorithm}
\caption{Makeset Initialization}
\begin{algorithmic}
\Procedure{Makeset}{$x$}
    \State $\text{parent}[x] \gets -1$
    \State $\text{rank}[x] \gets 0$
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

### Find(x)
Chases the parent array pointers upward to discover the core root representative of item $x$.

*   **Time Complexity:** $O(\log n)$ worst-case under raw balancing; drops instantly to an amortized $O(\alpha(n))$ when Path Compression is active.

```pseudo
\begin{algorithm}
\caption{Find with Path Compression}
\begin{algorithmic}
\Procedure{Find}{$x$}
    \If{$\text{parent}[x] < 0$}
        \Return $x$
    \EndIf
    \State $\text{parent}[x] \gets$ \Call{Find}{$\text{parent}[x]$}
    \Return $\text{parent}[x]$
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

#### Path Compression Optimization
Every single invocation of `Find(x)` maps out a clear path up to the root node. Path compression optimizes this path: as the recursive execution unrolls, it rewrites the parent pointers of *every single node encountered along the search track* to point directly to the top root node.

![[Pasted image 20260301120853.png]]
*Traversal track passes through nodes (B, F) to reach root.*

![[Pasted image 20260301120937.png]]
*Resulting Flattened Topology: Future searches along this track hit in $O(1)$ time.*

### Union(x, y)
Merges the complete tree sets containing elements $x$ and $y$ by linking the root node of the smaller collection beneath the root node of the larger collection.

```pseudo
\begin{algorithm}
\caption{Union by Rank}
\begin{algorithmic}
\Procedure{Union}{$x, y$}
    \State $rootX \gets$ \Call{Find}{$x$}
    \State $rootY \gets$ \Call{Find}{$y$}
    \If{$rootX \neq rootY$}
        \If{$\text{rank}[rootX] < \text{rank}[rootY]$}
            \State $\text{parent}[rootX] \gets rootY$
        \Else
            \State $\text{parent}[rootY] \gets rootX$
            \If{$\text{rank}[rootX] == \text{rank}[rootY]$}
                \State $\text{rank}[rootX] \gets \text{rank}[rootX] + 1$
            \EndIf
        \EndIf
    \EndIf
\EndProcedure
\end{algorithmic}
\end{algorithm}
```

---

## Union Variants

### Union-by-Size
Always routes the parent index of the root with fewer nodes to point directly to the root bounding a larger size footprint.
*   *Storage Optimization:* Can be cleanly packed inside a single tracking array. A root tracking value of `-5` signifies a sentinel node leading a group component size of exactly 5 nodes. This completely eliminates the need for an independent auxiliary tracking array.

![[Pasted image 20260301120257.png]] $\rightarrow$ ![[Pasted image 20260301120354.png]]

### Union-by-Rank (Height)
Always binds the shorter tree beneath the root node of the taller tree structure.
*   *Gotcha:* Once Path Compression starts flattening branches during execution lookups, "rank" shifts from tracking literal, active tree heights to acting as a fixed upper bound on potential height metrics.

---

## Common Pitfalls

> [!WARNING] The Linear Chain Degeneration Danger
> If you naively implement `Union` by blindly mapping root $x$ to point to root $y$ without evaluating rank or size metrics, a sequence of skewed inputs can collapse your Up-Trees into long, single-file linear chains. This spikes the operational height to $O(n)$, breaking all efficiency guarantees.

*   **Direct Parent Comparisons:** Evaluating raw pointer structures via `parent[u] == parent[v]` to confirm group connectivity will fail silently. You must explicitly evaluate paths through the full lookup pipeline: `Find(u) == Find(v)`.

---

## Tradeoffs Compared to Other Data Structures

| Structure                                          | Check Connectivity       | Merge Groups                    | Computational Advantage                                           |
| :------------------------------------------------- | :----------------------- | :------------------------------ | :---------------------------------------------------------------- |
| **Up-Tree Forest**                                 | $O(\alpha(n))$ amortized | $O(\alpha(n))$ amortized        | High-efficiency lookup for asymmetric collections.                |
| [[Graph Representations\|Standard BFS/DFS Passes]] | $O(\|V\| + \|E\|)$       | $O(1)$                          | Simple architecture, but too slow for heavy interleaving lookups. |
| **Hash Set Map Registries**                        | $O(1)$ worst-case        | $O(\text{Size of Smaller Set})$ | Supports listing set elements, but incurs high merge overhead.    |

---

## Related Notes
*   **[[Minimum Spanning Trees]]** — Relies completely on this structure to catch cyclic edge violations inside Kruskal's verification loop.
*   **[[Graph Representations]]** — Explains the sequential flat array layouts used to build basic Up-Tree architectures.