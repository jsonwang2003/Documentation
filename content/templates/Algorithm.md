---
description: "{{One-sentence summary for search/graph previews — e.g. 'Graph traversal algorithm that explores as far as possible along each branch before backtracking.'}}"
tags:
  - algorithm
aliases:
---
> [!abstract] Abstract 
> {{One-sentence definition — e.g. "DFS is a graph/tree traversal algorithm that explores as far as possible along each branch before backtracking."}}
> 
> - **Category:** {{e.g. Graph Traversal / Sorting / Dynamic Programming / Greedy}}
> - **Input:** {{e.g. Graph $G=(V,E)$, source vertex $s$}}
> - **Output:** {{e.g. Set of reachable vertices, discovery/finish times, traversal order}}
> - **Paradigm:** {{e.g. Backtracking, Divide & Conquer, Greedy, DP}}
> - **Typical use cases:** {{e.g. cycle detection, topological sort, connected components}}

---

# Core Logic (High-Level)

<!-- Describe the intuition in plain English/steps before touching pseudocode. Aim for "how would I explain this to someone in 60 seconds". -->

1. {{Step 1 — e.g. Start at source node, mark as visited}}
2. {{Step 2 — e.g. Explore an unvisited neighbor, recurse/push to stack}}
3. {{Step 3 — e.g. Backtrack when no unvisited neighbors remain}}
4. {{Repeat until ...}}

> [!tip] Key Idea 
> {{The single "aha" idea that makes the algorithm work — e.g. "Using a stack (explicit or via recursion) ensures depth-first order."}}

---

# Pseudocode (Mid-Level Implementation)

<!-- Requires the "Pseudocode" (obsidian-pseudocode) plugin, which renders pseudocode.js-style \begin{algorithm} blocks. Use language tag "pseudo". -->

```pseudo
	\begin{algorithm}
	\caption{ {{Algorithm Name}} }
	\begin{algorithmic}
		\INPUT $G, s$
		\OUTPUT $\{{Output symbol}\}$
		\PROCEDURE{ {{AlgorithmName}} }{$G, s$}
			\State ...
		\ENDPROCEDURE
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`{{S / Q}}`|{{Stack / Queue / Priority Queue}}|{{Holds frontier of nodes to explore next}}|
|`visited`|{{Set / Boolean array}}|{{Tracks which nodes have been processed to avoid re-visiting}}|
|`{{u, v}}`|{{Node/vertex}}|{{Current node being processed / candidate neighbor}}|

## Helper Functions / Operations Used

- **`{{Neighbors(u)}}`** — {{returns adjacency list of u; O(1) amortized to get the list, O(deg(u)) to iterate}}
- **`{{push / pop / enqueue / dequeue}}`** — {{data structure primitive; complexity depends on backing structure (array vs linked list vs heap)}}
- **`{{Visit(u)}}`** — {{the "do work" hook, e.g. record discovery time, add to output list}}

> [!note] Low-Level Implementation  
> remove this if not needed — use for language-specific detail (array bounds, pointer handling, recursion stack depth, etc.)
> 
> ```{{python/cpp/java}}
> {{Actual code implementation here}}
> ```

---

# Proof of Correctness

<!-- Structure: Invariant -> Initialization -> Maintenance -> Termination, or induction, depending on algorithm type. -->

**Claim:** {{State what the algorithm guarantees, e.g. "Upon termination, visited contains exactly the set of vertices reachable from s."}}

**Loop Invariant:** {{e.g. "At the start of each iteration of the while loop, visited = set of all vertices whose full exploration has completed."}}

- **Initialization:** {{Show invariant holds before first iteration — e.g. visited = ∅, only s is in the frontier.}}
- **Maintenance:** {{Show that if invariant holds before an iteration, it still holds after — e.g. a node is only added to visited once, and all its neighbors are subsequently considered.}}
- **Termination:** {{Show the loop ends (frontier becomes empty) and invariant + termination condition together imply correctness.}}

**Why it doesn't miss/duplicate nodes:** {{e.g. the visited check prevents re-processing; every reachable vertex is eventually pushed because its predecessor is processed.}}

---

# Time & Space Complexity Analysis

## General Case

|       | Complexity | Notes                                                            |
| ----- | ---------- | ---------------------------------------------------------------- |
| Time  | $O(n)$     | {{Each vertex processed once, each edge examined at most twice}} |
| Space | $O(V)$     | {{visited set + frontier structure in worst case}}               |

## Implementation-Dependent Variations

|Data Structure Choice|Impact on Time|Impact on Space|Notes|
|---|---|---|---|
|{{Adjacency list vs matrix}}|{{O(V+E) vs O(V²)}}|{{O(V+E) vs O(V²)}}|{{matrix wastes time on sparse graphs}}|
|{{Recursive vs explicit stack}}|{{Same asymptotically}}|{{Recursion adds call-stack overhead, risk of stack overflow on deep graphs}}||
|{{Array-based queue/stack vs linked list}}|{{O(1) amortized vs O(1) but with pointer overhead}}|{{—}}||
|{{Visited set: hash set vs boolean array}}|{{O(1) avg vs O(1) worst}}|{{O(V) either way}}|array requires known/bounded ID space|

## Best / Worst / Average Case

- **Best case:** {{e.g. target found immediately / graph is a single path}}
- **Worst case:** {{e.g. must explore entire graph}}
- **Average case:** {{if applicable}}

---

# Drawbacks / Constraints

- **Preconditions:** {{e.g. graph must be represented with accessible adjacency info; no negative-cycle assumption if relevant}}
- **Fails / degrades when:** {{e.g. extremely deep recursion → stack overflow; disconnected graph → won't visit all nodes without outer loop over all vertices}}
- **Not suitable for:** {{e.g. finding shortest path in unweighted graph — use BFS instead; weighted shortest path — use Dijkstra/Bellman-Ford}}
- **Alternatives to consider:** {{e.g. BFS for shortest path, iterative deepening for memory-constrained depth search}}

---

# References / Links

- {{Textbook: CLRS Ch. X}}
- [[Related Algorithm Note]]
- {{Lecture slides / source}}