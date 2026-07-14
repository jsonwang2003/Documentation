---
description: Find the longest subsequence of strictly increasing values in a list — solved by reducing to a longest-path-on-a-DAG problem via edge negation.
tags:
  - algorithm
  - dynamic-programming
aliases:
  - LIS
  - Longest Increasing Subsequence
---


> [!abstract] Abstract Given a sequence of distinct positive integers $a[1], \dots, a[n]$, an **increasing subsequence** is a sequence $a[i_1], \dots, a[i_k]$ such that $i_1 < \dots < i_k$ and $a[i_1] < \dots < a[i_k]$.
> 
> - **Category:** Dynamic Programming / Sequence Problems
> - **Input:** A sequence of $n$ distinct positive integers
> - **Output:** The length (or an example) of the longest increasing subsequence
> - **Paradigm:** Reduction to Longest Path on a DAG (see [[Shortest Path in a DAG Example|Shortest Path in a DAG]])
> - **Typical use cases:** patience sorting, version-control diffing, any "find the longest chain of compatible items" problem

---

# Problem Specification

- **Instance:** A sequence $a[1], \dots, a[n]$ of distinct positive integers.
- **Solution Format:** A subsequence $a[i_1], \dots, a[i_k]$ with $i_1 < \dots < i_k$.
- **Constraints:** $a[i_1] < a[i_2] < \dots < a[i_k]$ (strictly increasing values), and indices strictly increasing.
- **Objective:** $k$, the length of the subsequence.
- **Goal:** Maximize.

**Example:**

$$15, 18, 8, 11, 5, 12, 16, 2, 20, 9, 10, 4$$

The longest increasing subsequence here is $8, 11, 12, 16, 20$ (length 5).

---

# Viewing LIS as Shortest Path in a DAG

- **What could the vertices be?** The values themselves — one vertex per element $a[i]$.
- **When is there an edge?** An edge from $a_i \to a_j$ if $i < j$ **and** $a_i < a_j$ (i.e. $a_j$ could immediately follow $a_i$ in an increasing subsequence).
- **What are the weights of edges?** $-1$ each.

> [!tip] Key Idea 
> Maximizing subsequence _length_ is the same as maximizing the _number of edges_ in a path through this DAG. [[Shortest Path in a DAG Example#Notable Properties|Notable Properties]] turns "maximize number of edges" into "minimize total (negative) weight" — a plain shortest-path problem, solvable by the same $O(V+E)$ topological-order DP, no priority queue needed.

> [!note] Completing the Reduction 
> This setup isn't quite single-source yet — an increasing subsequence can start at _any_ element, not just a fixed one. The standard fix: add a virtual source vertex $s$ with a $0$-weight edge to every $a_i$. Then $dist(v)$ (in this $-1$-weighted graph) is $-(k-1)$ for the longest increasing subsequence _ending_ at $v$, of length $k$. The overall answer is $1 - \min_v dist(v)$.

---

# Complexity

This DAG has $|V| = n$ vertices and up to $|E| = O(n^2)$ edges (every pair $i<j$ with $a_i < a_j$ potentially contributes an edge). Running [[Shortest path in a DAG Example#5. Iterative Algorithm|DAGDP]] on it costs $O(|V|+|E|) = O(n^2)$ — matching the classic direct-DP solution to Longest Increasing Subsequence (where $L[i] = 1 + \max_{j<i,\ a_j<a_i} L[j]$, computed directly without explicitly building a graph).

> [!note] 
> A better-known $O(n\log n)$ algorithm for LIS exists (using patience sorting / binary search over a list of smallest tail values per length), but it doesn't fit the DAG-shortest-path framing directly — it's a genuinely different technique, not a faster implementation of this reduction.

---

# References / Links

- [[Shortest Path in a DAG Example|Shortest Path in a DAG]]
- [[Dynamic Programming]]