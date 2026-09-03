---
description: Linear-time single-source shortest path using topological order instead of a priority queue — works even with negative weights, since a DAG can't have cycles.
tags:
  - algorithm
  - dynamic-programming
  - graph-algorithms
  - Examples
aliases:
  - Shortest Path in a DAG
---
> [!abstract] 
> Given a DAG with edge weights, find the shortest path from a given vertex $s$ to all other vertices.
> 
> ![[Pasted image 20260712135947.png]]
> 
> - **Category:** Dynamic Programming / Graph Shortest Path
> - **Input:** A DAG $G=(V,E)$ with edge weights $\ell(e)$ (positive **or** negative), source vertex $s$
> - **Output:** $dist(v)$ for every vertex — the shortest distance from $s$ to $v$
> - **Paradigm:** Dynamic Programming over a topological order
> - **Typical use cases:** shortest (or longest) path whenever the graph has no cycles — task/dependency DAGs, and as the underlying reduction target for other DP problems like [[Edit Distance]] and [[Longest Increasing Subsequence Example]]

---

# Problem Specification

- **Instance:** A DAG $G=(V,E)$ with edge weights $\ell(e)$, and a source vertex $s$.
- **Solution Format:** $dist(v)$ for every $v \in V$.
- **Constraints:** $dist(v)$ must equal the length of the shortest path from $s$ to $v$ (undefined/infinite if unreachable).
- **Objective:** Minimize path length, per vertex.
- **Goal:** Minimize.

---

# Candidate Strategies / Approaches

## Dijkstra's Algorithm (general graphs) ✘ — works, but overkill

[[Dijkstra's Algorithm]] solves this too, but pays for a priority queue it doesn't need here, and can't handle negative edge weights at all in the general case. It also doesn't exploit the fact that the graph has no cycles.

## DAG DP via Topological Order ✔

Since a DAG has no cycles, there's a fixed order (a topological order) in which every edge points "forward." Process vertices in that order, and every predecessor of a vertex is guaranteed already finalized by the time you reach it — no need to repeatedly search for "the next closest unfinished vertex" the way Dijkstra's does.

> [!tip] Key Idea 
> A priority queue exists to answer "which unfinished vertex is currently closest?" — but in a DAG, topological order already answers that question for free, once and for all, before the algorithm even starts. That's what turns this into a single $O(V+E)$ pass instead of Dijkstra's $O((V+E)\log V)$.

---

# Dynamic Programming Solution

## 1. Subproblem

Let $dist(v)$ be the shortest distance from $s$ to $v$.

## 2. Base Case

$$
dist(s) = 0 
$$

## 3. Recursion

$$
dist(x) = \min_{(v,x) \in E} { dist(v) + \ell(v,x) } 
$$

## 4. Ordering

**Topological Order.** The trick: order the vertices topologically. Then, by the time you compute $dist$ for a given vertex, every one of its predecessors (every $v$ with an edge $(v,x)$) has already been finalized — so a single left-to-right pass suffices.

![[Pasted image 20260712140317.png]]

## 5. Iterative Algorithm

```pseudo
	\begin{algorithm}
	\caption{Shortest Path in a DAG}
	\begin{algorithmic}
	\Procedure{DAGDP}{$G, s$}
		\State $dist(s) = 0$
		\State Let $v_1, \dots, v_n$ be a list of all vertices after $s$ in topological ordering
		\For{$i = 1 \dots n$}
			\State $dist(v_i) = \underset{(v, v_i) \in E}{\min} \{ dist(v) + \ell(v, v_i) \}$
        \EndFor
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`dist`|Array (vertex → number)|Shortest known distance from `s`; finalized in topological order, never revisited|
|`v_1,...,v_n`|Topologically sorted vertex list|The processing order — guarantees every predecessor of `v_i` is already finalized when `v_i` is reached|

## Helper Functions / Operations Used

- **Topological sort** — computed once up front; $O(V+E)$.
- **Min over incoming edges** — for each vertex, examine each incoming edge once; summed over all vertices, this is $O(E)$ total across the whole algorithm.

---

# Proof of Correctness / Optimality

**Claim:** upon termination, $dist(v)$ equals the true shortest-path distance from $s$ to $v$, for every $v \in V$.

**Proof (induction on position in the topological order):**

- **Base case:** $dist(s) = 0$ is correct — the shortest path from $s$ to itself has length 0.
- **Inductive Hypothesis:** every vertex processed before $v_i$ in the topological order has a correctly computed $dist$.
- **Inductive Step:** any shortest path to $v_i$ must arrive via its last edge, $(v, v_i)$, from some predecessor $v$. Since $G$ is a DAG and vertices are processed in topological order, every such predecessor $v$ appears **before** $v_i$ in the ordering — so by the Inductive Hypothesis, $dist(v)$ is already correct when $v_i$ is processed. Taking the minimum of $dist(v) + \ell(v,v_i)$ over every incoming edge therefore considers every possible "last edge" of a shortest path to $v_i$, and picks the best one — so $dist(v_i)$ is set correctly.

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(\|V\| + \|E\|)$|Topological sort is $O(V+E)$; the main loop examines every incoming edge of every vertex exactly once, total $O(E)$|
|Space|$O(V+E)$|Adjacency structure for the graph, plus the `dist` array ($O(V)$)|

## Best / Worst / Average Case

- **Best / Worst / Average case:** all $O(V+E)$ — every vertex and edge is examined exactly once regardless of the specific weights or graph shape.

---

# Notable Properties

- **Works for positive _and_ negative weights.** Being a DAG prevents the one thing that breaks shortest-path algorithms with negative weights in general graphs: negative cycles. Since there are no cycles at all, there's nothing to loop around indefinitely.
- **Longest path for free.** To find the _longest_ path instead, just negate every edge weight and run the same algorithm — safe here specifically because a DAG can never have a negative cycle to exploit, unlike general graphs (where longest path is NP-hard).
- **Powers other DP solutions.** Solving [[Edit Distance Example|Edit Distance]] this way is $O(nm)$ time — matching that problem's $nm$ vertices and $\approx 3nm$ edges exactly. See [[Longest Increasing Subsequence Example|Longest Increasing Subsequence]] for a worked example using the negation trick above.

---

# Drawbacks / Constraints

- **Only works on DAGs.** If the graph has any cycle — even with all-positive weights — this algorithm doesn't apply; use [[Dijkstra's Algorithm]] (non-negative weights) or Bellman-Ford (negative weights, general graphs) instead.
- **Requires a topological order to exist and be computed first**, adding a real (if linear) upfront cost.

---

# References / Links

- [[Computer Science Introduction/Algorithms/Dynamic Programming/index|Dynamic Programming]]
- [[Dijkstra's Algorithm]]
- [[Edit Distance Example|Edit Distance]]
- [[Longest Increasing Subsequence Example]]
- [[Graph Reachability]]