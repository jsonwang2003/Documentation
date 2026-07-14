---
aliases:
  - Bellman Ford
  - Shortest Path with Negative Weights
tags:
  - algorithm
  - dynamic-programming
  - graph-algorithms
description: Dynamic programming algorithm for single-source shortest paths on graphs with possibly negative edge weights, detecting negative cycles along the way, in O(VE) time.
---
> [!abstract] 
> Given a graph with vertices $v_0, v_1, \dots, v_{n-1}$ (source $v_0$) and possibly **negative** edge weights, find the shortest distance from $v_0$ to every other vertex — or detect that no shortest path even exists, because a negative cycle is reachable.
> 
> - **Category:** Dynamic Programming / Graph Shortest Path
> - **Input:** A graph $G$ with $n$ vertices, possibly negative edge weights, source $v_0$
> - **Output:** Either `"NEGATIVE CYCLE"`, or the shortest distance from $v_0$ to every vertex
> - **Paradigm:** Dynamic Programming, budgeting the number of edges allowed
> - **Typical use cases:** graphs with negative weights in general (not just DAGs); detecting negative cycles specifically (e.g. currency arbitrage detection, where a negative cycle means a risk-free profit loop exists)

---

# Negative Cycles

> [!info] Definition A **negative cycle** is a cycle in a graph such that the sum of its edge weights is negative.
> 
> From [[Dijkstra's Algorithm#Drawbacks / Constraints|Dijkstra's]], one of its main constraints is that it only works with non-negative edge weights.
> 
> ![[Pasted image 20260712172426.png]]
> 
> From the image, the cycle ${A, B, C, E}$ forms a negative cycle, since $(-2) + 4 + (-8) + 4 = -2$.

If a graph has negative edge weights _and_ cycles, finding the shortest path can be problematic: with a negative cycle, there are paths whose lengths are **unbounded from below** — you can always go around the cycle one more time and get a "shorter" path, forever.

---

# Problem Specification

- **Instance:** Graph $G$ with vertices $v_0, \dots, v_{n-1}$, possibly negative edge weights, source $v_0$.
- **Solution Format:** Either a negative-cycle report, or an array of shortest distances.
- **Constraints:** None on the input weights (unlike Dijkstra's).
- **Objective / Goal:** Find the true shortest distance to every vertex, or correctly detect that no finite shortest distance exists for some vertex.

---

# Candidate Strategies / Approaches

## Dijkstra's Algorithm ✘

If there are **no** negative cycles, [[Dijkstra's Algorithm|Dijkstra's]] can technically be adapted to find shortest paths — but its efficiency guarantee relies entirely on all edge weights being non-negative. With negative edge weights present (even without a negative cycle), Dijkstra's runtime can blow up to exponential, since a vertex may need to be revisited and improved many times after being "finalized" too early.

## Dynamic Programming (Bellman-Ford) ✔

Since DP always solves shortest paths on DAGs by processing vertices in a fixed (topological) order — see [[Shortest Path in a DAG Example|Shortest Path in a DAG]] — the natural generalization to graphs _with_ cycles is to put a **budget** $T$ on how many edges we're allowed to use. Budgeting the path length effectively "unrolls" the graph into layers indexed by $t = 0, 1, \dots, T$, sidestepping the cycle problem entirely: you can never revisit an earlier, smaller budget layer.

---

# Dynamic Programming Solution

## 1. Subproblems

Let $B[i,t]$ be the length of the shortest path from $v_0$ to $v_i$ using **at most** $t$ edges.

## 2. Base Cases (assuming no negative cycles)

$$ B[0,t] = 0 \qquad B[i,0] = \infty, \text{ for } i \geq 1 $$

## 3. Recursion

To compute $B[i,t]$: ask which vertex is the **second-to-last** vertex on the shortest path from $v_0$ to $v_i$ using at most $t$ edges.

![[Pasted image 20260712200928.png]]

$$ 
\begin{align*} 
\text{Case 0}&: v_0 \text{ is the second-to-last vertex}\\
\text{Case 1}&: v_1 \text{ is the second-to-last vertex}\\
&\vdots\\
\text{Case } n-1&: v_{n-1} \text{ is the second-to-last vertex} 
\end{align*} 
$$

$$ 
B[i, t] = \min
\begin{cases}  \\
B(0, t-1) + w(v_0, v_i) \\
B(1, t-1) + w(v_1, v_i) \\ \\
\dots \\ \\ \\
B(n-1, t-1) + w(v_{n-1}, v_i)  \\
\end{cases} 
= \min_{(v_j, v_i) \in E} \big[ B(j, t-1) + w(v_j, v_i) \big] 
$$

## 4. Ordering — What's the Maximum Budget?

**Answer: $n-1$.** Assuming no negative cycles, every shortest path must be a _simple_ path (never repeats a vertex) — since repeating a vertex would mean a cycle exists on the path, and a non-negative cycle could only be removed to make the path shorter or equal, while a negative cycle would already violate the "no negative cycles" assumption. A simple path in a graph with $n$ vertices has at most $n-1$ edges. So order $t$ from $0, 1, \dots, n-1$.

### Detecting Negative Cycles

What if we don't know beforehand whether the graph has negative cycles? If there are **no** negative cycles, the array values will never improve after $t$ grows past $n-1$ (there's no longer/better simple path to find). So:

$$ \text{If there exists } i \text{ such that } B[i, n-1] \neq B[i, n], \text{ then a negative cycle exists.} $$

---

# Bellman-Ford Algorithm

```pseudo
	\begin{algorithm}
	\caption{Bellman Ford}
	\begin{algorithmic}
	\Procedure{BFDP}{$G, v_0$}
		\State $B[0,0] = 0$
		\State $B[i, 0] = \infty$ for all $i \neq 0$
		\For{$t = 1, \dots , n$}
			\For{$i = 0, \dots, n-1$}
				\State $B[i,t] = \underset{(v_j, v_i) \in E}{\min}[B[j, t-1] + w(v_j, v_i)]$
            \EndFor
        \EndFor
        \For{$i = 0, \dots, n-1$}
	        \If{$B[i, n-1] \neq B[i,n]$}
		        \Return "Negative Cycle"
            \EndIf
        \EndFor
        \Return $[B[0, n], B[1, n], \dots, B[n-1, n]]$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`B`|2D array, $n \times (n+1)$|`B[i][t]` = shortest distance from $v_0$ to $v_i$ using at most $t$ edges|
|`t`|Budget|Number of edges allowed so far, from $0$ up to $n$ (one extra round beyond $n-1$, used purely to detect negative cycles)|

## Helper Functions / Operations Used

- **Min over incoming edges** — for each $(i,t)$, scan every edge $(v_j, v_i) \in E$; $O(\deg^{-}(v_i))$ per cell, summing to $O(E)$ per full round of $t$.

---

# Proof of Correctness

**Claim (assuming no negative cycle):** $B[i,t]$ equals the true shortest distance from $v_0$ to $v_i$ using at most $t$ edges.

- **Base case:** $B[0,0] = 0$ (staying at the source uses 0 edges, correctly length 0), $B[i,0] = \infty$ for $i \neq 0$ (no vertex besides the source is reachable with 0 edges).
- **Inductive Hypothesis:** $B[j, t-1]$ is correct for every $j$.
- **Inductive Step:** consider the true shortest path to $v_i$ using at most $t$ edges. If $v_i = v_0$, the trivial 0-edge path already achieves length 0, and no path can do better without a negative cycle, so $B[0,t]=0$ remains correct. Otherwise ($i \neq 0$), that path — however many edges it actually uses, up to $t$ — arrives via _some_ last edge $(v_j, v_i)$, and the portion before that last edge is itself a shortest path to $v_j$ using at most $t-1$ edges (since removing the last edge removes exactly one edge from the budget). By the Inductive Hypothesis, that sub-path's length is exactly $B[j,t-1]$. Since $B[i,t]$ takes the minimum of $B[j,t-1] + w(v_j,v_i)$ over every possible predecessor $j$, it correctly recovers the shortest such path.

**Negative cycle detection is correct** because, absent a negative cycle, every shortest path is simple (at most $n-1$ edges), so $B[i,\cdot]$ must stabilize by $t=n-1$ and never improve again — meaning any observed improvement between $t=n-1$ and $t=n$ can only be explained by a cycle that keeps helping, which (since it _does_ help, i.e. strictly decreases the distance) must be a negative cycle.

---

# Time & Space Complexity Analysis

## General Case

For a graph $G$ with $n$ vertices and $m$ edges:

$$ O(n(n+m)) $$

(the outer loop runs $n$ times for $t$; each round does $O(n+m)$ work — $O(n)$ for the vertices themselves, $O(m)$ summed over all the incoming-edge scans).

Assuming $n = O(m)$ (a connected graph, roughly):

$$ O(nm) = O(|V||E|) $$

| |Complexity|Notes|
|---|---|---|
|Time|$O(\|V\|E\|)$|Substantially slower than [[Dijkstra's Algorithm]]'s $O((\|V\|+\|E\|)\log\|V\|)$ — the cost of tolerating negative weights|
|Space|$O(n^2)$ for the full `B` table|Reducible to $O(n)$ (two rows at a time) if negative-cycle detection isn't needed and only final distances matter|

---

# Drawbacks / Constraints

- **Much slower than Dijkstra's.** Only use Bellman-Ford when negative edge weights are actually possible — if all weights are known non-negative, [[Dijkstra's Algorithm]] is strictly better.
- **Detects _that_ a negative cycle exists, but not automatically _which_ vertices/edges form it.** Recovering the actual cycle needs additional bookkeeping (e.g. tracing back predecessor pointers from a vertex whose distance kept improving past $t=n-1$).
- **Preconditions:** the correctness proof above assumes no negative cycle for the "true shortest path" claim to even make sense — the algorithm's real job when a negative cycle _is_ present is just to detect that fact, not to report a (nonexistent) finite shortest distance.

---

# References / Links

- [[Dynamic Programming]]
- [[Dijkstra's Algorithm]]
- [[Shortest Path in a DAG Example|Shortest Path in a DAG]]