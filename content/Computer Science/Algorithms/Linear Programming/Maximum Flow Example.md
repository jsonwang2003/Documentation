---
description: Find the maximum flow from source to sink in a capacitated directed graph — solved via Ford-Fulkerson's augmenting-path method, which also proves the Max-Flow Min-Cut theorem along the way.
tags:
  - algorithm
  - linear-programming
  - graph-algorithms
aliases:
  - Maximum Flow
  - Network Flow
  - Max Flow
  - Ford-Fulkerson
---
> [!abstract] 
> Given a directed graph where edges have capacities, and two special vertices — a source $s$ and a sink $t$ — how much "flow" can be pushed from $s$ to $t$ without exceeding any edge's capacity?
> 
> ![[Pasted image 20260712225413.png]]
> 
> - **Category:** Network Flow / Combinatorial Optimization (also formulable as [[Computer Science/Algorithms/Linear Programming/index|Linear Programming]])
> - **Input:** Directed graph with non-negative edge capacities $c(e)$; source $s$, sink $t$
> - **Output:** An assignment of flow values $f(e)$ to every edge, maximizing total flow
> - **Paradigm:** Augmenting-path method (Ford-Fulkerson), improving a flow via a residual graph
> - **Typical use cases:** bipartite matching, project/task selection, image segmentation, traffic and logistics routing — anything shaped like "maximize throughput subject to capacity limits"

---

# Problem Specification

- **Instance:** Directed graph with non-negative edge weights called **capacities**. Two vertices $s$ (source), $t$ (sink).
- **Solution Format:** An assignment of non-negative values (flow) to each edge.
- **Constraints:**
    - **Capacity:** for each edge $e$, $f(e) \leq c(e)$.
    - **Conservation:** for each vertex $u \neq s,t$, $\sum_{e \in in(u)} f(e) = \sum_{e' \in out(u)} f(e')$ (flow in equals flow out).
- **Objective:** total flow out of $s$ (equivalently, total flow into $t$).
- **Goal:** Maximize.

---

# Candidate Strategies / Approaches

## Linear Programming (Simplex) ✔ — works, but not the specialized choice

Since every constraint above (capacity, conservation) and the objective (total flow) are linear, Max Flow is technically solvable directly as an instance of [[Computer Science/Algorithms/Linear Programming/index|Linear Programming]]. But the problem's graph structure supports a more specialized, more efficient combinatorial algorithm instead.

## Ford-Fulkerson Method ✔ (chosen)

Start with a trivial solution, and keep trying to make it better.

- **Trivial flow:** zero flow along every edge.
- **Better than zero:** any path of positive-capacity edges from $s$ to $t$.
- **How much better?** Send the _minimum_ capacity of any edge along that path (the bottleneck) — that's how much extra flow the whole path can carry.

---

# The Ford-Fulkerson Method

## The Residual Graph

**Ford-Fulkerson's insight:** represent the problem of _improving_ a flow as another flow problem, on a **residual graph**. If $f(e)$ is the current flow on edge $e$ and $c(e)$ its capacity, the residual graph changes that edge's capacity to $c(e) - f(e)$, and adds $f(e)$ to the capacity of the _reverse_ edge (representing the option to "undo" flow already sent).

## Key Observation

Let $f$ be any flow in the original graph, and $f'$ be any flow in the residual network with respect to $f$. Then $f + f'$ is a valid flow in the original network, and:

$$ Flow(f+f') = Flow(f) + Flow(f') $$

So we can keep finding flow in the residual network, update the residual graph, and repeat — until we can't increase the flow any further.

## Termination and the Min Cut

We stop when there's no path from $s$ to $t$ left in the residual graph. Let $S$ be the set of all vertices reachable from $s$ in the residual graph at that point, and $T = V - S$ the unreachable vertices ($t \in T$, since we just said there's no path to it).

Every edge $e=(u,v)$ with $u \in S$ and $v \in T$ can't be in the residual graph (or $v$ would be reachable too) — meaning every such edge is being used at **full capacity**. Let $Cut(S,T)$ be the total capacity of all such crossing edges. Then, at termination:

$$ Flow(f) = Cut(S,T) $$

---

# Pseudocode (Chosen Approach)

```pseudo
	\begin{algorithm}
	\caption{Network Flow}
	\begin{algorithmic}
	\Procedure{FF}{$G, c, s, t$}
		\While{there is a path in the residual graph}
			\State Find a path in the residual graph (DFS) from $s$ to $t$
			\State Augment the flow along every edge in this path
			\State Create new residual graph
        \EndWhile
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`f(e)`|Number, per edge|Current flow assigned to edge $e$|
|Residual graph|Graph|Same vertices, with each edge's remaining capacity $c(e)-f(e)$, plus a reverse edge of capacity $f(e)$ to allow "undoing" flow|

## Helper Functions / Operations Used

- **Find an $s$-$t$ path in the residual graph** — [[Depth First Search (DFS)]] (as specified), $O(|V|+|E|)$ per call.
- **Augment along the path** — push flow equal to the path's bottleneck (minimum residual capacity along it); $O(\text{path length})$.

---

# Proof of Correctness / Optimality

**Weak duality (needed first):** for _any_ flow $f$ and _any_ $s$-$t$ cut $(S,T)$, $Flow(f) \leq Cut(S,T)$. Every unit of flow from $s$ to $t$ must cross from $S$ to $T$ somewhere, and no edge can carry more than its capacity — so total flow can never exceed any cut's total capacity.

**What Ford-Fulkerson shows at termination:** the algorithm stops precisely when no augmenting path remains. At that point, taking $S$ = vertices reachable from $s$ in the residual graph and $T = V-S$, every crossing edge is fully saturated, giving $Flow(f) = Cut(S,T)$ exactly (shown above).

**Putting them together — the Max-Flow Min-Cut Theorem:** since $Flow(f) \leq Cut(S,T)$ holds for _every_ cut, and Ford-Fulkerson exhibits a flow $f$ and a specific cut $(S,T)$ with equality, two things follow simultaneously:

- $f$ **is** a maximum flow — no flow can exceed $Cut(S,T)$, and $f$ already equals it.
- $(S,T)$ **is** a minimum cut — no cut can be smaller than $Flow(f)$, and $(S,T)$ already equals it.

So Ford-Fulkerson doesn't just find _a_ flow it can't improve — it provably finds _the_ maximum flow, with a matching minimum cut as a certificate. 

---

# Time & Space Complexity Analysis

## General Case

At most $O(W|V|)$ iterations of the outer loop, each costing $O(|E|)$ time (one DFS), giving:

$$ O(W|V||E|) $$

> [!note] Where does $W|V|$ come from? 
> $W$ (not explicitly defined in the source) is the maximum edge capacity. Each augmenting path increases total flow by at least 1 if capacities are integers, and the maximum possible flow value is bounded by (the number of edges leaving $s$) $\times$ (max capacity) $\leq O(|V| \cdot W)$ in the worst case — giving at most $O(|V|W)$ iterations before the flow can't increase any further, each costing $O(|E|)$ for the DFS.

| |Complexity|Notes|
|---|---|---|
|Time|$O(W\|V\|E\|)$|Pseudo-polynomial — see Drawbacks below|
|Space|$O(\|V\|+\|E\|)$|The residual graph, same size as the original plus reverse edges|

---

# Drawbacks / Constraints

- **Pseudo-polynomial runtime.** $O(W|V||E|)$ depends on the _numeric value_ of capacities, not just the graph's size — with large capacities, this can be slow, matching the same caveat seen in [[The Knapsack Problem Example|Knapsack Problem]]'s $O(nC)$.
- **Can fail to terminate with irrational capacities.** If edge capacities aren't required to be rational, Ford-Fulkerson's augmenting-path process can (in pathological cases) run forever without converging to the max flow at all.
- **Path choice matters.** Using DFS to find _any_ augmenting path (as specified) can be much slower in practice than always choosing the _shortest_ augmenting path — the Edmonds-Karp variant (BFS instead of DFS) guarantees a strongly polynomial $O(VE^2)$ bound, independent of capacity values, fixing both the pseudo-polynomial and non-termination issues above.

---

# References / Links

- [[Computer Science/Algorithms/Linear Programming/index|Linear Programming]]
- [[Graph Reachability]]
- [[Depth First Search (DFS)]]