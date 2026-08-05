---
description: Two general strategies for deriving a new algorithm from an existing one — modifying its internals, or reducing the new problem to reuse it unchanged — worked through with the Max Bandwidth problem.
tags:
  - algorithm
  - concepts
aliases:
  - Modification vs Reduction
  - Reduction (Algorithms)
  - Algorithm Design Techniques
---
> [!abstract] Overview 
> When you have a new problem to solve and you already have an algorithm that solves a _related_ problem, there are two general strategies for building an algorithm out of it:
> 
> 1. **Modification** — change the existing algorithm's internal logic so it directly tracks/solves the new problem.
> 2. **Reduction** — leave the existing algorithm untouched, and instead transform the _input_ of the new problem into an instance of the old one, calling the existing algorithm as a subroutine.
> 
> This note works through both strategies using the **Max Bandwidth Problem** as a running example.

> [!Question] Driving Example: Max Bandwidth Problem Graph represents a network, with edges representing communication links. Edge weights are the bandwidth of the link — what is the largest bandwidth of a path from $A$ to $H$?
> 
> ![[Pasted image 20260629201804.png]]

---

# Defining the Problem Formally

Before picking a strategy, it helps to pin down the problem precisely. Any optimization problem can be broken into 4 parts:

1. **Instance (Input):** what you're given.
2. **Solution Type (Output):** the shape of the answer.
3. **Constraints:** what must be true of a valid answer.
4. **Objective:** what you're optimizing over all valid answers.

**Driving Example — Max Bandwidth, formalized:**

1. **Instance:** Directed graph $G = (V, E)$ with positive edge weights $w(e)$, two vertices $s, t \in V$.
2. **Solution Type:** A sequence of edges.
3. **Constraints:** The sequence of edges is a path $p$ from $s$ to $t$ in $G$.
4. **Objective:** Over all possible paths $p$ between $s$ and $t$, find one that maximizes the bandwidth of a path:

$$BW(p) = \underset{ e \in p }{ min } \ w(e)$$

---

# Approach 1: Algorithm Modification

Take an existing algorithm that solves a structurally similar problem, and change what it tracks internally so it solves the new problem instead — e.g. starting from [[Graph Reachability#Graph Search Algorithm|Graph Search]] and having it track a new quantity per vertex instead of just visited/unvisited.

> [!Error] Limitations
> 
> 1. Runtime is no longer guaranteed to match the original — it must be **reanalyzed** from scratch.
> 2. The modified algorithm is **not automatically correct** just because the original was — it must be **reproven**, in full, from the ground up.

## Driving Example: Max Bandwidth via Modification

Use the basic structure of [[Graph Reachability#Graph Search Algorithm|Graph Search]], and for each vertex $v$, keep track of the max bandwidth to $v$ found so far. Then move a vertex into $F$ only if its max bandwidth has improved.

```pseudo
	\begin{algorithm}
	\caption{Max Bandwidth Modify Algorithm Approach}
	\begin{algorithmic}
		\Procedure{MaxBand1}{$G: \text{directed graph}, s, t$}
			\State Initialize $X$ = emtpy, $F = \{s\}$
			\State B($v$) = 0 for $v \in V$
			\State B($s$) = $\infty$
			\While{$F$ is not empty}
				\State Pick $v$ in $F$
				\For{each neighbor $u$ of $v$}
					\State $m$ = min(B($v$), w($v, u$))
					\If{$m >$ B($u$)}
						\State move $u$ to $F$
						\State B($u$) = $m$
                    \EndIf
                \EndFor
                \State move $v$ from $F$ to $X$
            \EndWhile
            \Return B($t$)
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

**Variables:** $B(v)$ — the bandwidth of the best path found so far from $s$ to $v$.

### Proof of Correctness

Note this whole proof is only necessary _because_ this is a modification — nothing about Graph Search's original correctness proof transfers over automatically.

**Claim:** At the end of the algorithm, $B(v)$ is the maximum bandwidth from $s$ to $v$, for all vertices $v \in V$.

**Part 1 — $B(v)$ is always _achievable_:** for all $v \in V$, there is a path $p$ from $s$ to $v$ such that $BW(p) = B(v)$.

- **Loop Invariant:** after every iteration, for all $v$, there is a path from $s$ to $v$ with $BW(p) = B(v)$.
- **Base Case:** before the first iteration, $B(s) = \infty$ and $B(v) = 0$ for every other vertex.
- **Inductive Hypothesis:** assume the claim holds after $t$ iterations.
- **Inductive Step:** pick $v$ in $F$, let $u$ be a neighbor of $v$, and $m = \min(B(v), w(v,u))$.
    - **Case 1:** $m \leq B(u)$ — $B(u)$ doesn't change.
    - **Case 2:** $B(u) < m$ — $B(u)$ updates to $m$.
    - Either way, there still exists a path from $s$ to $u$ with bandwidth exactly $B(u)$.
- So the loop invariant holds after every iteration, including the last — meaning by the end, every vertex $v$ has a real path from $s$ to $v$ achieving bandwidth $B(v)$.

**Part 2 — $B(v)$ is never an _underestimate_:** for all $v \in V$, $B(v)$ is the _maximum_ bandwidth among all paths from $s$ to $v$ (not just some achievable value).

- Suppose by contradiction there's a vertex $v$ with some path $p$ from $s$ to $v$ where $BW(p) > B(v)$. Let $b = BW(p)$.
- Let $y$ be the first vertex along $p$ where $B(y) < b$, and $z$ the vertex right before $y$ on $p$ (so $B(z) \geq b$).
- Since $p$ has bandwidth $b$, every edge on it — including $(z,y)$ — has weight $\geq b$, so $w(z,y) \geq b$.
- When $z$ is processed, the algorithm computes $m = \min(B(z), w(z,y)) \geq b$. Since $B(y) < b \leq m$, the algorithm updates $B(y)$ to $m \geq b$.
- This contradicts the assumption that $B(y) < b$ at the end of the algorithm. $\blacksquare$

---

# Approach 2: Reduction

> [!abstract] What is a Reduction? 
> Instead of modifying an existing algorithm, we modify the **input** so we can use the existing algorithm as a **subroutine**. We map instances of one problem to instances of another, then use any known algorithm for that second problem as a subroutine to build an algorithm for the first — the existing algorithm's correctness and runtime proofs carry over **unchanged**.

## Reduction From a Decision Version

A useful general pattern: to relate a decision problem to an optimization problem, look at the **decision version** of the optimization problem instead of the optimization problem itself. A decision version asks a yes/no question ("is there a solution at least this good?") rather than "find the best solution" — and it's often much easier to reduce to something else.

## Driving Example: Max Bandwidth via Reduction

**Decision Version of Max Bandwidth:** Given $G, s, t, M$, is there a path of bandwidth $M$ or better from $s$ to $t$?

```pseudo
	\begin{algorithm}
	\caption{Max Bandwidth Reduction Approach}
	\begin{algorithmic}
		\Procedure{MaxBandDecision}{$G, s, t, M$}
			\State Construct $G_M$ by removing all edges less than $M$ from $G$
			\State Run graphSearch($G_M, s$)
			\If{$t$ is visited}
				\Return \True
			\Else
				\Return \False
            \EndIf
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

The transformation itself — building $G_M$ — is the _entire_ reduction. [[Graph Reachability#Graph Search Algorithm|Graph Search]] runs completely unmodified on $G_M$.

### Proof of Correctness

Note how much smaller this proof is compared to the Modification approach — we only need to prove the reduction step correct, since `graphSearch` itself is already proven correct elsewhere.

**Direction 1 — if there's a path in $G$ with bandwidth $\geq M$, the algorithm returns TRUE:** Suppose path $p$ in $G$ from $s$ to $t$ has bandwidth at least $M$. Then every edge in $p$ has weight $\geq M$, so $p$ survives entirely in $G_M$ (no edge of $p$ gets removed). So `graphSearch` visits $t$, and the algorithm outputs TRUE.

**Direction 2 — if there's no such path, the algorithm returns FALSE:** Restated as the contrapositive: if the algorithm returns TRUE, then there is a path from $s$ to $t$ in $G$ with bandwidth at least $M$. Suppose the algorithm returns TRUE. Then there's a path $p$ in $G_M$ from $s$ to $t$. Every edge in $G_M$ has weight $\geq M$ by construction, so $p$ is also a path in $G$ where every edge weight is $\geq M$ — meaning $BW(p) \geq M$ in $G$ as well.

### Time Analysis

Let $n = |V|, m = |E|$.

- Time to construct $G_M$: $O(n + m)$
- Time to run `graphSearch` (already analyzed, unchanged): $O(n + m)$

**Total Time:** $O(n + m)$ — note this reuses `graphSearch`'s existing runtime bound rather than re-deriving it.

To solve the full optimization problem (not just the decision version), binary search over candidate values of $M$ (e.g. the distinct edge weights) and call `MaxBandDecision` at each — this reuses the decision procedure as a subroutine without needing a new correctness proof for the search itself.

---

# Comparing the Two Approaches

| |Modification|Reduction|
|---|---|---|
|**What changes**|The existing algorithm's internal logic|The input, via a transformation step; the existing algorithm runs unchanged|
|**Correctness burden**|Reprove the _entire_ modified algorithm from scratch|Only prove the transformation step correct — the existing algorithm's proof carries over|
|**Runtime burden**|Reanalyze the runtime of the modified algorithm|Runtime = cost of the transformation + the existing algorithm's already-known runtime|
|**Reusability**|A one-off algorithm specific to this problem|Any correct, analyzed algorithm for the target problem can be swapped in|
|**Max Bandwidth outcome**|New proof (2 parts, induction) + new complexity argument|Reduction proof (2 directions) + `graphSearch`'s complexity reused as-is|

---

# When to Use Which

- **Prefer Reduction** when a well-analyzed algorithm already exists for a problem your new problem's _decision version_ (or some other variant) can be mapped onto — you inherit its correctness and runtime for free, paying only for the transformation.
- **Reach for Modification** when no existing algorithm is close enough to reduce to, or when the transformation needed for a reduction would itself be as expensive or as hard to prove correct as just modifying the algorithm directly.
- Either way, the goal is the same: get to a [[Levels of Algorithm Design#Low Level Design|fully specified, analyzed algorithm]] while doing as little _new_ correctness/runtime work as possible.

---

# References / Links

- [[Graph Reachability]]
- [[Levels of Algorithm Design]]