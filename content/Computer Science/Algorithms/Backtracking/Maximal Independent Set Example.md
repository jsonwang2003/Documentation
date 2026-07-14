---
description: Backtracking algorithm to find the largest independent set in a graph, refined through three iterations from O(2^n) down to O(1.48^n) by exploiting low-degree vertices.
tags:
  - algorithm
  - backtracking
  - graph-algorithms
aliases:
  - Maximum Independent Set
  - MIS
  - Maximal Independent Set
---
> [!abstract] 
>  Given an undirected graph with nodes representing people, and an edge between $A$ and $B$ if $A$ and $B$ are enemies, find the largest set of people such that no two are enemies. In other words: given an undirected graph, find the largest set of vertices such that no two are connected by an edge.
> 
> - **Category:** Backtracking / Graph Optimization (NP-Hard)
> - **Input:** An undirected graph $G$
> - **Output:** The largest independent set of $G$
> - **Paradigm:** Backtracking — branch on "include this vertex or not," recursively, refined over three iterations below
> - **Typical use cases:** scheduling/conflict-avoidance problems, the canonical example of squeezing a much better exponential base out of brute-force backtracking via case analysis

---

# Problem Specification

- **Instance:** Undirected graph $G$.
- **Solution Format:** Subset of vertices.
- **Constraint:** No two vertices in the subset are connected by an edge.
- **Objective:** Maximize the size of the subset.
- **Goal:** Maximize.

---

# Candidate Strategies / Approaches

**Backtracking approach:** do exhaustive search locally, but use the constraints to simplify the problem as you go.

- **What is a local decision?** Do we pick vertex $A$ or not?
- **What are the possible answers to this decision?** Yes or no.
- **How do the answers affect the future of the problem?**
    - If we pick $A$: recurse on the subgraph $G - (A \cup {A\text{'s neighbors}})$, and add 1 for $A$ (since none of $A$'s neighbors can ever be picked alongside $A$).
    - If we don't pick $A$: recurse on the subgraph $G - A$.

This one branching rule is the seed for all three algorithm versions below — what changes across MIS1 → MIS2 → MIS3 is _when_ it's safe to skip computing one of the two branches entirely.

---

# Algorithm Iterations

## MIS1 — Naive Backtracking

```pseudo
	\begin{algorithm}
	\caption{Maximal Independent Set}
	\begin{algorithmic}
	\Procedure{MIS1}{$G: \text{undirected graph}$}
		\If{$|V| = 0$}
			\Return $\emptyset$
        \EndIf
        \State Pick a vertex $v$
        \State In = $MIS1(G - \{ v \text{ and all of } v \text{'s neighbors} \} \cup \{v\})$
        \State Out = $MIS1(G - \{v\})$
        \If{$|\text{In}| > |\text{Out}|$}
	        \Return In
	    \Else
		    \Return Out
        \EndIf
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Correctness

- **Base Case ($n=0$):** `MIS1` correctly returns the empty set.
- **Inductive Hypothesis:** for $n > 0$, `MIS1` correctly returns the maximum independent set of any graph with $k$ vertices, for $1 \leq k < n$.
- **Argument:** `In` is the best independent set _containing_ $v$; `Out` is the best independent set _not containing_ $v$. Every independent set either contains $v$ or doesn't, so the better of the two is the true maximum independent set of $G$.

### Time Analysis

Both `In` and `Out` cost $T(n-1)$ in the worst case (`In` removes at least $v$ itself — possibly more if $v$ has neighbors, but the _worst_ case for the bound is when $v$ has no neighbors and only 1 vertex is removed; `Out` always removes exactly $v$):

$$ T(n) \leq 2T(n-1) + O(n) \implies T(n) \in O(2^n) $$

### Worst Case for MIS1

When you pick a vertex $v$ with **no neighbors**, the `In` subproblem only decreases by 1 (same as `Out`) — but do we actually need to consider `Out` at all in that case? Shouldn't we just pick $v$? More generally: **if a vertex $v$ has no neighbors, the `In` case is always at least as good as the `Out` case** — including an isolated vertex can never conflict with anything else, so there's no reason to ever leave it out.

---

## MIS2 — Skip `Out` When $\deg(v) = 0$

```pseudo
		\begin{algorithm}
	\caption{Maximal Independent Set}
	\begin{algorithmic}
	\Procedure{MIS2}{$G: \text{undirected graph}$}
		\If{$|V| = 0$}
			\Return $\emptyset$
        \EndIf
        \State Pick a vertex $v$
        \State In = $MIS2(G - \{ v \text{ and all of } v \text{'s neighbors} \} \cup \{v\})$
        \If{$deg(v) == 0$}
	        \Return In
        \EndIf
        \State Out = $MIS2(G - \{v\})$
        \If{$|\text{In}| > |\text{Out}|$}
	        \Return In
	    \Else
		    \Return Out
        \EndIf
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Time Analysis

$$ 
\begin{align*} 
T(n) &= \max(T(n-1), T(n-1) + T(n-2) + O(n))\\
T(n) &= T(n-1) + T(n-2) + O(n)\\
T(n) &\in O(1.62^{n}) \ll O(2^{n})
\end{align*} 
$$

(This is exactly the Fibonacci recurrence — $1.62 \approx \varphi$, the golden ratio.) A huge improvement from the initial approach.

### Worst Case for MIS2

When you pick a vertex $v$ with exactly **one** neighbor, the `In` subproblem only decreases by 2 (removing $v$ and its one neighbor) — but do we actually need to consider `Out` here either? Shouldn't we just pick $v$? More generally: **if a vertex $v$ has one neighbor, the `In` case is always at least as good as the `Out` case.** This one takes a bit more convincing than the degree-0 case.

**Claim:** suppose $v$ is a vertex of $G$ with only one neighbor, $u$. Suppose $OS$ is an independent set that does not include $v$. There is an independent set $OS'$ that does include $v$, with $|OS'| \geq |OS|$.

**Proof:** consider two cases based on whether $OS$ contains $u$ (v's only neighbor):

- **Case 1 — $OS$ does not contain $u$:** let $OS' = OS \cup {v}$. Since $OS$ doesn't contain $v$'s only neighbor, adding $v$ can't create a conflict, so $OS'$ is valid, and $|OS'| = |OS| + 1$.
- **Case 2 — $OS$ contains $u$:** let $OS' = (OS - {u}) \cup {v}$. $OS'$ no longer contains $v$'s only neighbor, so by the validity of $OS$ elsewhere, $OS'$ is valid, and $|OS'| = |OS|$.

Either way, $|OS'| \geq |OS|$, so including $v$ is never worse than excluding it.

---

## MIS3 — Also Skip `Out` When $\deg(v) = 1$

```pseudo
	\begin{algorithm}
	\caption{Maximal Independent Set}
	\begin{algorithmic}
	\Procedure{MIS3}{$G: \text{undirected graph}$}
		\If{$|V| = 0$}
			\Return $\emptyset$
        \EndIf
        \State Pick a vertex $v$
        \State In = $MIS3(G - \{ v \text{ and all of } v \text{'s neighbors} \} \cup \{v\})$
        \If{$deg(v) == 0$ or $deg(v) == 1$}
	        \Return In
        \EndIf
        \State Out = $MIS3(G - \{v\})$
        \If{$|\text{In}| > |\text{Out}|$}
	        \Return In
	    \Else
		    \Return Out
        \EndIf
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Time Analysis

$$ 
\begin{align*} 
T(n) &= T(n-1) + T(n-3) + O(n)\\
&\in O(1.48^{n}) 
\end{align*} 
$$

---

# Time & Space Complexity Analysis

## Summary Across Iterations

|Version|Skips `Out` when|Recurrence|Bound|
|---|---|---|---|
|MIS1|Never|$T(n) = 2T(n-1) + O(n)$|$O(2^n)$|
|MIS2|$\deg(v) = 0$|$T(n) = T(n-1) + T(n-2) + O(n)$|$O(1.62^n)$|
|MIS3|$\deg(v) \in {0, 1}$|$T(n) = T(n-1) + T(n-3) + O(n)$|$O(1.48^n)$|

The pattern is clear: proving that low-degree vertices are always safe to include (never worse than excluding) lets you skip computing an entire recursive branch for them, and handling more and more low-degree cases keeps shrinking the exponential base. The **best known** MIS algorithm is around $O(1.2^n)$, due to Robson, building on Tarjan and Trojanowski — it does much more elaborate case analysis for small-degree vertices, following exactly this same pattern to its logical extreme.

## Space

Each version's space is dominated by recursion depth, which is $O(n)$ in the worst case (one vertex removed per level in the shallowest branch).

---

# Drawbacks / Constraints

- **Still exponential.** Maximal (maximum) Independent Set is NP-hard, so no polynomial-time algorithm is expected for the general case, no matter how much low-degree case analysis is added.
- **Diminishing returns, rapidly increasing complexity.** Going from MIS1 → MIS2 → MIS3 required proving a genuine exchange-argument-style claim just to handle degree-1 vertices; Robson's $O(1.2^n)$ result needs "much more elaborate case analysis" for small-degree vertices, illustrating that each further improvement costs substantially more implementation and proof effort for a shrinking marginal gain.
- **Only helps low-degree vertices.** This whole line of refinement exploits the fact that a low-degree vertex barely shrinks the graph on the `Out` branch — it doesn't directly help with dense graphs where most vertices have high degree.

> [!tip] Toward Dynamic Programming 
> Each `MIS` call recurses on an induced subgraph obtained by deleting a small set of vertices — but which specific subgraph you get depends on the sequence of choices made to get there, so in general graphs there's no small, reusable set of subproblems to memoize. This is exactly the boundary case that makes [[Computer Science/Algorithms/Dynamic Programming/index|Dynamic Programming]] work beautifully on some backtracking problems (where subproblems collapse to a polynomial-size, reusable set — e.g. "the best solution using only elements $1..i$") but not on others like general-graph MIS, where the reachable subgraphs don't collapse that way.
> 
> **The exception that proves the rule:** restrict the input to a **tree**, and the subproblems _do_ collapse — each vertex's subtree is a clean, reusable subproblem, since subtrees never overlap. See [[Maximum Independent Set in Trees]] for the resulting $O(n)$ DP solution, a direct contrast to this note's exponential general-graph result.

---

# References / Links

- [[Computer Science/Algorithms/Backtracking/index|Backtracking]]
- [[Computer Science/Algorithms/Dynamic Programming/index|Dynamic Programming]]