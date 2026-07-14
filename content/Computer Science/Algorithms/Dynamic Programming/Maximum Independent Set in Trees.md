---
description: Weighted maximum independent set restricted to trees — unlike general graphs, this admits a linear-time DP solution by tracking two states (include/exclude) per subtree.
tags:
  - algorithm
  - dynamic-programming
  - graph-algorithms
aliases:
  - MIS in Trees
  - Tree Independent Set
---
> [!abstract] 
>  Consider the problem of finding the maximum independent set in trees:
> 
> ![[Pasted image 20260712222638.png]]
> 
> - **Category:** Dynamic Programming / Graph Optimization (restricted to trees)
> - **Input:** A tree $T$ with vertex weights $w(v)$
> - **Output:** The maximum-weight independent set of $T$
> - **Paradigm:** Dynamic Programming, bottom-up from leaves to root
> - **Typical use cases:** the go-to example showing that restricting a graph to a tree can turn an NP-hard general problem — see [[Maximal Independent Set Example|Maximal Independent Set]] on general graphs — into one solvable in linear time

---

# Problem Specification

- **Instance:** A tree $T$ with a weight $w(v)$ on each vertex.
- **Solution Format:** A subset of vertices $S$.
- **Constraints:** No two vertices in $S$ are connected by an edge.
- **Objective:** $\sum_{v \in S} w(v)$.
- **Goal:** Maximize.

---

# Candidate Strategies / Approaches

## General-Graph Backtracking ✘ (correct, but wasteful here)

The [[Maximal Independent Set Example|Maximal Independent Set]] backtracking approach (MIS1 → MIS2 → MIS3) works on _any_ graph, including trees — but even its best refinement is exponential ($O(1.48^n)$, or Robson's $O(1.2^n)$). Trees have far more exploitable structure than that approach uses.

## Dynamic Programming ✔

> [!tip] Key Idea 
> A tree has no cycles, so removing a vertex splits it cleanly into independent subtrees that share no vertices — this is exactly the structural property that turns an exponential general-graph search into a polynomial one (the same reason [[Shortest Path in a DAG Example|Shortest Path in a DAG]] beats general shortest-path search). For each vertex $k$, track **two** answers instead of one: the best independent set of the subtree rooted at $k$ that _includes_ $k$, and the best one that _excludes_ $k$. Two answers are necessary because whether $k$'s parent can safely include itself depends on whether $k$ was included.

---

# Dynamic Programming Solution

## 1. Subproblems

Let $M[k] = (IN, OUT)$, where $IN$ is the weight of the maximum independent set of the subtree hanging from $k$ **including** vertex $k$, and $OUT$ is the weight of the maximum independent set of the subtree hanging from $k$ **excluding** $k$.

## 2. Base Case

If $v$ is a leaf, $M[k] = (w(v), 0)$ — a leaf's only two options are "take just itself" (weight $w(v)$) or "take nothing" (weight $0$).

## 3. Recursion

To compute $M[k] = (IN_k, OUT_k)$, we need to know $M[c]$ for every child $c$ of $k$:

![[Pasted image 20260712222902.png]]

$$ IN_k = w(k) + \sum_{c} OUT(c) \qquad\qquad OUT_k = \sum_{c} \max(IN(c), OUT(c)) $$

- **$IN_k$:** if $k$ is included, none of $k$'s children can be included (they're adjacent to $k$), so each child subtree must use its _excluding_ answer — sum $OUT(c)$ over all children $c$, plus $k$'s own weight.
- **$OUT_k$:** if $k$ is excluded, each child subtree is free to independently pick whichever of its two answers is larger, since there's no longer any constraint coming from $k$.

## 4. Ordering of the Subproblems

Order by layers — start at the bottom (leaves) and work up to the root. Equivalently, a **post-order traversal**: finish computing $M[c]$ for every child $c$ before computing $M[k]$.

## 5. Output

$$ \max(IN_{root}, OUT_{root}) $$

---

# Pseudocode (Chosen Approach)

```pseudo
	\begin{algorithm}
	\caption{Tree MIS}
	\begin{algorithmic}
	\Input Tree $T$ rooted at $r$, vertex weights $w$
	\Output Weight of the maximum independent set of $T$
	\Procedure{TreeMIS}{$k$}
		\If{$k$ is a leaf}
			\Return $(w(k), 0)$
        \EndIf
        \State $inSum = w(k)$
        \State $outSum = 0$
        \ForAll{children $c$ of $k$}
	        \State $(IN_c, OUT_c) = $ TreeMIS($c$)
	        \State $inSum = inSum + OUT_c$
	        \State $outSum = outSum + \max(IN_c, OUT_c)$
        \EndFor
        \Return $(inSum, outSum)$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

Final answer: $\max(IN_r, OUT_r)$ where $(IN_r, OUT_r) = \text{TreeMIS}(r)$.

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`M[k]` / `(IN_k, OUT_k)`|Pair of numbers, per vertex|The two subproblem answers for the subtree rooted at `k`|
|`r`|Root vertex|Chosen arbitrarily if the input tree is unrooted — any vertex works as root|

## Helper Functions / Operations Used

- **Children lookup** — for a rooted tree, each vertex's children are simply its tree-neighbors other than its parent.
- **Post-order recursion** — the natural way to guarantee every child's $(IN,OUT)$ is computed before its parent's.

---

# Proof of Correctness

**Claim:** $M[k] = (IN_k, OUT_k)$ as computed above equals the true maximum-weight independent set of the subtree rooted at $k$, including or excluding $k$ respectively.

- **Base case:** a leaf has exactly two options — include itself (weight $w(v)$, valid since a single vertex trivially has no internal conflicts) or include nothing (weight $0$) — so $M[k] = (w(v), 0)$ is correct.
- **Inductive Hypothesis:** $M[c]$ is correct for every child $c$ of $k$ (guaranteed by the post-order ordering).
- **Inductive Step:**
    - **$IN_k$:** if $k$ is in the chosen set, no child of $k$ can be (they're each adjacent to $k$), so every child subtree must contribute its _best excluding_ answer. Since different children's subtrees share no vertices (tree structure), these choices don't interact — the total is exactly $w(k) + \sum_c OUT(c)$, correct by the Inductive Hypothesis.
    - **$OUT_k$:** if $k$ is not in the chosen set, each child subtree is unconstrained by $k$ and can independently pick whichever of its two options is better — again, no interaction between different children's subtrees, so the total is $\sum_c \max(IN(c), OUT(c))$, correct by the Inductive Hypothesis.
- Since every valid independent set of the subtree at $k$ either includes $k$ or doesn't, $M[k]$ correctly captures the best of each case. At the root, $\max(IN_{root}, OUT_{root})$ correctly picks the better of the two, giving the true maximum-weight independent set of the whole tree. $\blacksquare$

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(n)$|Each vertex is processed exactly once; the work done at vertex $k$ is $O(\deg(k))$ (one constant-time step per child), and $\sum_v \deg(v) = O(n)$ for a tree ($n-1$ edges total)|
|Space|$O(n)$|The `M` table stores 2 values per vertex; the recursion stack adds up to $O(\text{height})$, which is $O(n)$ worst case (a path-shaped tree) or $O(\log n)$ for a balanced tree|

## Best / Worst / Average Case

- **Best / Worst / Average case:** all $O(n)$ — every vertex is visited exactly once regardless of the tree's shape or the specific weights.

---

# Drawbacks / Constraints

- **Only works on trees.** The correctness argument relies entirely on subtrees sharing no vertices — the moment cycles exist (general graphs), this clean separation breaks down, which is exactly why [[Maximal Independent Set Example|Maximal Independent Set]] on general graphs stays exponential even after heavy refinement.
- **Requires a rooted tree.** If given an unrooted tree, pick any vertex as root first (a single $O(n)$ traversal) — the choice of root doesn't affect the final answer, only how the recursion is organized.
- **Recursive implementation risks stack overflow on very unbalanced trees** (e.g. a long path) — an explicit iterative post-order traversal (using an explicit stack) avoids this if tree height could be large.

---

# References / Links

- [[Dynamic Programming]]
- [[Maximal Independent Set Example|Maximal Independent Set]]
- [[Computer Science/Algorithms/Backtracking/index|Backtracking]]