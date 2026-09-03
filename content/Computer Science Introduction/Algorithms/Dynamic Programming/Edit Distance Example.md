---
description: Minimum number of insertions, deletions, and substitutions needed to transform one string into another — computed via a 2D dynamic programming table, equivalently viewable as shortest path in a DAG.
aliases:
  - Levenshtein Distance
  - Edit Distance
tags:
  - algorithm
  - dynamic-programming
  - String
  - Examples
---
> [!abstract] Abstract 
> Given two words (strings), how can we define a notion of "closeness"?
> 
> > [!Info] Definition We can keep track of how many "changes" we need to change one word into another. The changes can be:
> > 
> > - insertion
> > - deletion
> > - substitution
> 
> For example, lining up the words PELICAN and OSTRICH:
> 
> ![[Pasted image 20260711194139.png]]
> 
> this alignment uses 7 changes, but it is **not** the cheapest.
> 
> - **Category:** Dynamic Programming / String Processing
> - **Input:** Two strings $x[1\dots n]$ and $y[1\dots m]$
> - **Output:** The minimum number of edits (insert/delete/substitute) to transform $x$ into $y$
> - **Paradigm:** Dynamic Programming — equivalently, shortest path in a DAG (see below)
> - **Typical use cases:** spell checking/correction, diff tools, DNA sequence alignment, fuzzy string matching

---

# Problem Specification

- **Instance:** Two strings $x[1\dots n]$, $y[1\dots m]$ (WLOG $n \leq m$).
- **Solution Format:** A sequence of edit operations (insert, delete, substitute) that transforms $x$ into $y$.
- **Constraints:** The sequence must actually transform $x$ into $y$ exactly.
- **Objective:** The number of edit operations used.
- **Goal:** Minimize.

---

# Candidate Strategies / Approaches

## Brute Force ✘

Try all possible alignments/combinations and find the minimum cost among them. A lower bound on how many combinations exist: each of (at least) the first $n$ columns of an alignment table could independently be one of three things (delete, insert, or substitute/match) — so there are at least $3^n$ different combinations. Exponential.

## Dynamic Programming ✔

Define $E[i,j]$ = the edit distance to transform $x_1 \dots x_i$ into $y_1 \dots y_j$. Solve smallest prefixes first, reusing each answer.

---

# Dynamic Programming Solution

## 1. Define Subproblems

Let $E[i, j]$ be the edit distance to transform $x_1 \dots x_i$ into $y_1 \dots y_j$ (the minimum number of changes).

## 2. Base Cases

When the first word is empty, the edit distance is the length of the second word; when the second word is empty, it's the length of the first word:

$$ 
E[0, j] = j \qquad E[i, 0] = i 
$$

## 3. Express Recursively

What does the last column of the alignment table look like? Three cases:

**Case 1 — Delete** $x_i$:

|$x_1\dots x_{i-1}$|$x_i$|
|---|---|
|$y_1\dots y_j$|―|

$$
E[i, j] = 1 + E[i-1, j]
$$

**Case 2 — Insert** $y_j$:

|$x_1\dots x_i$|―|
|---|---|
|$y_1\dots y_{j-1}$|$y_j$|

$$
E[i,j] = 1 + E[i, j-1]
$$

**Case 3 — Substitute** (or match, if equal):

|$x_1\dots x_{i-1}$|$x_i$|
|---|---|
|$y_1\dots y_{j-1}$|$y_j$|

$$
E[i, j] = \begin{cases} 1 + E[i-1, j-1] &\text{if } x_i \neq y_j \\ 0 + E[i-1, j-1] &\text{if } x_i = y_j\end{cases}
$$

Since we don't know which case is cheapest, take the minimum of all three.

## 4. Ordering

To calculate $E[i,j]$, we need $E[i-1, j]$, $E[i, j-1]$, and $E[i-1, j-1]$ — all already computed if we visit cells **left to right through rows, top to bottom**.

![[Pasted image 20260711210145.png]]

## 5. Iterative Algorithm

```pseudo
	\begin{algorithm}
	\caption{Edit Distance}
	\begin{algorithmic}
	\Procedure{EditDist}{$x[1\dots n], y[1 \dots m]$}
		\For{$i$ from $1$ to $n$}
			\State $E[i, 0] = i$
        \EndFor
        \For{$j$ from $1$ to $m$}
	        \State $E[0, j] = j$
        \EndFor
        \For{$i$ from $1$ to $n$}
	        \For{$j$ from $1$ to $m$}
		        \If{$x[i] == y[j]$}
			        \State $E[i, j] = \min(1 + E[i-1, j], 1 + E[i, j-1], 0 + E[i-1, j-1])$
                \EndIf
                \If{$x[i] \neq y[j]$}
	                \State $E[i, j] = \min(1 + E[i-1, j], 1 + E[i, j-1], 1 + E[i-1, j-1])$
                \EndIf 
            \EndFor
        \EndFor
        \Return $E[n, m]$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## 6. Final Output

$$ 
E[n, m] 
$$

![[Pasted image 20260711210739.png]]

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`E`|2D array, $(n+1)\times(m+1)$|`E[i][j]` = edit distance between $x_1\dots x_i$ and $y_1\dots y_j$|

## Helper Functions / Operations Used

- **Character comparison `x[i] == y[j]`** — $O(1)$.

---

# Edit Distance as a DAG

This table can be viewed as a [[Graph Reachability#Directed Acyclic Graphs (DAG)|DAG]]:

![[Pasted image 20260711211016.png]]

This graph has $|V| = nm$ vertices and $|E| \approx 3nm$ edges (each cell has up to 3 incoming edges — from the delete, insert, and substitute/match cases above). The goal becomes: find the length of the shortest path from the top-left corner $(0,0)$ to the bottom-right corner $(n,m)$.

We could use [[Dijkstra's Algorithm]] for a runtime of:

$$ 
O(nm\log(nm)) 
$$

But there's a faster way, using the fact that this graph is specifically a DAG — see [[Shortest Path in a DAG Example]].

> [!tip] Why This Connection Matters 
> The DP recurrence above **is** a shortest-path-in-a-DAG algorithm, just described in array terms instead of graph terms: filling `E` row by row, left to right is exactly a topological order of this DAG, and each cell's `min` over three incoming edges is exactly the DAG shortest-path relaxation step. That's why the DP solution's own $O(nm)$ runtime already beats Dijkstra's $O(nm\log(nm))$ — it's implicitly using the DAG structure (no comparisons/priority queue needed) rather than Dijkstra's general-graph machinery.

---

# Proof of Correctness / Optimality

**Claim:** $E[i,j]$ equals the true minimum edit distance between $x_1\dots x_i$ and $y_1\dots y_j$.

- **Base cases:** $E[0,j] = j$ (transform empty string to $y_1\dots y_j$ by $j$ insertions) and $E[i,0] = i$ (transform $x_1\dots x_i$ to empty by $i$ deletions) are both correct by inspection — there's no cheaper way to create or destroy $k$ characters than $k$ single-character operations.
- **Inductive Hypothesis:** every cell visited before $(i,j)$ in the row-by-row, top-to-bottom order — in particular $E[i-1,j]$, $E[i,j-1]$, $E[i-1,j-1]$ — is correct.
- **Inductive Step:** the last operation in any optimal transformation of $x_1\dots x_i$ into $y_1\dots y_j$ must be one of exactly three things: delete $x_i$, insert $y_j$, or substitute/match $x_i$ with $y_j$. Each case's cost is $1$ (or $0$ for a free match) plus the cost of optimally solving the remaining smaller prefix problem — which is correct by the Inductive Hypothesis. Since $E[i,j]$ takes the minimum over exactly these three cases, and every valid transformation's last step falls into one of them, $E[i,j]$ is the true minimum.

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(nm)$|One $O(1)$ computation per cell, $(n+1)\times(m+1)$ cells|
|Space|$O(nm)$|The full table; reducible to $O(\min(n,m))$ if only the distance value is needed (keep just the current and previous row), at the cost of losing the ability to reconstruct the actual edit sequence|

## Best / Worst / Average Case

- **Best / Worst / Average case:** all $O(nm)$ — every cell is filled regardless of how similar or different the two strings are.

---

# Drawbacks / Constraints

- **Doesn't directly output the edit sequence**, only its length — recovering the actual operations (like [[String Reconstruction Example|String Reconstruction]]'s `prev` pointers) requires tracing back through the table from $(n,m)$ to $(0,0)$, following whichever case achieved the minimum at each step.
- **$O(nm)$ space can be heavy** for very long strings if the full table is kept; see the space-reduction note above when only the distance value is needed.
- **All operations cost the same (1 each) here.** A weighted variant (e.g. substitutions costing more than insertions, or cost depending on which characters are involved) is a natural extension — same recurrence shape, different constants per case.

---

# References / Links

- [[Computer Science Introduction/Algorithms/Dynamic Programming/index|Dynamic Programming]]
- [[Shortest Path in a DAG Example]]
- [[Dijkstra's Algorithm]]
- [[Graph Reachability]]
- [[String Reconstruction Example|String Reconstruction]]