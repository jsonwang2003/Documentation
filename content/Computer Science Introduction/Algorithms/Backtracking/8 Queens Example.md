---
aliases:
  - Eight Queens
  - 8 Queens
description: "Classic backtracking problem: place 8 non-attacking queens on an 8x8 chessboard, pruning attacked columns as you go instead of generating full permutations."
tags:
  - algorithm
  - backtracking
  - Examples
---


> [!abstract] 
> Is it possible to place 8 non-attacking queens on an 8x8 chessboard? If so, how do you do it?
> 
> ![[Pasted image 20260710202721.png]]
> 
> - **Category:** Backtracking / Constraint Satisfaction
> - **Input:** An $8\times8$ chessboard
> - **Output:** A boolean — whether a valid placement exists (and, along the way, the placement itself)
> - **Paradigm:** Backtracking (recurse column by column, pruning attacked cells)
> - **Typical use cases:** the canonical constraint-satisfaction example; generalizes to $N$-Queens and to other placement/coloring problems with pairwise-conflict constraints

---

# Problem Specification

- **Instance:** An $8\times8$ board.
- **Solution Format:** A placement of 8 queens on the board.
- **Constraints:** No two queens attack each other (same row, column, or diagonal).
- **Objective / Goal:** This is a decision problem — does _any_ valid placement exist? — not an optimization over many valid solutions.

---

# Candidate Strategies / Approaches

## Brute Force (successively tightened) ✘

Each added assumption shrinks the search space, but all of these remain exhaustive search:

- **8 queens, any of 64 squares, queens distinguishable:** $64^{8} \approx 2.8 \times 10^{14}$
- **No two queens on the same square, queens distinguishable:** $P(64, 8) \approx 1.7 \times 10^{14}$
- **No two queens on the same square, queens indistinguishable:** $\binom{64}{8} \approx 4$ billion
- **+ one queen per row:** $8^{8} = 16$ million
- **+ one queen per row _and_ column:** $8! \approx 40{,}000$

## Backtracking ✔

Rather than generating a full candidate placement and then checking it, build the placement one column at a time, and **prune** the moment a partial placement can no longer be extended to a full solution — never even generating the doomed branches in the first place.

> [!tip] Key Idea 
> Each constraint baked directly into the search space (one queen per row, one per column) shrinks brute force dramatically before backtracking even enters the picture. Backtracking then adds a further layer of savings on top: instead of finishing a full placement and checking it, it detects a dead column — one where every cell is already attacked — as early as possible and abandons that branch immediately.

---

# Pseudocode (Chosen Approach)

```pseudo
	\begin{algorithm}
	\caption{8 Queens}
	\begin{algorithmic}
	\Input $8 \times 8$ chess board $X$ partially filled with integers, and a column number $c$
	\Output A boolean value whether it is possible fit 8 queens in the chessboard
	\Procedure{Queens}{$X, c$}
		\If{$c == 8$}
			\Return \True
        \EndIf
        \ForAll{cells $r$ in column $c$}
	        \If{$r==0$}
		        \State Create $X'$ from $X$ by incrementing each square to the right of column $c$ that is attacked by cell $r$
		        \For{each column $d$ to the right of $c$}
			        \If{$d$ is all non-zero}
				        \Return \False
                    \EndIf
                \EndFor
                \If{$Queens(X', c+1)$ is True}
	                \Return \True
                \EndIf
            \EndIf
        \EndFor
        \Return \False
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!note] Reading This Pseudocode 
> $X$ tracks, per cell, how many already-placed queens currently attack it — so a cell $r$ is safe to place a queen on exactly when $X[r] = 0$. After placing a queen at cell $r$ in column $c$, the algorithm builds $X'$ by incrementing the attack-count of every cell to the right that the new queen threatens (same row and both diagonals). It then immediately checks whether this placement has made some future column **entirely** unsafe (every cell in it has a nonzero attack count) — if so, it returns `False` right away instead of wasting time recursing into a branch that can never succeed. That early-exit check is the "prune" step that makes this backtracking rather than plain exhaustive search.

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`X`|Board of attack-counts|For each cell, how many currently-placed queens attack it; a cell is safe iff its count is 0|
|`c`|Column index|The column currently being filled|
|`X'`|Board (copy)|`X` updated with the attack-counts from the queen just placed at cell `r`|

## Helper Functions / Operations Used

- **Attack-count increment** — for a queen placed at cell $r$ in column $c$, mark every cell to the right that shares its row or either diagonal as newly attacked.
- **Dead-column check** — scan each column to the right of $c$ for any cell still at count 0; if none exists, that column (and hence this whole branch) is unsalvageable.

---

# Proof of Correctness / Optimality.

`Queens(X, c)` explores every placement of a queen in column $c$ that is safe given the queens already placed in columns $0, \dots, c-1$, and recurses to column $c+1$ for each. It returns `True` as soon as some sequence of choices reaches $c=8$ (all columns filled). Because every recursive call only considers cells with attack-count 0 — i.e. genuinely unattacked by every previously-placed queen — no branch the algorithm explores can ever contain an attacking pair. The dead-column pruning check only discards branches that are _provably_ unable to reach a full solution (some later column has zero safe cells left), so no valid solution is ever incorrectly discarded. Since every column has finitely many cells, and the recursion always moves to $c+1$, the search terminates.

---

# Time & Space Complexity Analysis

Backtracking here is still bounded by the same $O(8!)$ ceiling as the "one queen per row and column" brute-force estimate in the worst case — pruning doesn't change that asymptotic ceiling for general $N$-Queens, since an adversarial board layout could in principle still force exploration of a large fraction of that space. What pruning _does_ change dramatically is the **typical-case** runtime: dead columns are usually detected long before 8 queens are placed, so in practice only a small fraction of the $8! \approx 40{,}000$ row/column-valid permutations are ever actually constructed.

---

# Drawbacks / Constraints

- **This is a decision problem, not enumeration.** As written, `Queens` stops at the first valid placement found — finding _all_ solutions (there are 92 for the standard 8-Queens board) requires continuing the search instead of returning immediately.
- **Pruning doesn't lower the worst-case asymptotic bound in general.** The savings shown here are typically an average-case/practical improvement, not a proven better worst-case order for arbitrary $N$-Queens.
- **Still exponential for general $N$.** No known polynomial-time algorithm solves $N$-Queens for arbitrary $N$.

---

# References / Links

- [[Computer Science Introduction/Algorithms/Backtracking/index|Backtracking]]
- [[Sudoku Example|Sudoku]]
- [[Maximal Independent Set Example|Maximal Independent Set]]