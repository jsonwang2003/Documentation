---
description: "Backtracking approach to solving Sudoku: fill the least-constrained cell with the smallest valid number, recursing and backtracking on dead ends."
tags:
  - algorithm
  - backtracking
aliases:
  - Sudoku
---

> [!abstract] Abstract 
> Given a partially-filled Sudoku puzzle, find a completion satisfying the usual Sudoku constraints.
> 
> ![[Pasted image 20260710205931.png]]
> 
> - **Category:** Backtracking / Constraint Satisfaction
> - **Input:** A partially filled $9\times9$ grid
> - **Output:** A grid with all squares filled, or an indication no solution exists
> - **Paradigm:** Backtracking (fill in order, prune on constraint violation)
> - **Typical use cases:** the other canonical constraint-satisfaction example alongside [[8 Queens]]; generalizes to Latin squares and graph-coloring-style problems

---

# Problem Specification

- **Instance:** A partially filled puzzle.
- **Solution:** A grid with all squares filled with the numbers 1 through 9.
- **Constraint:** No repeats of any number within a given sub-square, row, or column.
- **Decision:** Find a solution satisfying the constraint (or determine none exists).

---

# Candidate Strategies / Approaches

## Exhaustive Search ✘

Try every possible digit in every blank cell, independently, then check the whole grid against all constraints at the end. For $b$ blank cells, this is $O(9^b)$ — the constraints are only used to _validate_ a finished guess, never to cut the search short.

## Backtracking ✔

- Fill in the first available cell with the least possible number, and recurse.
- If a cell is reached that can't be legally filled with _any_ number, backtrack to the last decision point and try the next-largest possible number there instead (if one is available).

> [!tip] Key Idea 
> Unlike Exhaustive Search, backtracking checks the row/column/sub-square constraints **as each digit is placed**, not just at the end — an illegal digit is rejected immediately, so the search never wastes time filling in the other 80 cells behind a guess that was already doomed.

---

# Pseudocode (Chosen Approach)

```pseudo
	\begin{algorithm}
	\caption{Sudoku Backtracking}
	\begin{algorithmic}
	\Input Grid $G$ (partially filled), cell index $i$ (in some fixed cell ordering)
	\Output A boolean value: whether the remaining cells from $i$ onward can be completed
	\Procedure{SolveSudoku}{$G, i$}
		\If{$i$ is past the last cell}
			\Return \True
        \EndIf
        \If{cell $i$ is already filled}
	        \Return SolveSudoku($G, i+1$)
        \EndIf
        \For{$d = 1 \dots 9$}
	        \If{placing $d$ at cell $i$ violates no row/column/sub-square constraint}
		        \State Place $d$ at cell $i$
		        \If{$SolveSudoku(G, i+1)$ is True}
			        \Return \True
                \EndIf
                \State Remove $d$ from cell $i$ \Comment{backtrack}
            \EndIf
        \EndFor
        \Return \False
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`G`|$9\times9$ grid|The puzzle state, partially filled|
|`i`|Cell index|Which cell is currently being considered, in a fixed traversal order|
|`d`|Digit, $1$–$9$|The candidate value being tried for cell `i`|

## Helper Functions / Operations Used

- **Constraint check** — verify digit `d` doesn't already appear in cell $i$'s row, column, or $3\times3$ sub-square; $O(1)$ (bounded by grid size).
- **Backtrack (remove `d`)** — undo a placement when it leads to a dead end further down the recursion, restoring the grid to try the next candidate digit.

---

# Proof of Correctness / Optimality

`SolveSudoku` only ever places a digit that satisfies all three constraints at the moment of placement, so no branch it explores can violate the row/column/sub-square rules. It tries every digit $1$–$9$ at each cell in order, backtracking to try the next digit whenever a placement leads to failure further down the recursion — so every legally reachable completion is eventually tried. Since there are finitely many cells and finitely many digits per cell, and the recursion always advances to $i+1$ on success, the search terminates.

---

# Time & Space Complexity Analysis

Worst-case, naive backtracking Sudoku solving remains exponential — generalized Sudoku (on an $n^2 \times n^2$ grid) is known to be NP-complete, so no polynomial-time algorithm is expected for the general case. In practice, the constraint checks prune the search so aggressively that even the hardest standard $9\times9$ puzzles solve near-instantly; this is a case where empirical performance is far better than the worst-case bound suggests.

---

# Drawbacks / Constraints

- **Plain backtracking can still be slow on adversarially hard puzzles** without additional heuristics — e.g. picking the cell with the _fewest_ remaining legal candidates first (minimum-remaining-values), rather than a fixed left-to-right cell order, typically prunes much faster in practice.
- **Doesn't scale to generalized $N\times N$ Sudoku** — the general problem is NP-complete, so worst-case exponential blowup is expected as $N$ grows, regardless of how well-tuned the backtracking is.

---

# References / Links

- [[Computer Science/Algorithms/Backtracking/index|Backtracking]]
- [[8 Queens Example|8 Queens]]