---
title: "IV. Backtracking"
description: "Generic method for exponentially-large search/optimization problems that prunes dead-end branches using the problem's constraints."
tags:
  - CS/index
  - CS/algorithms
aliases: ["IV. Backtracking"]
---

> [!abstract] Overview 
> Backtracking is a generic method applicable to many problems with an exponentially large solution set (search and optimization problems)[cite: 65]. It often gives a more efficient runtime than exhaustive search / brute force, but usually doesn't reach a polynomial-time algorithm — typically it's an improved exponential time instead[cite: 65]. It applies even to NP-Complete problems, where we don't expect to find sub-exponential algorithms at all, and can perform much better than its worst case on typical inputs[cite: 65].

---
## Foundational Concepts

**Backtracking vs. Exhaustive Search**
Many problems involve finding the best (or any valid) solution from among a large space of possibilities[cite: 65]. 
- **Solution Format:** Exhaustive search generally loops through all possibilities that satisfy this format[cite: 65].
- **Constraints:** Backtracking uses the constraints to eliminate impossible solutions early — often before a candidate is even fully built[cite: 65].

Applying this pruning recursively, at every level of the search, gives substantial savings over exhaustive search — even though the worst-case asymptotic order often remains exponential[cite: 65].

**Reduce and Conquer (Recursion)**
The main implementation strategy for backtracking is to recurse on smaller subproblems and use the results to solve the original problem — just like [[Computer Science Introduction/Algorithms/III. Divide and Conquer/index|Divide and Conquer]][cite: 65]. The key difference: backtracking algorithms often only reduce the problem size by a constant difference (e.g. removing one vertex) rather than a constant factor (e.g. halving the input) the way Divide and Conquer does[cite: 65]. 

> [!tip] Where This Leads: Dynamic Programming 
> Backtracking recursions often revisit structurally identical subproblems multiple times along different branches — for example, [[3. Maximal Independent Set Example\|Maximal Independent Set]]'s recursive calls on overlapping vertex subsets, or two different partial Sudoku fills that happen to leave the same remaining sub-grid[cite: 65]. Dynamic Programming is the natural next step from here: instead of recomputing a subproblem's answer every time it's encountered, cache (memoize) it — which can turn an exponential backtracking algorithm into a polynomial one, when the number of distinct subproblems is itself polynomial[cite: 65]. (Note this isn't automatic — [[3. Maximal Independent Set Example\|Maximal Independent Set]] is a case where the subproblems, arbitrary induced subgraphs, don't collapse down to a small polynomial set, which is part of why it stays exponential even after heavy refinement)[cite: 65].

---
## Notes in This Section

| Note                                                            | Description                                                                                                                                                              |
| :-------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[1. 8 Queens Example\|8 Queens]]                               | Place 8 non-attacking queens on a chessboard — pruning attacked cells column by column instead of generating full permutations[cite: 65].                                |
| [[2. Sudoku Example\|Sudoku]]                                   | Fill a partially-completed grid — pick the least-constrained cell, try the smallest valid digit, backtrack on dead ends[cite: 65].                                       |
| [[3. Maximal Independent Set Example\|Maximal Independent Set]] | Find the largest set of mutually non-adjacent vertices — refined through three iterations from $O(2^n)$ down to $O(1.48^n)$ by exploiting low-degree vertices[cite: 65]. |

---
## Related Categories

- [[Computer Science Introduction/Algorithms/III. Divide and Conquer/index\|Divide and Conquer]][cite: 65]
- [[Computer Science Introduction/Algorithms/II. Greedy Algorithms/index\|Greedy Algorithms]][cite: 65]
- [[Computer Science Introduction/Algorithms/V. Dynamic Programming/index\|Dynamic Programming]][cite: 65]