---
title: Dynamic Programming
description: Algorithmic paradigm that solves a problem by identifying overlapping subproblems and solving them smallest-first, reusing answers instead of recomputing them.
tags:
  - CS/index
  - CS/algorithms
aliases:
  - V. Dynamic Programming
  - DP
  - Dynamic Programming
---

> [!abstract] Overview 
> Dynamic Programming is an algorithmic paradigm in which a problem is solved by identifying a collection of subproblems and tackling them one by one, smallest first, using the answers to small problems to help figure out larger ones, until they are solved.

---
## Foundational Concepts

**Why Dynamic Programming? (From Backtracking to Memoization)**
Many [[Computer Science Introduction/Algorithms/IV. Backtracking/index|Backtracking]] algorithms make exponentially many recursive calls, but on closer inspection, often only revisit a small number of genuinely distinct subproblems, over and over, along different branches. See [[1. Weighted Event Scheduling Example|Weighted Event Scheduling Example]] for a fully worked example of exactly this: a backtracking solution making up to $2^n$ recursive calls, where only $n+1$ of them are actually distinct.
- **Memoization:** Store the answer to each distinct subproblem (e.g. in a hashmap or array) the first time it's computed, and reuse it instead of recomputing. This alone can turn an exponential algorithm polynomial, when the number of distinct subproblems is itself polynomial.
- **Dynamic Programming:** Usually goes one step further than plain memoization. Instead of top-down recursion with a cache, it solves the subproblems bottom-up, smallest first, filling in an array/table directly — avoiding recursion overhead entirely.

**The 8 Steps to Design a Dynamic Programming Algorithm**
A general recipe (see [[1. Weighted Event Scheduling Example|Weighted Event Scheduling Example]] for every one of these applied concretely to a real problem):
1. **Define sub-problems and the corresponding array:** The sub-problems are often just restatements of the original problem on a smaller instance.
2. **Determine the base case(s).**
3. **Give a recursion for the sub-problems (case analysis):** Break the sub-problem into distinct cases based on one key local decision.
4. **Order the sub-problems:** Ensure that each one only depends on already-solved, smaller sub-problems.
5. **Identify the final output:** Determine which array entry actually answers the original problem.
6. **Put it all together:** Create an iterative algorithm that fills in the array step by step (steps 1-5 assembled into one procedure).
7. **Prove correctness:** Usually done by induction, matching the case analysis from step 3.
8. **Runtime analysis:** Usually calculated as (number of sub-problems) $\times$ (cost to compute each one).

---
## Notes in This Section

| Note                                                                                  | Description                                                                                                                                                                                           |
| :------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[1. Weighted Event Scheduling Example\|Weighted Event Scheduling Example]]           | Interval scheduling where each event has a value to maximize, not just a count — the motivating example for why backtracking alone isn't enough, and how the 8-step DP recipe fixes it.               |
| [[2. String Reconstruction Example\|String Reconstruction Example]]                   | Determine whether a string can be split into valid words — boolean subproblems over string prefixes, plus `prev` pointers to reconstruct the actual split.                                            |
| [[3. The Knapsack Problem Example\|The Knapsack Problem Example]]                     | Maximize value packed into a weight-limited knapsack with reusable items — 2D table over (item prefix, capacity), $O(nC)$ pseudo-polynomial time.                                                     |
| [[4. Edit Distance Example\|Edit Distance Example]]                                   | Minimum insertions/deletions/substitutions to transform one string into another — 2D table over (prefix of $x$, prefix of $y$); also viewable as shortest path on a DAG.                              |
| [[5. Shortest Path in a DAG Example\|Shortest Path in a DAG Example]]                 | Linear-time single-source shortest path using topological order instead of a priority queue — works with negative weights too, since a DAG can't have cycles.                                         |
| [[6. Longest Increasing Subsequence Example\|Longest Increasing Subsequence Example]] | Reduces to longest path on a DAG (negate edge weights, add a virtual source) — $O(n^2)$ via the DAG framing.                                                                                          |
| [[7. Bellman-Ford Algorithm\|Bellman-Ford Algorithm]]                                 | Generalizes DAG shortest path to graphs with cycles, by budgeting the number of edges allowed — $O(VE)$, and detects negative cycles along the way.                                                   |
| [[8. Maximum Independent Set in Trees\|Maximum Independent Set in Trees]]             | Same problem as [[3. Maximal Independent Set Example\|Maximal Independent Set]], restricted to trees — subtrees never overlap, so this collapses to a clean $O(n)$ DP instead of staying exponential. |

---
## Related Categories

- [[Computer Science Introduction/Algorithms/IV. Backtracking/index\|Backtracking]]
- [[Computer Science Introduction/Algorithms/III. Divide and Conquer/index\|Divide and Conquer]]
- [[Computer Science Introduction/Algorithms/II. Greedy Algorithms/index\|Greedy Algorithms]]