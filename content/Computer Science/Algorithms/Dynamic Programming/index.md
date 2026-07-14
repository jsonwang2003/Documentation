---
title: Dynamic Programming
aliases:
  - Dynamic Programming
  - DP
description: Algorithmic paradigm that solves a problem by identifying overlapping subproblems and solving them smallest-first, reusing answers instead of recomputing them — often the fix for an exponential backtracking algorithm.
tags:
  - dynamic-programming
---
> [!abstract] 
>  Dynamic Programming is an algorithmic paradigm in which a problem is solved by identifying a collection of subproblems and tackling them one by one, smallest first, using the answers to small problems to help figure out larger ones, until they are solved.

---

# Foundational Concepts

## Why Dynamic Programming? (From Backtracking to Memoization)

Many [[Computer Science/Algorithms/Backtracking/index|Backtracking]] algorithms make exponentially many recursive calls, but on closer inspection, often only revisit a small number of genuinely _distinct_ subproblems, over and over, along different branches. See [[Weighted Event Scheduling Example|Weighted Event Scheduling]] for a fully worked example of exactly this: a backtracking solution making up to $2^n$ recursive calls, where only $n+1$ of them are actually distinct.

- **Memoization:** store the answer to each distinct subproblem (e.g. in a hashmap or array) the first time it's computed, and reuse it instead of recomputing — this alone can turn an exponential algorithm polynomial, _when_ the number of distinct subproblems is itself polynomial.
- **Dynamic Programming** usually goes one step further than plain memoization: instead of top-down recursion with a cache, solve the subproblems **bottom-up**, smallest first, filling in an array/table directly — avoiding recursion overhead entirely.

## The 8 Steps to Design a Dynamic Programming Algorithm

A general recipe (see [[Weighted Event Scheduling Example|Weighted Event Scheduling]] for every one of these applied concretely to a real problem):

1. **Define sub-problems and the corresponding array.** Hint: the sub-problems are often just restatements of the original problem on a smaller instance.
2. **Determine the base case(s).**
3. **Give a recursion for the sub-problems (case analysis).** Hint: break the sub-problem into distinct cases based on one key local decision.
4. **Order the sub-problems** so that each one only depends on already-solved, smaller sub-problems.
5. **Identify the final output** — which array entry actually answers the original problem.
6. **Put it all together** into an iterative algorithm that fills in the array step by step (steps 1-5 assembled into one procedure).
7. **Prove correctness** — usually by induction, matching the case analysis from step 3.
8. **Runtime analysis** — usually (number of sub-problems) $\times$ (cost to compute each one).

---

# Notes in This Section

| Note                                                                       | One-line description                                                                                                                                                                              |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[Weighted Event Scheduling Example\|Weighted Event Scheduling]]           | Interval scheduling where each event has a value to maximize, not just a count — the motivating example for why backtracking alone isn't enough, and how the 8-step DP recipe fixes it            |
| [[String Reconstruction Example\|String Reconstruction]]                   | Determine whether a string can be split into valid words — boolean subproblems over string prefixes, plus `prev` pointers to reconstruct the actual split                                         |
| [[The Knapsack Problem Example\|The Knapsack Problem]]                     | Maximize value packed into a weight-limited knapsack with reusable items — 2D table over (item prefix, capacity), $O(nC)$ pseudo-polynomial time                                                  |
| [[Edit Distance Example\|Edit Distance]]                                   | Minimum insertions/deletions/substitutions to transform one string into another — 2D table over (prefix of $x$, prefix of $y$); also viewable as shortest path on a DAG                           |
| [[Shortest Path in a DAG Example\|Shortest Path in a DAG]]                 | Linear-time single-source shortest path using topological order instead of a priority queue — works with negative weights too, since a DAG can't have cycles                                      |
| [[Longest Increasing Subsequence Example\|Longest Increasing Subsequence]] | Reduces to longest path on a DAG (negate edge weights, add a virtual source) — $O(n^2)$ via the DAG framing                                                                                       |
| [[Bellman-Ford Algorithm]]                                                 | Generalizes DAG shortest path to graphs _with_ cycles, by budgeting the number of edges allowed — $O(VE)$, and detects negative cycles along the way                                              |
| [[Maximum Independent Set in Trees]]                                       | Same problem as [[Maximal Independent Set Example\|Maximal Independent Set]], restricted to trees — subtrees never overlap, so this collapses to a clean $O(n)$ DP instead of staying exponential |

---

# Related Categories

- [[Computer Science/Algorithms/Backtracking/index|Backtracking]]
- [[Computer Science/Algorithms/Divide and Conquer/index|Divide and Conquer]]
- [[Computer Science/Algorithms/Greedy Algorithms/index|Greedy Algorithms]]