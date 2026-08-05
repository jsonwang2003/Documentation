---
title: Backtracking
description: Generic method for exponentially-large search/optimization problems that prunes dead-end branches using the problem's constraints — often a stepping stone toward Greedy or Dynamic Programming algorithms.
tags:
  - backtracking
aliases:
  - Backtracking
  - Backtracking Algorithms
---
> [!abstract] Overview 
> Backtracking is a generic method applicable to many problems with an exponentially large solution set (search and optimization problems). It often gives a more efficient runtime than **exhaustive search** / **brute force**, but usually doesn't reach a polynomial-time algorithm — typically it's an _improved_ exponential time instead. It applies even to NP-Complete problems, where we don't expect to find sub-exponential algorithms at all, and can perform much better than its worst case on typical inputs.
> 
> Backtracking is often a first step toward finding a [[Computer Science Introduction/Algorithms/Greedy Algorithms/index|Greedy]] or [[Computer Science Introduction/Algorithms/Dynamic Programming/index|Dynamic Programming]] algorithm — see the callout at the bottom of this note.

---
# Foundational Concepts

## Backtracking vs. Exhaustive Search

Many problems involve finding the best (or any valid) solution from among a large space of possibilities. This is usually specified with:

- **Solution Format:** exhaustive search generally loops through all possibilities that satisfy this format.
- **Constraints:** backtracking uses the constraints to **eliminate impossible solutions early** — often before a candidate is even fully built.

Applying this pruning recursively, at every level of the search, gives substantial savings over exhaustive search — even though the worst-case asymptotic order often remains exponential.

## Reduce and Conquer (Recursion)

The main implementation strategy for backtracking is to recurse on smaller subproblems and use the results to solve the original problem — just like [[Computer Science Introduction/Algorithms/Divide and Conquer/index|Divide and Conquer]]. The key difference: backtracking algorithms often only reduce the problem size by a constant _difference_ (e.g. removing one vertex) rather than a constant _factor_ (e.g. halving the input) the way Divide and Conquer does. The general recursive idea is otherwise the same.

---

# Notes in This Section

| Note                                                         | One-line description                                                                                                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[8 Queens Example\|8 Queens]]                               | Place 8 non-attacking queens on a chessboard — pruning attacked cells column by column instead of generating full permutations                                |
| [[Sudoku Example\|Sudoku]]                                   | Fill a partially-completed grid — pick the least-constrained cell, try the smallest valid digit, backtrack on dead ends                                       |
| [[Maximal Independent Set Example\|Maximal Independent Set]] | Find the largest set of mutually non-adjacent vertices — refined through three iterations from $O(2^n)$ down to $O(1.48^n)$ by exploiting low-degree vertices |

---

# Related Categories

- [[Computer Science Introduction/Algorithms/Divide and Conquer/index|Divide and Conquer]]
- [[Computer Science Introduction/Algorithms/Greedy Algorithms/index|Greedy Algorithms]]
- [[Computer Science Introduction/Algorithms/Dynamic Programming/index|Dynamic Programming]]

> [!tip] Where This Leads: Dynamic Programming 
> Backtracking recursions often revisit structurally identical subproblems multiple times along different branches — for example, [[Maximal Independent Set Example|Maximal Independent Set]]'s recursive calls on overlapping vertex subsets, or two different partial Sudoku fills that happen to leave the same remaining sub-grid. [[Computer Science Introduction/Algorithms/Dynamic Programming/index|Dynamic Programming]] is the natural next step from here: instead of recomputing a subproblem's answer every time it's encountered, cache (memoize) it — which can turn an exponential backtracking algorithm into a polynomial one, _when_ the number of distinct subproblems is itself polynomial. (Note this isn't automatic — [[Maximal Independent Set Example|Maximal Independent Set]] is a case where the subproblems, arbitrary induced subgraphs, don't collapse down to a small polynomial set, which is part of why it stays exponential even after heavy refinement.)