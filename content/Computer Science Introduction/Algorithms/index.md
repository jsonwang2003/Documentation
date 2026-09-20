---
title: "Algorithms"
description: "Foundational definitions, the hierarchy of algorithm obviousness, basic arithmetic time costs, and links to all algorithm families."
tags:
  - CS/index
  - CS/algorithms
aliases: ["Algorithms"]
---

> [!abstract] Overview 
> This is the root index for the Algorithms area of the vault — foundational definitions and cost models that apply across every algorithm family below, plus links to each family's own index.

---
## Foundational Concepts

> [!info] Definition: What is an Algorithm?
> A procedure for performing a computation, broken into well-specified steps.
> - **Input:** $X$, an instance
> - **Output:** $Y$, a solution
> 
> Both $X$ and $Y$ should be finitely describable. A good algorithm must produce the correct answer, in a reasonable amount of time and space, using the least energy.

## Hierarchy of Obviousness

![[Pasted image 20260109102601.png]]

1. **Obvious algorithms:** implicit in the problem statement — brute force, exhaustive search.
2. **Methodical algorithms:** applying general principles and paradigms that improve algorithms across a wide variety of problems (e.g., [[Levels of Algorithm Design|the design paradigms]] this vault is organized around).
3. **Clever algorithms:** stretching those general paradigms in a way that best fits one particular problem — usually where the real insight and difficulty of a course lives.

---
## Core Concept / Shared Building Block

**Time for Arithmetic**
The CPU is designed to process instructions on word-sized inputs.
- Inputs **less than** word size: performed on the CPU in a single access.
- Inputs **greater than** word size: must be broken down into word-sized chunks.

| Operation | floating point | $n < \text{wordsize}$ | arbitrary $n$ |
|---|---|---|---|
| Addition | $O(1)$ | $O(1)$ | $O(n)$ |
| Subtraction | $O(1)$ | $O(1)$ | $O(n)$ |
| Comparison | $O(1)$ | $O(1)$ | $O(n)$ |
| Multiplication | $O(1)$ | $O(1)$ | $O(n^2)$ |

---
## Notes in This Section

| Category                                                                                       | One-line description                                                                                                                                 |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[Computer Science Introduction/Algorithms/I. Graph Algorithms/index\|Graph Algorithms]]       | Traversal (DFS/BFS), shortest paths (Dijkstra's), and MSTs (Prim's/Kruskal's) — all specializations of the generic Graph Search procedure. |
| [[Computer Science Introduction/Algorithms/II. Greedy Algorithms/index\|Greedy Algorithms]]    | Locally-optimal-choice algorithms, plus the three general techniques for proving one actually is optimal.                                  |
| [[Computer Science Introduction/Algorithms/III. Divide and Conquer/index\|Divide and Conquer]] | Break into smaller similar subproblems, solve recursively, combine — sorting, selection, and the Master Theorem that analyzes them all.    |
| [[Computer Science Introduction/Algorithms/V. Dynamic Programming/index\|Dynamic Programming]] | Identify overlapping subproblems and solve smallest-first — often the fix for an exponential Backtracking algorithm.                       |
| [[Computer Science Introduction/Algorithms/IV. Backtracking/index\|Backtracking]]              | Exhaustive search that prunes dead-end branches using the problem's constraints — usually exponential, but much better than brute force.   |
| [[Computer Science Introduction/Algorithms/VI. Linear Programming/index\|Linear Programming]]  | Optimization with linear constraints and objective — no local optima, global optimum always at a vertex of the feasible region.            |

---
## Related Categories

- [[Levels of Algorithm Design]] — the High/Mid/Low-Level Design framework used throughout this vault's individual algorithm notes.
- [[Algorithm Base.base\|Algorithm Base]] ― A structured, filterable, sortable database view over the algorithm notes.