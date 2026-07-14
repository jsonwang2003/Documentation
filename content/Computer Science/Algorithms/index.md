---
title: Algorithms
description: Root index for the Algorithms area — what an algorithm is, the hierarchy of algorithm 'obviousness,' basic arithmetic time costs, and links to every algorithm family in this vault.
tags:
  - algorithm
aliases:
  - Algorithms
---
> [!abstract] Overview 
> This is the root index for the Algorithms area of the vault — foundational definitions and cost models that apply across every algorithm family below, plus links to each family's own index.

---

# Foundational Concepts

## What is an Algorithm?

> [!INFO] Definition A procedure for performing a computation, broken into well-specified steps.
> 
> - **Input:** $X$, an instance
> - **Output:** $Y$, a solution
> 
> Both $X$ and $Y$ should be finitely describable.

> [!Note] A **good** algorithm must produce the **correct answer**, in a _reasonable_ amount of **time** and **space**, using the **least energy**.

## Hierarchy of Obviousness

![[Pasted image 20260109102601.png]]

1. **Obvious algorithms:** implicit in the problem statement — brute force, exhaustive search.
2. **Methodical algorithms:** applying _general principles and paradigms_ that improve algorithms across a wide variety of problems (e.g. [[Levels of Algorithm Design|the design paradigms]] this vault is organized around).
3. **Clever algorithms:** stretching those general paradigms in a way that best fits one _particular_ problem — usually where the real insight and difficulty of a course lives.

## Time for Arithmetic

The CPU is designed to process instructions on word-sized inputs.

- Inputs **less than** word size: performed on the CPU in a single access.
- Inputs **greater than** word size: must be broken down into word-sized chunks.

| |floating point|$n < \text{wordsize}$|arbitrary $n$|
|---|---|---|---|
|Addition|$O(1)$|$O(1)$|$O(n)$|
|Subtraction|$O(1)$|$O(1)$|$O(n)$|
|Comparison|$O(1)$|$O(1)$|$O(n)$|
|Multiplication|$O(1)$|$O(1)$|$O(n^2)$|

---

# Categories in This Vault

|Category|Index|One-line description|
|---|---|---|
|Graph Algorithms|[[Computer Science/Algorithms/Graph Algorithms/index\|Graph Algorithms]]|Traversal (DFS/BFS), shortest paths (Dijkstra's), and MSTs (Prim's/Kruskal's) — all specializations of the generic Graph Search procedure|
|Greedy Algorithms|[[Computer Science/Algorithms/Greedy Algorithms/index\|Greedy Algorithms]]|Locally-optimal-choice algorithms, plus the three general techniques for proving one actually is optimal|
|Divide and Conquer|[[Computer Science/Algorithms/Divide and Conquer/index\|Divide and Conquer]]|Break into smaller similar subproblems, solve recursively, combine — sorting, selection, and the Master Theorem that analyzes them all|
|Dynamic Programming|[[Computer Science/Algorithms/Dynamic Programming/index\|Dynamic Programming]]|Identify overlapping subproblems and solve smallest-first — often the fix for an exponential Backtracking algorithm|
|Backtracking|[[Computer Science/Algorithms/Backtracking/index\|Backtracking]]|Exhaustive search that prunes dead-end branches using the problem's constraints — usually exponential, but much better than brute force|
|Linear Programming|[[Computer Science/Algorithms/Linear Programming/index\|Linear Programming]]|Optimization with linear constraints and objective — no local optima, global optimum always at a vertex of the feasible region|

---

# Related Notes

- [[Levels of Algorithm Design]] — the High/Mid/Low-Level Design framework used throughout this vault's individual algorithm notes.
- [[Algorithm Base.base|Algorithm Base]] ― A structured, filterable, sortable database view over the algorithm notes