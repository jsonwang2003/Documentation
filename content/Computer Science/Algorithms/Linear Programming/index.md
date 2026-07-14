---
description: General framework for optimization problems with linear constraints and objective — the global optimum always sits at a corner of the feasible polyhedron, found efficiently via the Simplex algorithm.
tags:
  - linear-programming
aliases:
  - LP
  - Linear Programming
---
> [!abstract] Overview 
> Linear Programming is a general method of solving optimization problems, applicable whenever the constraints and objective function are all linear equations or inequalities. Many problems that look combinatorial — like [[Maximum Flow]] — can be formulated and solved this way, though specialized combinatorial algorithms often outperform a generic LP solver for problems with enough extra structure.

---

# Foundational Concepts

## Local Optima

One can view the set of all possible solutions as a high-dimensional region. The objective function then gives a "height" for each point — we want to find the highest point. Greedy algorithms or iterative-improvement algorithms sometimes only find a **local optimum**: a point higher than every point near it, but not necessarily the global optimum. Every global optimum is a local optimum, but the reverse isn't always true.

## Linear Programming

Linear Programming works when the constraints and objective function are all linear. The constraints limit the solution space to a polygon (2D) or a multi-dimensional **polyhedron**. Since the objective is linear too, there are **no local optima to get stuck at** — the global optimum always occurs at a **corner (vertex)** of the polyhedron.

> [!tip] Key Idea 
> Linear Programming is the process of traveling from one vertex of the feasible polyhedron to another, always improving, until you can't improve any further. Because linearity rules out local optima entirely, "can't improve any further" is guaranteed to mean "this is the global optimum" — unlike greedy or iterative-improvement methods on non-linear landscapes.

## The Simplex Algorithm

In practice, Linear Programming reduces to solving the **Simplex algorithm**, which requires relatively simple matrix operations (not covered in depth here). Simplex can be solved relatively efficiently, so any problem whose constraints and objective are linear can, in principle, be solved this way.

---

# Notes in This Section

| Note                                   | One-line description                                                                                                                                                                         |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[Maximum Flow Example\|Maximum Flow]] | Network flow — technically an LP problem (linear constraints and objective), but solved via the specialized, more efficient Ford-Fulkerson augmenting-path method instead of generic Simplex |

---

# Related Categories

- [[Computer Science/Algorithms/Greedy Algorithms/index|Greedy Algorithms]]
- [[Dynamic Programming]]
- [[Computer Science/Algorithms/Graph Algorithms/index|Graph Algorithms]]