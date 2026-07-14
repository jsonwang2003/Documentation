---
title: Greedy Algorithm
description: Overview of the greedy design paradigm — locally optimal choices at each step — and why every greedy algorithm needs its own correctness proof.
tags:
  - algorithm
  - greedy-algorithms
  - Greedy
aliases:
  - Greedy Algorithms
  - Greedy Algorithm
  - Greedy
---
> [!abstract] Overview 
> A **Greedy Algorithm** builds a solution step by step, at each step making the choice that looks best _right now_, without reconsidering earlier choices. See [[The Greedy Method]] for the full definition and general schema.

> [!Warning] 
> Greedy Method does not always work The **Greedy Method** does not always work. In order to use it, we must prove the correctness of the algorithm every time it's developed — or else present a counterexample showing that a particular greedy strategy will not work.
> 
> Furthermore, for a single problem, there may be more than one potential greedy strategy (more than one way to choose the "best" possible choice at each step). The problem may be solved by one strategy but not another — see [[Event Scheduling]] for a worked example where three plausible-looking strategies fail and only a fourth succeeds.

---
# Foundational Concepts

## [[The Greedy Method]]

At each step, make the locally optimal choice, never revisiting or reconsidering a decision once made. This is fast and simple to implement, but — per the warning above — is only correct for certain problems, and only for certain choices of "what looks best." See [[The Greedy Method]] for the complete write-up.

## Proving Optimality

Since greedy doesn't always work, every greedy algorithm needs either a correctness proof or a counterexample. The general obligation: for every instance $I$, letting $GS$ be the greedy algorithm's solution and $OS$ be _any other_ solution,

$$ \underbrace{ \boxed{Value(OS) \leq Value(GS)} }_{ \text{Maximize} } \text{ or } \underbrace{ \boxed{Cost(GS) \leq Cost(OS)} }_{ \text{Minimize} } $$

The tricky part: $OS$ is an arbitrary solution, not one that makes sense to reason about directly — we don't know much about it. Three general techniques exist to get around this:

1. **Modify the Solution (Exchange)** — most general.
2. **Greedy-Stays-Ahead** — more intuitive.
3. **Greedy Achieves the Bound** — also comes up in approximation algorithms, LP, and network flow.

See [[Techniques to Prove Optimality]] for the complete write-up, and [[Prove Kruskal's with Exchange Argument]] for a worked example of technique 1.

---

# Notes in This Section

| Note                               | One-line description                                                                                                                               |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[Event Scheduling]]               | Interval scheduling — pick the maximum number of non-overlapping events; earliest-end-time is the only one of four candidate strategies that works |
| [[Techniques to Prove Optimality]] | Proof strategies for greedy choices; primarily relies on **greedy stays ahead** (induction) or **exchange arguments** (gradual transformation).    |
| [[The Greedy Method]]              | Proof strategies for greedy choices; primarily relies on **greedy stays ahead** (induction) or **exchange arguments** (gradual transformation).    |
## Greedy Algorithms 

|                          |                                                                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| [[Prim's Algorithm]]     | Minimum Spanning Tree — greedily grows one tree by always adding the cheapest edge connecting it to a new vertex |
| [[Kruskal's Algorithm]]  | Minimum Spanning Tree — greedily adds the cheapest edge overall that doesn't create a cycle                      |
| [[Dijkstra's Algorithm]] | Shortest path — greedily finalizes the closest unvisited vertex at each step                                     |


## Worked Optimality Proofs

|Note|One-line description|
|---|---|
|[[Prove Kruskal's with Exchange Argument]]|Applies the Exchange technique to prove Kruskal's Algorithm optimal|

---

# Related Categories

- [[Graph Algorithms Index]]
- [[Minimum Spanning Trees]]