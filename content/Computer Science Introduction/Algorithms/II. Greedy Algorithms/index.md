---
title: Greedy Algorithms
description: Overview of the greedy design paradigm — locally optimal choices at each step — and why every greedy algorithm needs its own correctness proof.
tags:
  - CS/index
  - CS/algorithms
aliases:
  - Greedy Algorithms
---

> [!abstract] Overview 
> A Greedy Algorithm builds a solution step by step, at each step making the choice that looks best right now, without reconsidering earlier choices[cite: 64]. 

---
## Foundational Concepts

> [!info] Definition: The Greedy Method
> At each step, make the locally optimal choice, never revisiting or reconsidering a decision once made[cite: 64]. This is fast and simple to implement, but is only correct for certain problems[cite: 64]. The Greedy Method does not always work, meaning we must prove its correctness every time[cite: 64].

**Proving Optimality:**
Every greedy algorithm needs either a correctness proof or a counterexample[cite: 64]. Three general techniques exist:
1. **Modify the Solution (Exchange):** most general[cite: 64].
2. **Greedy-Stays-Ahead:** more intuitive[cite: 64].
3. **Greedy Achieves the Bound:** also comes up in approximation algorithms[cite: 64].

---
## Notes in This Section

| Note | Description |
|---|---|
| [[1. The Greedy Method]] | Full definition of what is the greedy method and general scheme of the greedy method[cite: 59]. |
| [[2. Event Scheduling Example]] | Interval scheduling — pick the maximum number of non-overlapping events; earliest-end-time strategy[cite: 64]. |
| [[3. Techniques to Prove Optimality]] | Proof strategies for greedy choices; relies on greedy stays ahead, exchange arguments, or bounds[cite: 64]. |
| [[4. Event Scheduling with Multiple Rooms]] | Greedy algorithm that assigns events to the minimum number of rooms without overlaps[cite: 62]. |

## Greedy Implementations
| Note | Description |
|---|---|
| [[9. Prim's Algorithm]] | Minimum Spanning Tree — greedily grows one tree by always adding the cheapest connecting edge[cite: 64]. |
| [[8. Kruskal's Algorithm]] | Minimum Spanning Tree — greedily adds the cheapest edge overall that doesn't create a cycle[cite: 64]. |
| [[7. Dijkstra's Algorithm]] | Shortest path — greedily finalizes the closest unvisited vertex at each step[cite: 64]. |

## Worked Optimality Proofs
| Note                                          | Description                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------ |
| [[5. Prove Kruskal's with Exchange Argument]] | Applies the Exchange technique to prove Kruskal's Algorithm optimal[cite: 64]. |

---
## Related Categories

- [[Computer Science Introduction/Algorithms/I. Graph Algorithms/index\|Graph Algorithms]]
- [[Minimum Spanning Trees]]