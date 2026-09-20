---
title: "Graph Algorithms Summary"
description: "Quick-reference summary of BFS, DFS, Dijkstra's, Prim's, and Kruskal's."
aliases:
  - Graph Algorithms Cheatsheet
tags:
  - CS/algorithms
  - graph-algorithms
  - summary
---

> [!abstract] Abstract
> Quick-reference summary of BFS, DFS, Dijkstra's, Prim's, and Kruskal's — what each explores with, and their time/space complexity.

---
## Concepts

In all graph traversal algorithms discussed, we choose a specific vertex at which to begin our traversal. We disallow multigraphs (parallel edges), so every graph here has at most $\vert{}V\vert{}^2$ edges.

**Graph Traversal Algorithms:**
- **[[6. Breadth First Search (BFS)|Breadth First Search]]:** Explores layer-by-layer via a [[Queues\|Queue]]. This guarantees shortest paths on unweighted graphs.
- **[[3. Depth First Search (DFS)|Depth First Search]]:** Explores the current path as far as possible before backtracking, implemented using a [[Computer Science Introduction/Data Structures/Introductory Data Structures/Stack\|Stack]]. Tracking the entire current path only costs $O(\Vert{}E\Vert{})$.
- **[[7. Dijkstra's Algorithm|Dijkstra's Algorithm]]:** Explores the shortest possible path using a [[Priority Queue]] (e.g., using a [[Heap]]) ordered by cumulative distance. Requires non-negative edge weights.

**Minimum Spanning Tree Algorithms:**
- A [[Minimum Spanning Trees]] of $G$ is a Spanning Tree of $G$ with minimum overall cost.
- **[[9. Prim's Algorithm|Prim's Algorithm]]:** Vertex-centric. Starts with a one-node tree and repeatedly adds a minimum-weight edge connecting a node not yet in the tree, using a [[Priority Queue]] keyed by single edge cost.
- **[[8. Kruskal's Algorithm|Kruskal's Algorithm]]:** Edge-centric. Repeatedly finds the minimum-weight edge that connects two previously unconnected trees in a forest, relying on sorted edges and [[Disjoint Sets & Up-Trees]].

---
## Examples

| Algorithm                                               | Time                                                    | Space                                | Key Structure Used                                                                                                |
| ------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| [[6. Breadth First Search (BFS)\|Breadth First Search]] | $O(\Vert{}V\Vert{}+\Vert{}E\Vert{})$                    | $O(\Vert{}V\Vert{}+\Vert{}E\Vert{})$ | [[Queue]]                                                                                                         |
| [[3. Depth First Search (DFS)\|Depth First Search]]     | $O(\Vert{}V\Vert{}+\Vert{}E\Vert{})$                    | $O(\Vert{}V\Vert{}+\Vert{}E\Vert{})$ | [[Computer Science Introduction/Programming Languages/C++/Standard Template Library/Stack\|Stack]] (or recursion) |
| [[7. Dijkstra's Algorithm\|Dijkstra's Algorithm]]       | $O(\Vert{}V\Vert{}+\Vert{}E\Vert{}\log\Vert{}E\Vert{})$ | $O(\Vert{}V\Vert{}+\Vert{}E\Vert{})$ | [[Priority Queue]] (by cumulative distance)                                                                       |
| [[9. Prim's Algorithm\|Prim's Algorithm]]               | $O(\Vert{}V\Vert{}+\Vert{}E\Vert{}\log\Vert{}E\Vert{})$ | $O(\Vert{}V\Vert{}+\Vert{}E\Vert{})$ | [[Priority Queue]] (by single edge cost)                                                                          |
| [[8. Kruskal's Algorithm\|Kruskal's Algorithm]]         | $O(\Vert{}V\Vert{}+\Vert{}E\Vert{}\log\Vert{}E\Vert{})$ | $O(\Vert{}V\Vert{}+\Vert{}E\Vert{})$ | Sorted edge list + Disjoint Sets                                                                                  |

---
# Related Notes

- [[6. Breadth First Search (BFS)|Breadth First Search]]
- [[3. Depth First Search (DFS)|Depth First Search]]
- [[7. Dijkstra's Algorithm|Dijkstra's Algorithm]]
- [[9. Prim's Algorithm|Prim's Algorithm]]
- [[8. Kruskal's Algorithm|Kruskal's Algorithm]]
- [[Minimum Spanning Trees]]