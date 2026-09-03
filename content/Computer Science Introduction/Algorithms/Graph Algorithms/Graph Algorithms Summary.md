---
description: Quick-reference summary of BFS, DFS, Dijkstra's, Prim's, and Kruskal's — what each explores with, and their time/space complexity.
tags:
  - summary
  - graph-algorithms
aliases:
  - Graph Algorithms Cheatsheet
---
> [!Note] Section Overview
> 
> - In all graph traversal algorithms discussed, we choose a specific vertex at which to begin our traversal.
> - We disallow "multigraphs" (parallel edges — multiple edges with the same start and end node), so every graph here has at most $|V|^2$ edges.

---

# Graph Traversal Algorithms

## [[Breadth First Search (BFS)]]

We explore the starting vertex, then its neighbors, then their neighbors, etc. — the graph is explored in layers spreading out from the starting vertex. Easily implemented using a [[Queues|Queue]] to keep track of vertices to explore. See [[Breadth First Search (BFS)]] for the complete write-up.

- **Time Complexity:** $O(|V| + |E|)$ — we potentially visit all $|V|$ vertices and traverse all $|E|$ edges, each in $O(1)$.
- **Space Complexity:** $O(|V| + |E|)$ — we might have to keep track of every possible vertex and edge during exploration. If we wanted to also keep track of the entire current path of every vertex in the [[Queues|Queue]], the space complexity would blow up.
- **Key detail:** layer-by-layer exploration via a [[Queues|Queue]] is what guarantees shortest paths on unweighted graphs.

## [[Depth First Search (DFS)]]

We explore the current path as far as possible before going back to explore other alternative paths. Easily implemented using a [[Computer Science Introduction/Data Structures/Introductory Data Structures/Stack|Stack]] to keep track of vertices to explore. See [[Depth First Search (DFS)]] for the complete write-up.

- **Time Complexity:** $O(|V| + |E|)$ — we potentially visit all $|V|$ vertices and traverse all $|E|$ edges, each in $O(1)$.
- **Space Complexity:** $O(|V| + |E|)$ — we might have to keep track of every possible vertex and edge during exploration.
- **Key detail:** because we only explore a single path at a time, tracking the _entire current path_ only costs $O(|E|)$, since a single path can have at most $|E|$ edges — much cheaper than BFS's equivalent.

## [[Dijkstra's Algorithm]]

We explore the shortest possible path at any given moment. Easily implemented using a [[Priority Queue]], ordered by _shortest distance_ from the starting vertex, to keep track of vertices to explore. See [[Dijkstra's Algorithm]] for the complete write-up.

- **Time Complexity:** $O(|V| + |E|\log|E|)$ — we initialize each of $|V|$ vertices, and in the worst case insert (and remove) one element into the Priority Queue per edge, assuming the **Priority Queue** is implemented intelligently (e.g. using a [[Heap]]).
- **Space Complexity:** $O(|V| + |E|)$ — we might have to keep track of every possible vertex and edge during exploration.
- **Key detail:** requires non-negative edge weights; the **Priority Queue** is keyed by _cumulative_ distance from the source, not a single edge weight (contrast with [[#Prim's Algorithm|Prim's]] below).

---

# Minimum Spanning Tree Algorithms

> [!Note] Section Overview
> 
> - Given a graph $G$, a [[Minimum Spanning Trees#1. Defining the Spanning Tree|Spanning Tree]] is a tree that hits every node in $G$.
> - A [[Minimum Spanning Trees|Minimum Spanning Tree]] of $G$ is a Spanning Tree of $G$ with minimum overall cost (minimizes the sum of all edge weights).
> - Prim's and Kruskal's both find an MST in an arbitrary graph $G$ equally efficiently, using different strategies.

## [[Prim's Algorithm]]

Starts with a one-node tree and repeatedly finds a minimum-weight edge that connects a node in the tree to a node not yet in the tree, adding that edge to the tree. See [[Prim's Algorithm]] for the complete write-up.

- **Time Complexity:** $O(|V| + |E|\log|E|)$ — initialize all $|V|$ vertices, and add each of $|E|$ edges to a [[Priority Queue]] (implemented using a [[Heap]]).
- **Key detail:** vertex-centric — grows _one_ tree at a time; the **Priority Queue** is keyed by cost of the single cheapest edge connecting to the tree, not cumulative path weight (contrast with [[#Dijkstra's Algorithm|Dijkstra's]] above).

## [[Kruskal's Algorithm]]

Starts with a forest of one-node trees and repeatedly finds the minimum-weight edge that connects two previously unconnected trees in the forest, merging them using that edge. See [[Kruskal's Algorithm]] for the complete write-up.

- **Time Complexity:** $O(|V| + |E|\log|E|)$ — initialize all $|V|$ vertices, and sort all $|E|$ edges (the fastest comparison sorts run in $O(n\log n)$).
- **Key detail:** edge-centric — grows a whole _forest_ at once; relies on [[Disjoint Sets & Up-Trees]] to check whether two endpoints are already connected before merging.

---

# Quick Reference Table

|Algorithm|Time|Space|Key Structure Used|
|---|---|---|---|
|[[Breadth First Search (BFS)]]|$O(\|V\|+\|E\|)$|$O(\|V\|+\|E\|)$|Queue|
|[[Depth First Search (DFS)]]|$O(\|V\|+\|E\|)$|$O(\|V\|+\|E\|)$|Stack (or recursion)|
|[[Dijkstra's Algorithm]]|$O(\|V\|+\|E\|\log\|E\|)$|$O(\|V\|+\|E\|)$|Priority Queue (by cumulative distance)|
|[[Prim's Algorithm]]|$O(\|V\|+\|E\|\log\|E\|)$|$O(\|V\|+\|E\|)$|Priority Queue (by single edge cost)|
|[[Kruskal's Algorithm]]|$O(\|V\|+\|E\|\log\|E\|)$|$O(\|V\|+\|E\|)$|Sorted edge list + Disjoint Sets|

---

# References / Links

- [[Breadth First Search (BFS)]]
- [[Depth First Search (DFS)]]
- [[Dijkstra's Algorithm]]
- [[Prim's Algorithm]]
- [[Kruskal's Algorithm]]
- [[Minimum Spanning Trees]]