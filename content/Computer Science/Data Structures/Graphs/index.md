---
title: Graphs
description: An index covering foundational graph properties, classifications, structural constraints, and storage hub notes.
aliases:
  - Graph Theory Hub
  - Network Algorithms Index
tags:
  - index
  - graph-theory
  - networks
---

# Overview 
Navigation systems, social networks, and routing pipelines are all structurally powered by Graphs. By abstracting real-world components into nodes and their corresponding relationships into edges, we can apply deterministic optimization mechanics to solve complex computational challenges like discovering the absolute shortest path across an arbitrary network.

---

## Foundational Concepts

### What is a Graph?
A graph is a non-linear mathematical structure used to model arbitrary relationships between discrete objects. Unlike [[Tree Structures/index|Tree Structures]], graphs have no built-in global root node or strict parent-child hierarchies. Instead, they form custom topographical networks that can be:
*   **Disconnected:** Containing distinct structural clusters with zero path channels between them.
*   **Sequential:** Nodes line up in a simple linear execution path.
*   **Hierarchical:** Emulating tree-like dependencies (in fact, a tree is simply a connected, acyclic undirected graph).
*   **Complex:** Intertwined multi-layered systems featuring dense cyclic tracking loops.

### Formal Definition
A graph $G$ is formally represented as an ordered pair $G = (V, E)$:
*   **$V$:** A finite, non-empty set of vertices (or nodes): $X = \{v_1, v_2, \dots, v_n\}$.
*   **$E$:** A set of edges (or links), where each individual edge $e \in E$ is a node pair $(v, w)$ mapping a connectivity lane.
*   **Sizing Bounds:** The structural scale of a network is quantified by $|V|$ (vertex cardinality) and $|E|$ (edge cardinality). For any simple graph, $|E|$ is strictly bounded by $O(|V|^2)$.

```
  Disconnected            Sequential            Hierarchical             Complex
   (A)   (B)              (A) -> (B) -> (C)         (Root)              (A) <---> (B)
                                                   /      \              ^         /
   (C)   (D)                                    (B)        (C)           \       v
                                               /   \                     (C) <-> (D)
```

### Classifying Graphs
*   **Directed Graph (Digraph):** Every edge has an explicit orientation arrow. The ordered pair $(v, w)$ defines a strictly one-way path from origin $v$ to destination $w$.
*   **Undirected Graph:** Edges are inherently bidirectional. The pair $(v, w)$ is structurally identical to $(w, v)$, forming a shared two-way lane.
*   **Weighted Graph:** Every edge is assigned a numerical cost metric $c$ representing physical distances, travel latency, or network bandwidth constraints.
*   **Unweighted Graph:** All connection links are considered equal, effectively tracking unit costs of $1$.

### Paths and Cycles
*   **Path:** A continuous sequence of edges linking a starting node to a destination node across a graph.
*   **Cycle:** A path that begins and terminates at the exact same vertex without re-evaluating edges.
*   **DAG (Directed Acyclic Graph):** A specialized directed graph architecture that contains absolutely zero internal cycles, serving as the foundational basis for scheduling and dependency sorting.

---

## Core Shared Matrix

| Property | Mathematical Constraint | Real-World Paradigm Examples |
| :--- | :--- | :--- |
| **Directed** | Edge relations are ordered pairs: $(v, w) \neq (w, v)$ | One-way street routes; asymmetric web links. |
| **Undirected** | Edge relations are symmetric sets: $(v, w) == (w, v)$ | Bidirectional highway lines; network peer connections. |
| **Weighted** | Functional mapping: $E \rightarrow \mathbb{R}$ | Fiber-optic cable length paths; airline fuel costs. |
| **Acyclic** | Graph contains no valid cyclic sub-paths | Academic course prerequisite tracks; compilation steps. |

---

## Notes in This Section

| Note | One-line description | Foundational Dependency |
| :--- | :--- | :--- |
| **[[Graph Representations]]** | Compares the architectural tradeoffs and spatial density profiles of Adjacency Matrices vs. Lists. | Memory bounds selection. |
| **[[Disjoint Sets & Up-Trees]]** | Implements a fast tracking forest to manage elements partitioned into isolated, independent subsets. | Dynamic connectivity engine. |
| **[[Minimum Spanning Trees]]** | Solves total edge cost reduction across weighted graphs without introducing cycles. | Optimization algorithms. |

---

## Related Categories
*   [[Tree Structures/index|Tree Structures Overview]]
*   [[Coding and Information Compression/index|Coding and Information Compression foundations]]