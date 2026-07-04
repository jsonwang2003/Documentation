---
aliases:
  - MST
description: A Minimum Spanning Tree is a subset of a graph’s edges that connects all vertices with no cycles and with the smallest possible total edge weight.
tags:
  - MinimumSpanningTree
  - Tree
---

> [!ABSTRACT]
> 
> The **Minimum Spanning Tree** problem asks for a subset of edges that connects all vertices in an undirected weighted graph without any cycles, while minimizing the total edge weight. This is the foundational logic for cost-efficient network design (e.g., laying fiber-optic cables across a campus).

---
# Defining the Spanning Tree

A **Spanning Tree** of a connected, undirected graph $G = (V,E)$ is a subgraph $G' = (V', E')$ that:

- **Spans:** It includes every vertex of $G$.
- **$G'$ Is a Tree:** It is connected and contains **no cycles**.
- **Edge Count:** For a graph with $|V|$ vertices, the spanning tree must have exactly $|V| - 1$ edges.

> **STOP and Think:** A single graph can have many different spanning trees. However, if all edge weights are unique, there is only **one** unique Minimum Spanning Tree.

---
# MST Algorithms: Prim’s vs. Kruskal’s

While both algorithms find the MST in $O(|E| \log |E|)$ time, they approach the problem with different "greedy" strategies.

## [[Prim's Algorithm]] (Vertex-Centric)

Prim’s grows the tree from a **starting root**, similar to how Dijkstra’s explores a graph.

1. Start with an arbitrary vertex.
2. At each step, identify all edges connecting the current tree to "untouched" vertices.
3. Add the **cheapest** edge to the tree.
4. Repeat until all vertices are in the tree.

**Implementation Note:** Uses a **Priority Queue** to store edges connected to the growing tree, sorted by individual edge weight.

## [[Kruskal's Algorithm]] (Edge-Centric)

Kruskal’s treats the graph as a **forest** of individual nodes and merges them together.

1. Sort **all** edges in the graph by weight from smallest to largest.
2. Pick the smallest edge.
3. If adding this edge does not create a cycle, add it to the MST.
4. If it creates a cycle, discard it.
5. Repeat until you have $|V| - 1$ edges.

**Implementation Note:** Uses a **Priority Queue** for all edges and a **Union-Find (Disjoint Set)** data structure to check for cycles efficiently.

> [!Important] Proving **Prim's** and **Kruskal's**
> Use [[Cut Property]] to prove the 2 algorithm, which indicates that for any way of splitting a graph's vertices into 2 groups, the cheapest edge crossing that split is guaranteed to be in the same **Minimum Spanning Tree**

---
# Comparison of Approaches

| Feature           | Prim's Algorithm           | Kruskal's Algorithm       |
| ----------------- | -------------------------- | ------------------------- |
| Strategy          | Grows a single tree        | Merges a forest of trees  |
| Priority Queue    | Stores edges from the tree | Stores ALL edges in graph |
| Cycle Detection   | "Done" boolean check       | Union-Find structure      |
| Best For          | Dense Graphs               | Sparse Graphs             |
| Negative Weights? | Yes (Unlike Dijkstra)      | Yes                       |
| Complexity        | $O(\|E\| \log \|V\|)$      | $O(\|E\| \log \|E\|)$     |

---
# Why BFS/DFS Isn't Enough

While **BFS** and **DFS** can find _any_ spanning tree in $O(|V| + |E|)$, they are "weight-blind." They will return the first edges they find, which rarely results in the minimum cost. Prim’s and Kruskal’s ensure that the final result is the absolute cheapest configuration possible.

> [!Note] 
> In Kruskal’s, an edge connecting two already-visited vertices doesn't _always_ create a cycle. This is because those vertices might belong to two different "islands" (trees) in the forest that haven't been connected yet.