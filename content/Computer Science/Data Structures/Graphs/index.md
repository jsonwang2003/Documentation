---
title: Graphs
---
> [!ABSTRACT]
> 
> Navigation systems, social networks, and airport connections are all powered by **Graphs**. By abstracting real-world locations into **Nodes** and their connections into **Edges**, we can solve complex problems like finding the "shortest path" between any two points in the world.

---
## 1. What is a Graph?

A graph is a mathematical structure used to model relationships between objects. It consists of:
- **Nodes (or Vertices):** The individual elements or "locations" in the system.
- **Edges:** The connections between pairs of nodes.

Unlike trees, graphs have no strict hierarchy. They can be:
- **Disconnected:** Some nodes have no paths between them.

![[Pasted image 20260222183847.png]]

- **Sequential:** Nodes follow a linear path.

![[Pasted image 20260222183859.png]]

- **Hierarchical:** Organized like a tree (though trees are actually just a specific type of graph).

![[Pasted image 20260222183908.png]]

- **Complex:** A mix of connected and disconnected components.

![[Pasted image 20260222183915.png]]

---
## 2. Formal Definition

A graph $G$ is formally represented as an ordered pair $G = (V, E)$:
- **$V$:** A set of vertices $\{v_1, v_2, \dots, v_n\}$.
- **$E$:** A set of edges, where each edge $e$ is a pair of vertices $(v, w)$.

> [!Example] 
> $e_1 = (a, b)$
> 
> ![[Pasted image 20260222184226.png]]

The size of a graph is denoted by 
- $|V|$ (number of vertices)
- $|E|$ (number of edges).

---
## 3. Classifying Graphs

Graphs are categorized based on how their edges behave:

### Directed vs. Undirected

- **Directed Graph:** Edges have a specific direction (one-way). $(v, w)$ means you can go from $v$ to $w$, but not necessarily back.

![[Pasted image 20260222184103.png]]

- **Undirected Graph:** Edges are bidirectional. $(v, w)$ is equivalent to $(w, v)$.

![[Pasted image 20260222184121.png]]

### Weighted vs. Unweighted

- **Weighted Graph:** Each edge has a "cost" or "weight" $c$ (e.g., distance or travel time).

![[Pasted image 20260222184133.png]]

- **Unweighted Graph:** All edges are considered equal (effectively having a weight of 1).

---
## 4. Paths and Cycles

- **Path:** A sequence of edges connecting a start node to an end node.
    
- **Cycle:** A path that starts and ends at the same node.

![[Pasted image 20260222184404.png]]

$$
\boxed{a} \to b \to c \to d \to \boxed{a}
$$

- **DAG (Directed Acyclic Graph):** A directed graph that contains no cycles.

> More information can be found [[Graph Reachability|here]]
---
## 5. Summary of Properties

|**Property**|**Description**|**Real-World Example**|
|---|---|---|
|**Directed**|Edges have arrows/direction|One-way streets; Twitter followers|
|**Undirected**|Edges are bidirectional|Two-way streets; Facebook friends|
|**Weighted**|Edges have numerical values|Distance in miles; Toll costs|
|**Acyclic**|No paths loop back to start|Prerequisites for a college major|
