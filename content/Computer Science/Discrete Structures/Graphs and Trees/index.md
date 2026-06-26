---
title: Graphs and Trees
---
> [!ABSTRACT]
> 
> Graphs and trees are mathematical structures used to model relationships between objects. This directory covers the transition from general networks (Graphs) to restricted hierarchical structures (Rooted Trees) used for efficient data storage and retrieval.

---
## Knowledge Map

### Fundamentals
- **[[Graphs]]**: The basic definition of vertices and edges. Covers the formal distinction between directed and undirected systems.
- **[[Graph Reachability]]**: Determining connectivity between nodes and identifying **Directed Acyclic Graphs (DAGs)**, the precursor to rooted trees.
### Tree Structures
1. **[[Undirected Trees]]**
    - **Concept**: A connected graph with no cycles.
    - **Property**: There is exactly one unique path between any two vertices, and $|E| = |V| - 1$.
2. **[[Directed Tree (Rooted Tree)]]**
    - **Concept**: A DAG with a single designated root where every other node has exactly one incoming edge.
    - **Binary Trees**: A specialized case where each node has at most two children.
    - **Binary Search Trees (BST)**: Applying a sorted property to enable $O(\log n)$ searching and insertion.

---
## Comparison Table

|**Structure**|**Directed?**|**Cycle-Free?**|**Main Use Case**|
|---|---|---|---|
|**General Graph**|Optional|No|Network modeling, social maps.|
|**Undirected Tree**|No|Yes|Minimum spanning trees, network topology.|
|**Rooted Tree**|Yes|Yes|File systems, HTML DOM, hierarchies.|
|**BST**|Yes|Yes|Fast data lookup and dynamic sorting.|

---
## Related Toolkits
- **Asymptotic Notation**: Measuring the efficiency of tree traversals and search operations.
- **Recursive Algorithms**: The standard paradigm for implementing most tree-based functions.
- **Searching**: Comparing the dynamic nature of BSTs against static Binary Search on arrays.