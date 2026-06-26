> [!ABSTRACT]
> 
> An undirected tree is a connected graph that contains no cycles. This structural simplicity leads to unique properties regarding edges, vertices, and paths, making them foundational in both graph theory and algorithm design.

![[Pasted image 20251204112801.png]]

---
## Facts and Definitions
- **Leaves**: Vertices of **degree 1**. If the tree consists of only a single vertex, that vertex is considered to have a **degree of 0**.
- **Forest**: A set or collection of disjoint trees. A forest is acyclic but not necessarily connected.
- **Unique Path Property**: Between any pair of distinct vertices in a tree, there is exactly **one simple path**.

---
## The Existence of Leaves
A tree with $n \geq 2$ vertices will **always** have at least one vertex of degree 1 (a leaf).

Proof by Contradiction:
Suppose there exists a tree with $n \geq 2$ vertices where every vertex has a degree of at least 2.
1. Start at an arbitrary vertex $v_1$ and move to a neighbor $v_2$.
2. Since $v_2$ has a degree $\geq 2$, there must be an edge to a vertex $v_3$ other than $v_1$.
3. Continue this process to form a walk: $(v_1, v_2, \dots, v_n, v_{n+1})$.
4. By the **Pigeonhole Principle**, in a walk of $n+1$ vertices where there are only $n$ unique vertices available, at least one vertex must be repeated.
5. The repetition of a vertex in this walk implies the existence of a **cycle**, which contradicts the definition of a tree.

> [!NOTE]
> 
> Pigeonhole Principle
> 
> If you have more items ("pigeons") than containers ("pigeonholes"), at least one container must contain more than one item.

---
## The Tree Edge Theorem
Every tree with $n$ vertices has exactly **$n-1$ edges**.
**Proof by Induction**:
- **Base Case**: If a tree has $n=1$ vertex, it has $1-1 = 0$ edges. This is true by definition.
- **Induction Hypothesis**: Suppose that all trees with $n$ vertices have $n-1$ edges for some $n \geq 1$.
- **Induction Step**: Consider an arbitrary tree $G$ with $n+1$ vertices.
    1. From our previous proof, we know $G$ must have at least one leaf vertex $L$.
    2. Remove the leaf vertex $L$ and its single incident edge.
    3. The remaining graph $G'$ is still connected and acyclic, meaning it is a tree with $n$ vertices.
    4. By the **Induction Hypothesis**, $G'$ has $n-1$ edges.
    5. Re-attaching the leaf and its edge to $G'$ returns us to the original graph $G$.
    6. Total edges in $G = (n-1) + 1 = \mathbf{n}$ edges.

---
## Summary of Properties

|**Property**|**Tree (n vertices)**|**Forest (k trees, n vertices)**|
|---|---|---|
|**Connectivity**|Connected|Disjoint Components|
|**Acyclic**|Yes|Yes|
|**Edge Count**|$n-1$|$n-k$|
|**Simple Paths**|1 between any two nodes|Max 1 (0 if in different trees)|

---
## Related Notes
- **[[Graphs]]**: Trees are a specific subclass of undirected graphs.
- **[[Directed Tree (Rooted Tree)]]**: Adding direction and a root to an undirected tree creates a hierarchy.
- **[[Graph Reachability]]**: The unique path property ensures that reachability in a tree is simple and unambiguous.
- **[[Recursive Proofs]]**: Induction is the primary tool used to prove the structural properties of trees.