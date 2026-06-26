> [!ABSTRACT]
> 
> Choosing how to store a graph in memory is a trade-off between **space** and **speed**. Depending on whether a graph is **dense** (many edges) or **sparse** (few edges), we use either an **Adjacency Matrix** or an **Adjacency List** to optimize our algorithms.

---
## 1. Density and Sparsity

To choose the right representation, we compare the number of edges $|E|$ to the maximum possible edges.

- **Maximum Edges:** In a graph with $|V|$ vertices, if every vertex connects to every other vertex (including itself), $|E| = |V|^2$.
    
- **Dense Graph:** $|E|$ is close to $|V|^2$.

![[Pasted image 20260222185147.png]]

- **Sparse Graph:** $|E|$ is significantly smaller than $|V|^2$.

![[Pasted image 20260222185154.png]]

---
## 2. The Adjacency Matrix

An **Adjacency Matrix** is a 2D array of size $|V| \times |V|$.

![[Pasted image 20260222185358.png]]

- **Logic:** The cell at $M(i, j)$ contains a `1` if there is an edge from vertex $i$ to vertex $j$, and a `0` otherwise.
- **Weights:** For weighted graphs, replace the `1`s with the actual edge costs (e.g., $M(i, j) = 5.5$).
- **Space Complexity:** $O(|V|^2)$.

> [!Example]- How would the **adjacency matrix** of an **undirected graph** look like?
> 
> **STOP and Think Answer:** The adjacency matrix of an **undirected graph** is always **symmetric** across the diagonal ($M(i, j) = M(j, i)$). This is because an edge between $i$ and $j$ works in both directions.

---
## 3. The Adjacency List

An **Adjacency List** uses an array of lists. Each index $i$ in the array corresponds to a vertex, and the list at that index contains its neighbors.

![[Pasted image 20260222185433.png]]

- **Logic:** If vertex $0$ connects to $1$ and $4$, the list at `index 0` is `[1, 4]`.
- **Weights:** Store neighbors as pairs: `(destination, weight)`.
- **Space Complexity:** $O(|V| + |E|)$.

---
## 4. Representation Comparison

| **Feature**            | **Adjacency Matrix**      | **Adjacency List**             |
| ---------------------- | ------------------------- | ------------------------------ |
| **Best For**           | Dense Graphs              | Sparse Graphs                  |
| **Space**              | $O(\|V\|^2)$              | $O(\|V\| + \|E\|)$             |
| **Edge Lookup**        | $O(1)$ (Very Fast)        | $O(\text{degree of } u)$       |
| **Find all Neighbors** | $O(\|V\|)$                | $O(\text{degree of } u)$       |
| **Add Vertex**         | $O(\|V\|^2)$              | $O(1)$                         |
| **Weighted Version**   | Replace $1$s with weights | Store pairs (neighbor, weight) |

---
## 5. Summary of Trade-offs

- **Matrices** are great when you need to know instantly if an edge exists between two nodes, but they waste a massive amount of memory on `0`s if the graph is sparse.
    
- **Lists** are the industry standard for most real-world applications (like web maps) because most real-world graphs are sparse.