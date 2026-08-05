> [!ABSTRACT]
> 
> A graph is a mathematical structure used to model pairwise relations between objects. It consists of a set of vertices (nodes) and a set of edges connecting them. Graphs are fundamental to computer science, used in everything from network routing to social media analysis.

---
## Types of Graphs
### Directed Graphs
A **directed graph** (or digraph) consists of:
- A nonempty **set of vertices** $V$.
- A **set of directed edges** $E$. Each edge is an **ordered pair** $(u, v)$, representing a one-way connection from vertex $u$ to vertex $v$.
### Undirected Graphs
An **undirected graph** consists of:
- A nonempty **set of vertices** $V$.
- A **set of undirected edges** $E$. Each edge is an **unordered pair** $\{u, v\}$, representing a two-way connection between $u$ and $v$.

---
## Special Types of Undirected Graphs
### Complete Graph ($K_n$)
A complete graph is an undirected graph where **every pair of distinct vertices is connected by a unique edge**.
- **Density**: This is the densest possible simple graph.
- Edge Count: Since every vertex connects to every other vertex, the total edges are:
    $$
    \boxed{|E| = \binom{n}{2} = \frac{n(n-1)}{2}}
    $$

![[Pasted image 20251204111955.png]]
### Bipartite Graph
A graph whose vertices can be divided into two disjoint sets $V_1$ and $V_2$ such that **every edge connects a vertex in $V_1$ to one in $V_2$**.
- **Constraint**: No two vertices within the same set are adjacent.
- **Use Case**: Modeling relationships between two different categories (e.g., Students and Classes).

---
## [[Computer Science Introduction/Data Structures/Tree Structures/index|Trees]]
Trees are a highly restricted but extremely common subset of graphs.
### Undirected Tree
An undirected graph that is **connected** and **contains no cycles**.
- See more: [[Undirected Trees]]
### Directed Tree (Rooted Tree)
A directed graph where one vertex is the **root** (in-degree 0) and every other vertex has **exactly one incoming edge** (in-degree 1).
- See more: [[Directed Tree (Rooted Tree)]]
---
## Simple Labeled Graphs
A graph is considered **simple** if it lacks the following complexities:
- **Self-Loops**: An edge that connects a vertex to itself.
- **Parallel Edges**: Multiple edges between the same two vertices (in the same direction for directed graphs).

![[Pasted image 20251203220426.png]]
### Encoding Simple Directed Graphs
For $n$ vertices, each vertex can potentially point to any of the other $n-1$ vertices.
- **Potential Edges**: $n(n-1)$
- **Total Possible Graphs**: $2^{n(n-1)}$
- **Bits Required**: $n(n-1)$ bits.
### Encoding Simple Undirected Graphs
For $n$ vertices, an edge can exist between any unique pair of vertices.
- **Potential Edges**: $\binom{n}{2} = \frac{n(n-1)}{2}$
- **Total Possible Graphs**: $2^{\binom{n}{2}}$
- **Bits Required**: $\binom{n}{2}$ bits.

---
## Graph Representations
The choice of representation depends on the **density** of the graph (how many edges exist relative to the number of vertices).

| **Representation**   | **Memory Efficiency** | **Best Use Case**         | **Graph for Best use Case**          |
| -------------------- | --------------------- | ------------------------- | ------------------------------------ |
| **Adjacency Matrix** | $O(n^2)$              | Dense Graphs (many edges) | ![[Pasted image 20251203223626.png]] |
| **Adjacency List**   | $O(n +E)$             | $\|E\|$                   | ![[Pasted image 20251203223641.png]] |

### Adjacency Matrix
An $n \times n$ matrix where the entry at row $i$, column $j$ represents the connection from vertex $i$ to vertex $j$.
- **Simple Graphs**: Entries are $0$ or $1$.
- **Parallel Edges**: Entries can be integers $>1$.
- **Self-Loops**: Indicated by non-zero values on the main diagonal.
- **Undirected Symmetry**: In an undirected graph, the matrix is symmetric ($M_{ij} = M_{ji}$), making the lower triangle a mirror of the upper triangle.
### Adjacency List
A collection of lists where each vertex $v$ is associated with a list of its neighbors. In a directed graph, this usually stores **outgoing neighbors**.
- **Adjacent**: Two vertices connected by an edge are said to be adjacent.
- **Neighbors (Undirected)**: $\{w : \{v, w\} \in E\}$
- **Outgoing Neighbors (Directed)**: $\{w : (v, w) \in E\}$
- **Incoming Neighbors (Directed)**: $\{w : (w, v) \in E\}$

---
## Related Notes
- **[[Graph Reachability]]**: These structures determine how search algorithms (BFS/DFS) behave.
- **[[Lossless Encoding]]**: How we calculate the bits needed to store these specific graph types.
- **[[Undirected Trees]]**: Deep dive into the mathematical properties of acyclic connected graphs.