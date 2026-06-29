
> [!ABSTRACT]
> 
> Graph reachability focuses on the ability to travel between vertices using edges. This note covers the vocabulary of connectivity, the fundamental graph search algorithm, and the specialized properties of Directed Acyclic Graphs (DAGs).

---
## Degrees and Handshaking
Understanding reachability starts with the density of connections (degrees).
### Undirected Degrees
The **degree** ($deg(v)$) of a vertex is the number of edges incident to it. A self-loop contributes twice to the degree.
- Handshake Theorem: In any undirected graph, the sum of all degrees is exactly twice the number of edges:
    $$
    2|E| = \sum_{v \in V} deg(v)
    $$
    
> [!NOTE]
> Every graph has an even number of odd-degree vertices.
### Directed Degrees
- **Indegree(v)**: Number of edges entering $v$.
- **Outdegree(v)**: Number of edges leaving $v$.
- **Summation**: $\sum Indegree(v) = \sum Outdegree(v) = |E|$.

---
## Connectivity Vocabulary
- **Walk** 
	- A sequence of edges that begins at a vertex of a graph and travels from vertex to vertex along edges of the graph
	- Describes a route from one vertex to another
	- $(v_1, e_1, v_2, e_2, v_k)$
	- Allowed to **repeat vertices and edges**
- **Trail (non-simple path)**
	- A *walk* that **doesn't repeat edges**
- **Path (simple path)**
	- A *walk* that doesn't repeat edges or vertices
- **Trivial Path**
	- Every vertex has a trivial path to itself
	- A *walk* that stays at the vertex itself
- **Length of *walk*/*path*/*trail***
	- **Number of edges** the connection have
	- *trivial path* has length $0$
- **Closed Walk**
	- A *walk* that **starts and ends at the same vertex**
- **Circuit (closed trail)**
	- A *trail* that starts and ends at the same vertex
	- length greater than $0$
- **Cycle (closed path)**
	- A *path* that **starts and ends at the same vertex**
	- length greater than $0$
- **Loop (self-loop)**
	- An edge from a vertex to itself
## Summary Comparison

| **Type**        | **Repeats Edges?** | **Repeats Vertices?** | **Closed?** |
| --------------- | ------------------ | --------------------- | ----------- |
| **Walk**        | Yes                | Yes                   | Optional    |
| **Trail**       | **No**             | Yes                   | Optional    |
| **Path**        | **No**             | **No**                | No          |
| **Closed Walk** | Yes                | Yes                   | **Yes**     |
| **Circuit**     | **No**             | Yes                   | **Yes**     |
| **Cycle**       | **No**             | **No**                | **Yes**     |

---
## Graph Search Algorithm
To find all vertices reachable from a starting vertex $s$, we partition the graph into three sets:
1. **Explored (X)**: Vertices we have finished processing.
2. **Frontier (F)**: Vertices discovered but not yet processed.
3. **Unreached (U)**: Vertices not yet seen.

![[Pasted image 20251204114925.png]]

The data structure used for the **Frontier (F)** determines the search strategy:
- **Stack**: [[Depth First Search (DFS)]]
- **Queue**: [[Computer Science/Algorithms/Graph Algorithms/Breadth First Search (BFS)]]
- **Priority Queue**: [[Dijkstra's Algorithm]]

---
## Connectedness
### Undirected Graphs
A graph is **connected** if a path exists between every pair of vertices.
- **Connected Component**: A maximal subgraph where all vertices are connected to each other.

![[Pasted image 20251204115447.png]]
### Directed Graphs
Reachability is more nuanced in directed systems:
- **Strongly Connected**: A path exists from $v$ to $w$ AND from $w$ to $v$ for all pairs.
- **Weakly Connected**: The graph is connected if all edge directions are ignored.

> [!WARNING]
> We cannot say that a directed graph is **connected** or **disconnected** → needs to be more specific

![[Pasted image 20251204120848.png]]
### Connected Components
Disconnected graphs can be broken up into pieces where each is connected
Each (**maximal**) connected piece of the graph is a **connected component**

> [!NOTE]
> **maximal** means that if you can make it any larger, it will lose the given (in this case "connected") property

![[Pasted image 20251204120220.png]]

---
## Hamiltonian Path
Visits every **vertex** exactly once. (NP-Hard to find).

---
## Eulerian Trails and Circuits
An **Eulerian trail** is a trail that visits every **edge** in the graph exactly once. 

![[Pasted image 20251204133235.png]]

If the trail starts and ends at the same vertex, it is called an **Eulerian circuit**.

![[Pasted image 20251204133443.png]]
### The Eulerian Theorem
An undirected graph $G$ (without isolated vertices) has an Eulerian trail **if and only if** $G$ is **connected** and has **at most 2 odd-degree vertices**.

> [!IMPORTANT]
> 
> From the Handshake Lemma (Sum of Degrees), every graph must contain an even number of odd-degree vertices. Therefore, a graph with an Eulerian trail will have exactly 0 or 2 odd-degree vertices.

**Summary of Conditions:**
- **Eulerian Circuit**: All vertices must have an **even degree**.
- **Eulerian Trail (not a circuit)**: Exactly **two** vertices have an **odd degree** (these serve as the start and end points).

---
### Fleury's Algorithm (Proof by Construction)
Fleury’s Algorithm provides a way to construct an Eulerian trail by following a simple rule: **Do not cross a bridge unless you have no other choice.**

![[Pasted image 20251204141308.png]]
#### Definitions
- **Bridge**: An edge which, if removed, would cause the graph $G$ to become disconnected.
- **Logic**: In an Eulerian trail, you must visit every edge on one side of a bridge before crossing it, because once you cross, there is no way to return to the previous component.
#### Step-by-Step Example

![[Pasted image 20251204135922.png]]

Suppose we have a graph where vertex 4 has an odd degree:
1. **Start**: Begin at a vertex with an odd degree (e.g., vertex 4).
2. **Select Edge**: Choose an edge to travel. If you have multiple options, prioritize edges that are **not** bridges.
    - _Example_: From vertex 4, moving to 2 or 3 is better than moving to 5 if the edge (4,5) is a bridge that isolates a section of the graph you haven't visited yet.    
3. **Traverse and Remove**: Move across the chosen edge and "remove" it from the graph (or mark it as used).
4. **Repeat**: Continue until all edges are traversed.

---
## Comparison: Eulerian vs. Hamiltonian

| **Property**    | **Eulerian Trail**             | **Hamiltonian Path**          |
| --------------- | ------------------------------ | ----------------------------- |
| **Focus**       | Every **edge** exactly once    | Every **vertex** exactly once |
| **Requirement** | Degree parity (Even/Odd count) | No simple degree-based rule   |
| **Complexity**  | Easy to find ($O(E)$)          | Hard to find (NP-Complete)    |

---
## Directed Acyclic Graphs (DAG)
A directed graph with **no cycles** (acyclic). DAGs are essential for representing dependencies.

### Test if Graph is Acyclic using [[Depth First Search (DFS)]]
1. Perform **DFS** on the graph
2. Test each edge to see if it is a [[]]
### Topological Ordering (Linearization)
An ordered list of vertices where for every edge $(v, w)$, $v$ appears before $w$ in the list.
- Only possible if the graph is a DAG.
- **Algorithm**: Repeatedly remove a **source** (vertex with Indegree 0) and add it to the list.

![[Pasted image 20251204145056.png]]
### Sources of a DAG
- Vertices with **no incoming edges** are called *sources*
	→ $A$ and $G$
- Vertices with **no outgoing edges** are called *sinks*
	→ $F$ and $I$

![[Pasted image 20251204143417.png]]

> [!IMPORTANT]
> Every finite DAG must have at least one source and **at least one sink**.

### Property of DAG

---
## Related Notes
- **[[Graphs]]**: The fundamental definition of vertices and edges.
- **[[Directed Tree (Rooted Tree)]]**: A rooted tree is a special case of a DAG with exactly one source.
- **[[Asymptotic Notation]]**: Used to analyze the $O(V + E)$ complexity of BFS and DFS.
- **[[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/index|Recursive Algorithms]]**: DFS is typically implemented as a recursive function.