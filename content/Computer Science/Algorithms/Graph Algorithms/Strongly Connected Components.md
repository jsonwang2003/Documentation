> [Abstract] Definition
> Two vertices $u$ and $v$ in a directed graph are **strongly connected** if there exists a path **from $u$ to $v$ *and* a path from $v$ to $u$**. The maximal set of strongly connected vertices is called a **Strongly Connected Component** (**SCC**)
> 
> Every directed graph is a [[Graph Reachability#Directed Acyclic Graphs (DAG)|DAGs]] of its strongly connected components, therefore some SCCs are **sink SCCs** and some are **source SCCs**

---
# Decomposition
There is a **linear time algorithm** that decomposes a **directed graph** into its **SCCs**. If [[Explore]] is performed on a vertex $u$, then it will visit **only the vertices that are reachable by $u$**. If the vertex $u$ is in a sink SCC, then by running `explore` on $u$ will only reach all the nodes in the SCC.

> [!Note] This suggests a way to look for SCCs
> 1. Start `explore` on a vertex in a sink SCC and visit its SCC
> 2. Remove the sink SCC from the graph and repeat

## Finding a Vertex in sink SCC
Ideally find a vertex in a sink SCC first. Unfortunately, there is no direct way to do this. The vertex with the **least post number** in a [[Explore#DFS Output Tree|DFS Output Tree]] does not necessarily belong to a sink SCC

However, there is a way to find a vertex in a source SCC.

> [!Info]
> The vertex with the **greatest post number** in any [[Explore#DFS Output Tree|DFS Output Tree]] belongs to a source SCC

### Proof
#### Property (Generalized)
If $C$ and $C'$ are strongly connected components and there is an edge from a vertex in $C$ to a vertex in $C'$, then the highest post number in $C$ is greater than the highest post number in $C'$

#### **Case 1**: DFS searches $C$ before $C'$
Then at some point [[Depth First Search (DFS)|DFS]] will cross into $C'$ and visit every edge in $C'$, then it will retrace its steps until it gets back to the first node in $C$ it started with and assign it the highest post number!

![[Pasted image 20260629195511.png]]

- $v$: the last vertex to be removed from visited (highest post number)
- $u$: a vertex that is popped before $v$ does

#### Case 2: DFS searches $C'$ before $C$
Then [[Depth First Search (DFS)|DFS]] will visit all vertices of $C'$ before getting stuck and assign a post number to all vertices of $C'$. Then it will visit some vertex of $C$ later and assign post numbers to those vertices

![[Pasted image 20260629195914.png|757]]

- $u$ is popped first, then later $v$ is pushed and popped after ($post(v) > post(u)$)

#### Conclusion
The SCCs can be **linearized** by arranging them in *decreasing order* of their highest post numbers

---
# How to Find Sink SCCs
Given a graph $G$, let $G^{R}$ be the **reverse graph of $G$**. Then the sources of $G^{R}$ are the sinks of $G$.

So if we perform DFS on $G^{R}$ then the vertex with the highest post number is in a source in $G^{R}$. This means that this vertex will be in a sink of $G$

So start with this vertex and explore the SCC

Then the vertex with the next greatest post number in $G^{R}$ is in the next SCC in linear order so start with that one next.

---
# SCC Algorithm
> [!Summary] SCC Algorithm
> 1. Construct $G^{R}$
> 2. Run DFS on $G^{R}$ and keep track of the post number.
> 3. Run DFS on $G$ and order the vertices in decreasing order of the post numbers from the previous step. Every time DFS increments `cc`, you have found a new SCC!

## Time Complexity
1. Construct $G^{R}$ → $O(|V| + |E|)$
2. DFS on $G^{R}$ → $O(|V| + |E|)$
3. DFS on $G$ → $O(|V| + |E|)$
Therefore the time complexity of the SCC algorithm is linear time: $O(|V| + |E|)$

