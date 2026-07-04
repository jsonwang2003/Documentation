---
title: Graph Algorithms
description: Foundational graph definitions, representations, and the generic Graph Search algorithm that DFS, BFS, and Explore each specialize
tags:
  - graph-algorithms
aliases:
  - Graph Algorithms
---
> [!abstract] Overview 
> Foundational definitions and the generic search procedure underlying every graph traversal algorithm in this vault. [[Depth First Search (DFS)]], [[Breadth First Search (BFS)]], and [[Explore]] are each a specific choice of data structure for the frontier $F$ in the [[#Graph Search]] algorithm below.

---

# Foundational Concepts

## Graphs

A graph is specified by **nodes** and **edges**:

$$ 
\begin{align*} 
G &= (V, E) \ \text{where}\ & V: \text{vertices/nodes}\ & E: \text{edges} \end{align*} 
$$

**Directed edge:** $(x, y)$ — an edge from $x$ to $y$.

## Graph Representations

![[Pasted image 20260404010858.png|559]]

|Adjacency Matrix|Adjacency List|
|---|---|
|![[Pasted image 20260404010715.png\|181]]|![[Pasted image 20260404010821.png\|184]]|

### Adjacency Matrix

A $V \times V$ matrix $A$:

$$ 
A(i, j) = \begin{cases} 
1 & if (i, j) \in E\\
0 & otherwise
\end{cases} 
$$

> Symmetric if $G$ is undirected.

- **PRO:** check for an edge in $O(1)$ time
- **CON:** uses up $O(V^2)$ space

### Adjacency List

For each node, there is a list of outgoing edges.

- **PRO:** just $O(E)$ space
- **PRO:** easily iterate through a node's neighbors
- **CON:** check for an edge in $O(V)$ time

---

# Graph Search
Graph Search is a core foundational outline of most graph algorithms, serving as a template for algorithms such as [[Depth First Search (DFS)|DFS]], [[Breadth First Search (BFS)|BFS]], [[Dijkstra's Algorithm|Dijkstra's]], etc.

- **Instance:** a graph $G = (V, E)$ and a starting vertex $s$
- **Output:** a list of all vertices reachable from $s$ by a directed path in $G$

At each point in a graph search algorithm, the vertices are partitioned into:

- $X$: explored
- $F$: frontier
- $U$: unreached

## Pseudocode

```pseudo
	\begin{algorithm}
	\caption{Graph Search}
	\begin{algorithmic}
		\Procedure{GraphSearch}{$G, s$}
			\State $X$ = empty, $F$ = $\{ s \}$, $U = V - F$
			\While{$F$ is not empty}
				\State Pick $w$ in $F$
				\ForAll{$(w, y) \in E$}
					\If{$y \not\in X$ or $F$}
						\State move $y$ from $U$ to $F$
                    \EndIf
                \EndFor
                \State move $w$ from $F$ to $X$
            \EndWhile
            \Return X
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!Note] Implementing $X$, $F$, $U$ $X$ is a set: **Array of Booleans** indexed by vertex
> 
> - Test Membership: $O(1)$
> - Insert: $O(1)$
> 
> $F$ is a set: **Stack** or **Queue** + **Array of Booleans**
> 
> - Find and Delete (Pop, Dequeue, Flip Boolean): $O(1)$
> - Test Membership: $O(1)$
> - Insert (Push, Enqueue, Flip Boolean): $O(1)$
> 
> $U$ is a set: **Array of Booleans**
> 
> - Test Membership: $O(1)$
> - Delete: $O(1)$
> 
> Choosing $F$ = stack gives [[Depth First Search (DFS)]] / [[Explore]]; choosing $F$ = queue gives [[Breadth First Search (BFS)]].

## Runtime Analysis

$$ \text{Runtime} = \sum_{w \in V}(c + c' \cdot (out)deg(w) + c'') $$

Since each $v$ is added to $F$ at most once, each $v$ is also deleted from $F$ at most once:

$$ O\left(\sum_{v\in V}(1 + (out)deg(v))\right) = \boxed{O(|V| + |E|)} $$

## Correctness

**If $v \in X$, then there is a path from $s$ to $v$:**

- **Loop Invariant:** after the $t^{th}$ iteration of the while loop, every element of $X$ or $F$ is reachable from $s$ in $G$.
- **Base Case:** before the loop, $X$ is empty and $F = {s}$.
- **Inductive Hypothesis:** suppose the loop invariant is true after $t$ iterations.
- **Inductive Step:**
    1. Pick a vertex $v$ in $F$.
    2. Move all neighbors of $v$ into $F$ if they're in $U$ — if there's a path from $s$ to $v$ and an edge $(v,u)$, then there's a path from $s$ to $u$.
    3. Move $v$ from $F$ to $X$ — by the IH, there is a path from $s$ to $v$.
- Thus, it remains true that all elements of $F$ and $X$ are reachable from $s$.

**If $v \notin X$ by the end of the algorithm, then there is not a path from $s$ to $v$:**

- Suppose by contradiction that there is a vertex $v$ reachable from $s$ that is not in $X$. Then there is a path from $s$ to $v$.
- Let $z$ be the last vertex in the path that is in $X$, and $w$ be the next vertex after $z$ in the path.
- Then $z$ must have been in $F$ at some point. When $z$ was picked from $F$, $w$ must have been moved from $U$ to $F$. And down the line, $w$ must have been moved from $F$ to $X$ — contradicting that $v \notin X$.

**Conclusion:** since both directions hold — $v \in X \iff$ there is a path from $s$ to $v$ — the algorithm correctly computes exactly the set of vertices reachable from $s$.

---

# Notes in This Section

| Note                                                                   | Desciption                                                                                                                          |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| [[Depth First Search (DFS)]]                                           | Graph Search with $F$ = stack (via recursion), run over all vertices to cover disconnected components                               |
| [[Explore]]                                                            | The low-level, single-component recursive implementation of Graph Search with $F$ = stack                                           |
| [[Breadth First Search (BFS)]]                                         | Graph Search with $F$ = queue; guarantees shortest paths on unweighted graphs                                                       |
| [[Dijkstra's Algorithm]]                                               | Graph Search with $F$ = priority queue keyed by cumulative distance; handles weighted graphs (non-negative weights)                 |
| [[Prim's Algorithm]]                                                   | Same loop shape as Dijkstra's, but $F$ is keyed by single-edge cost to build a minimum spanning tree instead                        |
| [[Kruskal's Algorithm]]                                                | A different greedy MST approach — sorts all edges globally and uses [[Disjoint Sets & Up-Trees]] instead of a frontier-based search |
| [[Strongly Connected Components\|Strongly Connected Components (SCC)]] | Two-pass DFS (with a reversed graph) that decomposes a directed graph into its strongly connected components                        |

---
# Related Categories

- [[Minimum Spanning Trees]]
- [[Levels of Algorithm Design]]