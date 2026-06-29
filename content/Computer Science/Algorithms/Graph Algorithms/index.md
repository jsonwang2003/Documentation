---
title: Graph Algorithms
---
## Graphs
Graph specified by **nodes** and **edges**

$$
\begin{align*}
G &= (V, E) \ \text{where}\\
& V: \text{vertices/nodes}\\
& E: \text{edges}
\end{align*}
$$
**Directed edge**: $(x, y)$: edge from $x$ to $y$

### [[Graphs#Graph Representations|Graph Representations]]

![[Pasted image 20260404010858.png|559]]

| Adjacency Matrix                          | Adjacency List                            |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260404010715.png\|181]] | ![[Pasted image 20260404010821.png\|184]] |
#### Adjacency Matrix
An $V \times V$ matrix $A$
$$
A(i, j) = \begin{cases*} 1 & if (i, j) is in E\\ 0 & otherwise\end{cases*}
$$

  > Symmetric if $G$ undirected

- PRO: check for an edge in $\mathcal{O}(1)$ time
- CON: uses up $\mathcal{O}(V^2)$ space
#### Adjacency List
For each node, there is a list of outgoing edges

- PRO: just $\mathcal{O}(E) space$
- PRO: easily iterate through node's neighbors
- CON: check for an edge in $\mathcal{O}(V)$ time

---
## Graph Search
- Instance: a graph $G = (V, E)$ and a starting vertex $s$
- Output: a list of all vertices reachable from s by a directed path in $G$

At each point in a graph search algorithm, the vertices are partitioned into:
- $X$: explored
- $F$: frontier
- $U$: unreached

### Pseudocode
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

> [!Note]
> $X$ is a set: **Array of Booleans** indexed by vertex
> - Test Membership: $O(1)$
> - Insert: $O(1)$
> 
> $F$ is a set: **Stack** or **Queue** + **Array of Booleans**
> - Find and Delete (Pop, Dequeue, Flip Boolean): $O(1)$
> - Test Membership: $O(1)$
> - Insert (Push, Enqueue, Flip Boolean): $O(1)$
> 
> $U$ is a set: $Array of Booleans$
> - Test Membership: $O(1)$
> - Delete: $O(1)$
### Runtime Analysis

$$
\text{Runtime} = \Sigma_{w \in V}(c + c' * (out)deg(w) + c'')
$$

Since each $v$ is added to $F$ at most once, therefore each $v$ is deleted from $F$ at most once:

$$
O(\Sigma_{v\in V}(1 + (out)deg(v))) = \boxed{O(|V| + |E|)}
$$

### Correctness
#### If $v \in X$ then there is a path from $s$ to $v$
- **Loop Invariant**: After the $t^{th}$ iteration of the while loop, every element of $X$ or $F$ is reachable from $s$ in $G$
- **Base Case**: Before going through the loop, $X$ is empty and $F$ is ${s}$
- **Inductive Hypothesis**: Suppose the loop invariant is true after $t$ iterations
- **Inductive Step**: 
	1. You pick a vertex $v$ in $F$
	2. Move all neighbors of $v$ into $F$ if there are in $U$
		If there is a path from $s$ to $v$ and an edge $(v, u)$ then there is a path from $s$ to $u$
	3. Move $v$ from $F$ to $X$
		By the **IH** we know there is a path from $s$ to $v$
- Thus, it remains true that all elements of $F$ and $X$ are reachable from $s$
#### If $v \not\in X$ (by the end of the algorithm), then there is not a path from $s$ to $v$
- Suppose **by contradiction** that "there is a vertex $v$ reachable from $s$ that is not in $X$". Then there is a path from $s$ to $v$.
- Let $z$ be the last vertex in the path that is in $X$ and $w$ be the next vertex after $z$ in the path
- Then $z$ must have been in $F$ at some point. And when $z$ was picked from $F$, $w$ must have been moved from $U$ to $F$. And down the line, $w$ must have been moved from $F$ to $X$

#### Conclusion
We showed both scenarios where:
- If $v \in X$ then there is a path from $s$ to $v$
- If $v \not\in X$ then there is not a path from $s$ to $v$
Therefore we can be confident the algorithm will correctly do its job