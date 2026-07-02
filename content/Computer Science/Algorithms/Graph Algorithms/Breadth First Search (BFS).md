---
tags:
  - BreadthFirstSearch
---
 > [!ABSTRACT] **Breadth-First Search (BFS)** is the primary algorithm for finding the **shortest path in an unweighted graph**. It explores a graph layer-by-layer, ensuring that it visits every node at distance $k$ before moving on to any node at distance $k+1$. While **BFS** 

---

# The Core Logic: Layer-by-Layer Exploration
The intuition behind BFS is similar to a ripple in a pond. Starting from a source node, the search expands outward in concentric circles:

1. **Level 0:** The starting node s.
2. **Level 1:** All immediate neighbors of s.
3. **Level 2:** All neighbors of Level 1 nodes that haven’t been visited yet.

> [!Question] Why does including an "early out" option **not** improve the worst-case time complexity of a shortest path algorithm (or any algorithm, really)?
> Including an “early out” doesn’t change the **worst-case** complexity because, in the worst case, the destination node is the very last node visited (or is unreachable), forcing the algorithm to traverse the entire graph anyway.

---

# Full Graph BFS

BFS is structured as a **single iterative procedure** rather than a recursive one because the level-by-level expansion requires follows a **FIFO** order, which maps naturally to a [[Queues|queue]] rather than a call **stack**. The outer loop continuously dequeues the earliest-discovered vertex and enqueues its unvisited neighbors, ensuring every vertex at distance $d$ is processed before any vertex at distance $d+1$. This ordering is what guarantees **shortest paths** on **unweighted graphs**.

```pseudo
	\begin{algorithm}
	\caption{Breadth First Search}
	\begin{algorithmic}
		\Procedure{BFS}{$G, s$}
			\For{each vertex $u \in V$}
				\State dist($u$) = $\infty$
            \EndFor			
			\State dist($s$) = $0$
			\State $Q = [s]$
			\Comment{queue that just containing $s$}
			\While{$Q$ is not empty}
				\State $u$ = dequeue($Q$)
				\ForAll{edges $(u, v) \in E$}
					\If{dist($v$) = $\infty$}
						\State enqueue($Q, v$)
						\State dist($v$) = dist($u$) + 1
                    \EndIf
                \EndFor
            \EndWhile
	    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

- `Q` — A [[Queues|queue]] that holds the current **frontier**
	- `dequeue(Q)` ― Pop and returns the **earliest inserted** vertex 
	- `enqueue(Q, v)` ― Insert the vertex $v$ into the **queue** $Q$
- `dist(vertex)` — The distance of the between the starting vertex $s$ to the vertex from the parameter

---
# Proof of Correctness
For each vertex $v$, we want to show that $dist(v)$ is the minimum distance of all paths from $s$ to $v$. Prove by Induction:

## Claim:
For each distance value $d = 0, 1, 2 \dots$, there is a moment in the algorithm when:
1. All vertices at $\text{distance} \leq d$ from $s$ have their distance values correctly set
2. All other vertices ($\text{distance} > d$ from $s$) have distances set to $\infty$
3. The queue contains exactly the nodes at distance $d$

## Proof

### Base Case (for $d = 0$)
1. $dist(v) = 0$ is the correct distance value (only value of distance $0$ from $s$ to $s$)
2. All other vertices have distances set to $\infty$ (Initialization step)
3. The queue contains only $s$ which is the only vertex at distance $0$

### Induction Step
Let $k$ be an arbitrary integer such that $k \geq 0$. Assume that the [[#Claim|above statements]] are true for when $d = k$

Want to show that the 3 statements are true for when $d = k+1$
- All vertices distance $\leq k$ have been set and the queue only contains vertices at distance $ = k$
- Suppose $v$ is the next vertex to be popped from the queue and let $u$ be a neighbor of $v$.
	- If $dist(u) \neq \infty$, then by **Inductive Hypothesis**, $dist(u)$ has been *set correctly* and it is *not updated*
	- If $dist(u) = \infty$, then $dist(u) = dist(v) + 1 = k+1$ and is set correctly (by **Inductive Hypothesis**, since $u$ was not reachable before going through $v$, and $dist(v) = k$, the minimum distance from $s$ to $u$ must be $k+1$)

Therefore:
1. All new vertices added to the queue have distance $k+1$ and are set correctly
2. All vertices of distance $k+1$ have been added to the queue
3. The queue only contains the nodes at distance $k+1$

---
# Time & Space Complexity
> [!Info] Notice
> In **BFS**, each vertex enters the queue ($F$) **at most one time**. This was the assumption we made about [[Computer Science/Algorithms/Graph Algorithms/index#Graph Search|Graph Search]] when we calculated its runtime

This algorithm has time complexity $O(|V| + |E|)$.

Memory usage depends on the shape of the graph:
- **Wide, shallow graph** — DFS uses very little memory (proportional to the height of the DFS tree)
- **Narrow, deep graph** — DFS must store the entire long path 

| Shape         | DFS Memory                 | BFS Memory                |
| ------------- | -------------------------- | ------------------------- |
| Wide, shallow | $O(\text{Height})$ → small | $O(\text{Width})$ → large |
| Narrow, deep  | $O(\text{Height})$ → large | $O(\text{Width})$ → small |

> [!Note]
> For each vertex $v$, $dist(v)$ is set at most one time


---
# Important Drawback(s)

**BFS** *only* works to find *shortest distances* on graphs in which **each edge has equal weight**

Although we can attempt to modify **BFS** to work on weighted graphs by forming $G'$ by adding $w_{e} - 1$ many new vertices between $u$ and $v$ for every edge $e = (u, v)$ and run **BFS**, it would be **impractical** when the edge weights are large integers.

In turn, we use [[Dijkstra's Algorithm]] to deal with **weighted graphs**.