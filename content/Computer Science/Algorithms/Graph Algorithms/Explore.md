> [!Abstract]
> **Explore** is the same algorithm as [[Computer Science/Algorithms/Graph Algorithms/index#Graph Search|Graph Search]] where $F$ is a [[Computer Science/Data Structures/Introductory Data Structures/Stack|Stack]]. The main difference is that **Graph Search** is a [[Levels of Algorithm Design#Mid Level Design|Mid-Level Implementation]] while **Explore** is a [[Levels of Algorithm Design#Low Level Design|Low-Level Implementation]] of **Graph Search**

---
# Explore Algorithm
> [!Note] Explore vs. DFS
> The explore algorithm uses the same underlying logic as [[Depth First Search (DFS)]] and undergoes **FILO** scheme. The difference lies in **Explore** will not visit any nodes that is disconnected from the starting vertex; whereas **DFS** continues to search in disconnected vertices after fully exploring the current sub-graph.

```pseudo
	\begin{algorithm}
	\caption{Graph Search}
	\begin{algorithmic}
		\Procedure{GraphSearch}{$G, s$}
			\State $X$ = empty, $F$ = $\{ s \}$, $U = V - F$
			\While{$F$ is not empty}
				\State Pick $v$ in $F$
				\ForAll{neighbors $u$ of $v$}
					\If{$u \not\in X$ or $F$}
						\State move $u$ from $U$ to $F$
                    \EndIf
                \EndFor
                \State move $v$ from $F$ to $X$
            \EndWhile
            \Return X
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```
```pseudo
	\begin{algorithm}
	\caption{Explore}
	\begin{algorithmic}
		\Procedure{explore}{$G = (V, E), s$}
		\State visited($s$) = \True
		\For{each edge $(s, u)$}
			\If{not visited($u$)}
				\State explore($G, u$)
            \EndIf
        \EndFor
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```
The output is an array `visited` such that `visited(u)` is true if and only if **$u$ is reachable from $s$** for all vertices $u \in V$

> [!Important]
> This implementation only gives information about whether there is a path from $s$ to another vertex. However, sometimes it is helpful to know what those paths are.

---
# Keep Track of Paths
We can include **another array of information**. Set `prev(u)` to be the **"parent"** of $u$ in the DFS output tree. By also tracking when the node **enter**(`pre`) and **leave** (`post`) the stack, we can know the connected structure of *directed graphs*

```pseudo
	\begin{algorithm}
	\caption{Explore with Path Record}
	\begin{algorithmic}
	\Procedure{explore}{$G = (V, E), s$}
		\State visited($s$) = \True
		\State previsit($s$)
		\For{each edge $(s, u)$}
			\If{not visited($u$)}
				\State prev($u$) = $s$
				\State explore($G, u$)
            \EndIf
        \EndFor
        \State postvisit($s$)
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

```pseudo
	\begin{algorithm}
	\caption{previsit}
	\begin{algorithmic}
		\Procedure{previsit}{$v$} \Comment{when vertex $v$ enters the stack}
			\State pre($v$) = clock
			\State clock++
	    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

```pseudo
	\begin{algorithm}
	\caption{postvisit}
	\begin{algorithmic}
		\Procedure{postvisit}{$v$}
			\Comment{when vertex $v$ leaves the stack}
			\State post($v$) = clock
			\State clock++
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```
---
# DFS Output Tree
> [!INFO] Definition  
> A **DFS output tree** is the tree structure formed by the edges in the `prev` array after **Explore** has been performed on a graph.

When `explore` discovers an unvisited neighbor `u` from vertex `s`, it records `prev(u) = s`. This means "I reached `u` by traveling the edge `(s, u)`." Collecting all such discovery edges across the entire Explore gives the output tree — it contains only the edges Explore actually used to find new vertices. Any edge that led to an already-visited vertex is discarded.

- **Connected graph** → a single **DFS output tree**, rooted at the first vertex explored
- **Disconnected graph** → a **[[Depth First Search (DFS)#DFS Output Forest|DFS Output Forest]]**, one tree per connected component, each rooted at the vertex that triggered `cc++` 

> [!EXAMPLE]  
> If DFS visits A → B → D → C, the output tree contains edges A–B, B–D, and D–C. Any edge encountered along the way that pointed to an already-visited vertex is not included.

---
# Connected Undirected Graph
> [!Abstraction] Definition
> An undirected graph $G$ is connected if **for every pair of vertices $(v, u)$** in $G$, there exists **a path from $v$ to $u$**

**Explore** only reaches one **connected component** of the graph, namely the **set of vertices reachable from $s$**. To examine the rest of the graph, we need to restart explore on a vertex that **has not been visited** ([[Depth First Search (DFS)]])