> [!Question] Max Bandwidth Problem
> Graph represents network, with edges representing communication links
> Edge weights are bandwidth of link, what is the largest bandwidth of a path from $A$ to $H$
> ![[Pasted image 20260629201804.png]]

---
# Problem Statement
Consists of 4 sections:
1. **Instance** (Input): Directed graph $G = (V, E)$ with positive edge weights, $w(e)$, two vertices $s, t \in V$ 
2. **Solution Type** (Output): A sequence of edges
3. **Constraints** (limitations of what must is true): The sequence of edges is a path $p$ from $s$ to $t$ in $G$
4. **Objective** (Goal): Over all possible paths $p$ between $s$ and $t$, find one that maximizes Bandwidth of a path: 
$$BW(p) = \underset{ e \in p }{ min } \ w(e)$$
---
# Approaches to Solve
## Algorithm Modification (Need to be Careful)

> [!Error] Limitations
> 1. Will mess up the runtime → need to analyze again
> 2. **Required** to prove the algorithm again, since after the modification, it is not guaranteed to be correct

- Use the basic structure of [[Graph Reachability#Graph Search Algorithm|Graph Search]] and for each vertex $v$, keep track of the max bandwidth to $v$ so far
- Then move vertices into $F$ only if their max bandwidth has improved

```pseudo
	\begin{algorithm}
	\caption{Max Bandwidth Modify Algorithm Approach}
	\begin{algorithmic}
		\Procedure{MaxBand1}{$G: \text{directed graph}, s, t$}
			\State Initialize $X$ = emtpy, $F = \{s\}$
			\State B($v$) = 0 for $v \in V$
			\State B($s$) = $\infty$
			\While{$F$ is not empty}
				\State Pick $v$ in $F$
				\For{each neighbor $u$ of $v$}
					\State $m$ = min(B($v$), w($v, u$))
					\If{$m >$ B($u$)}
						\State move $u$ to $F$
						\State B($u$) = $m$
                    \EndIf
                \EndFor
                \State move $v$ from $F$ to $X$
            \EndWhile
            \Return B($t$)
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

- $B(vertex)$: bandwidth of the best path from $s$ to the vertex

### Proof of Correctness
Want to show that B($t$) is the maximum bandwidth out of all paths from $s$ to $t$

**Claim**: At the end of the algorithm, $B(v)$ is the maximum bandwidth from $s$ to $v$ for all vertices $v \in V$
1. For all vertices $v \in V$ there is a path $p$ from $s$ to $v$ such that $BW(p) = B(v)$
	**Loop Invariant**: After every iteration, for all $v$, there is a path from $s$ to $v$ such that $BW(p) = B(v)$
	- **Base Case**: Before the first iteration, $B(s) = \infty$ and $B(v) = 0$ for the rest
	- **Inductive Hypothesis**: Assume that the above claim is true after $t$ iterations
	- **Inductive Step**: Pick $v$ in $F$. Let $u$ be a neighbor of $v$. Let $m = min(B(v), w(v, u))$
		1. Case 1: $m \leq B(u)$ then the value of $B(u)$ does not change
		2. Case 2: $B(u) < m$ then the value of $B(u)$ changes to $m$
		Either way, there exists a path from $s$ to $u$ that has a bandwidth of $B(u)$
	Therefore, the loop invariant is true after every iteration including the last
	So, after the algorithm is completed, for each vertex $v$. there is a path from $s$ to $v$ with bandwidth equal to $B(v)$
2. For all vertices $v \in V$, B($v$) is the maximum bandwidth among all paths from $s$ to $v$
	Suppose by way of contradiction that there is a vertex $v$ such that there is a path $p$ from $s$ to $v$ such that $BW(p) > B(v)$. Let $b$ be the value of $BW(p)$
	Let $y$ be the first vertex in the path $p$ such that $B(y) < b$
	Let $z$ be the vertex right before $y$ ($B(z) \geq b$)
	Then $w(z,y) \geq b$ because the bandwidth of $p$ is $b$
	So when $z$ is chosen, we set $m = min(B(z), w(z, y))$. We have that $m \geq b$ and $B(y) < b$
	So the algorithm will reset $B(y) = m \geq b$. Which is a contradiction because we assumed that at the end of the algorithm, $B(y) < b$

## Reduction
> [!Abstract]
> Instead of modifying the existing algorithm, we modify the **input** so we can use the existing algorithm(s) as a **subroutine**
> 
> We map instances of one problem to instances of another. We can then use any known algorithm for that second problem as a subroutine to create an algorithm for the first

### Reduction From a Decision Version
To relate a decision problem to an optimization problem, it is helpful to **look at the decision version of an optimization problem**

**Decision Version of Max Bandwidth Problem:**
Given $G, s, t, M$, is there a path of bandwidth $M$ or better from $s$ to $t$?

```pseudo
	\begin{algorithm}
	\caption{Max Bandwidth Reduction Approach}
	\begin{algorithmic}
		\Procedure{MaxBandDecision}{$G, s, t, M$}
			\State Construct $G_M$ by removing all edges less than $M$ from $G$
			\State Run graphSearch($G_M, s$)
			\If{$t$ is visited}
				\Return \True
			\Else
				\Return \False
            \EndIf
        \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Proof of Correctness
1. If there is a path in $G$ from $s$ to $t$ with bandwidth at least $M$ then return TRUE
	**Proof**: Suppose that there is a path $p$ in $G$ from $s$ to $t$ with bandwidth at least $M$. Then every edge in $p$ has weight greater than or equal to $M$. Therefore $p$ is a path in $G_{M}$. So the algorithm will output TRUE
2. If there is not a path from $s$ to $t$ with bandwidth at least $M$ then return FALSE
	**Restating Claim** (Contrapositive): If  the algorithm returns TRUE then there is a path from $s$ to $t$ in $G$ with bandwidth at least $M$
	**Proof**: Suppose the algorithm return TRUE. Then there is a path $p$ in $G_{M}$ from $s$ to $t$. Then $p$ is a path in $G$ such that all edge weights are greater than or equal to $M$

### Time Analysis:
Let $n = |V|, m = |E|$
	Time to create $G_{M}$ is $O(n + m)$
	Time to run `graphsearch` is $O(n + m)$
Total Time: $O(n + m)$