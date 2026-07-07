---
aliases:
  - Kruskal's Exchange Argument Proof
tags:
  - algorithm
  - greedy-algorithms
  - MinimumSpanningTree
  - proofs
description: Worked proof that Kruskal's Algorithm produces a minimum spanning tree, using the Exchange Argument technique.
---
> [!abstract] Abstract 
> [[Kruskal's Algorithm]] is already shown correct in its own note via the [[Cut Property]]. This note gives an **independent, second proof** of the same result, using the [[Techniques to Prove Optimality#Exchange Argument (Modify-the-Solution)|Exchange Argument]] technique instead — a worked example of applying that general technique to a specific algorithm.

---

# High Level Description

Given an undirected, connected graph with positive edge weights:

- Start with only the vertices.
- Repeat until the graph is connected:
    - Add the lightest edge that does not create a cycle.

## Alternate Description

Given a connected undirected graph $G = (V, E)$ with positive edge weights $w(e)$:

- Let $(u,v)$ be the edge with the lightest weight.
- Add $(u,v)$ to the output set.
- Create a graph $G'$ that fuses the vertices $u$ and $v$ together into one vertex.
- Repeat on $G'$ until there are no other edges.

This "fuse the endpoints together" framing is what the induction below builds on: each greedy choice shrinks the graph by one vertex, giving a clean recursive structure to induct over.

---

# Exchange Argument

Let $G = (V,E)$ be an undirected connected graph with positive edge weights. Let $g$ be the lightest edge (the first greedy choice). Let $OS$ be some arbitrary spanning tree that does **not** include $g$.

**Create $OS'$ that:**

- Must include $g$.
- Must be a spanning tree.
- Must be lighter than or equal to $OS$.

**Construction:** create $OS'$ by adding $g$ to $OS$ (this creates a cycle, since $OS$ is already a spanning tree and $g \notin OS$) and deleting the heaviest edge $h$ in that cycle.

![[Pasted image 20260706161647.png]]

**$OS'$ is a spanning tree:** it is a tree (the cycle created by adding $g$ is broken by removing $h$) and it still spans all vertices (removing an edge from a cycle never disconnects a graph, since the two endpoints remain connected via the rest of the cycle).

**$TotalWeight(OS') \leq TotalWeight(OS)$:** since we exchanged $g$ in for $h$, and $g$ is the lightest edge in the _entire graph_ — so in particular $w(g) \leq w(h)$ for the heaviest edge $h$ on that cycle — the swap can only keep the total weight the same or decrease it.

---

# Induction

**Base Case ($n=1$):** trivially true — a single vertex has no edges, so any (empty) spanning tree is optimal.

**Inductive Hypothesis:** suppose that for some $n$, Kruskal's is optimal for any graph on $n-1$ vertices.

**Inductive Step:** consider a graph $G$ with $n$ vertices. Let $OS$ be some arbitrary solution (spanning tree) of $G$. By the Exchange Argument above, there exists a solution $OS'$ that includes $g$ and has weight $\leq TotalWeight(OS)$.

Let $G'$ be the meta-graph obtained by fusing the two endpoints of $g$ together into one vertex (per the Alternate Description). Then $OS' = {g} \cup S(G')$, where $S(G')$ is the rest of $OS'$ viewed as a solution on $G'$. Since $G'$ has $n-1$ vertices, the Inductive Hypothesis gives $TotalWeight(Kruskal(G')) \leq TotalWeight(S(G'))$. By definition, $Kruskal(G) = {g} \cup Kruskal(G')$.

Putting it together:

$$ 
\begin{align*} 
	TotalWeight(OS) &\geq TotalWeight(OS') \\ 
	&= w(g) + TotalWeight(S(G')) \\ 
	&\geq w(g) + TotalWeight(Kruskal(G')) \\ 
	&= TotalWeight(Kruskal(G)) 
\end{align*} 
$$

So $TotalWeight(OS) \geq TotalWeight(Kruskal(G))$ for any spanning tree $OS$ — Kruskal's Algorithm is optimal.

> [!note] 
> On the quantity being compared This proof compares **total edge weight**, not edge count — every spanning tree on $n$ vertices has exactly $n-1$ edges regardless of which one you pick, so cardinality alone can't distinguish a minimum spanning tree from any other spanning tree. What Kruskal's actually minimizes is $TotalWeight(\cdot)$, which is why that's the quantity carried through every step of the induction above.

---

# References / Links

- [[Kruskal's Algorithm]] — the algorithm itself, plus the alternative Cut Property proof
- [[Cut Property]] — the technique used in Kruskal's own note
- [[Techniques to Prove Optimality]] — general write-up of the Exchange Argument and its alternatives