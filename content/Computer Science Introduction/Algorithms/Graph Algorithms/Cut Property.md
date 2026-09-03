---
description: "MST lemma: the lightest edge crossing any cut consistent with a partial MST can always be added and still be part of some MST. Basis for Prim's and Kruskal's correctness"
tags:
  - MinimumSpanningTree
  - lemma
aliases:
  - Cut Property
---
# Statement

Suppose $G' = (V, T)$ is a [MST](Minimum Spanning Trees) of $G = (V,E)$, and suppose $X \subseteq T$. Pick any subset of vertices $S \subseteq V$ such that there is no path from $S$ to $V-S$ using edges from $X$ (i.e. $X$ contains no edge crossing this cut). Let $e \in E$ be the lightest edge that connects $S$ to $V-S$.

**Then:** $X \cup \{e\}$ is part of some MST.

## Proof

### Case 1: $e \in T$

Then $X \cup \{e\} \subseteq T$, and we assumed that $T$ is the edge set of an MST, so $X \cup \{e\}$ is part of that MST.

### Case 2: $e \notin T$

Consider $T \cup \{e\}$. Since $T$ is the edge set of a connected tree, $T \cup \{e\}$ is the edge set of a graph that has a cycle, and that cycle contains $e$.

![[Pasted image 20260702224559.png]]

Since the cycle contains $e$ (which crosses the cut between $S$ and $V-S$) and a cycle must return to where it started, it has to cross back over that same cut — so there must be **another** edge $e' \in T$ on this cycle that also connects $S$ to $V-S$.

Consider the edge set $T \cup \{e\} - \{e'\}$. This is still a tree — removing $e'$ breaks the cycle we just created, leaving a connected, acyclic graph on the same vertex set.

We assumed $w(e) \leq w(e')$ (since $e$ is the _lightest_ edge connecting $S$ to $V-S$, and $e'$ is some edge connecting $S$ to $V-S$). So:

- $cost(T \cup \{e\} - \{e'\}) = cost(T) + w(e) - w(e')$
- $cost(T \cup \{e\} - \{e'\}) \leq cost(T)$

But $T$ is the edge set of an MST, so it is already minimal. Therefore $cost(T \cup \{e\} - \{e'\})$ must be minimal also, and since we showed $cost(T \cup \{e\} - \{e'\}) = cost(T) + w(e) - w(e') \leq cost(T)$, it must in fact equal $cost(T)$. So $T \cup \{e\} - \{e'\}$ is also the edge set of an MST.

Finally, since $X$ has no edges crossing the cut $(S, V-S)$ (by assumption) and $e'$ crosses that cut, $e' \notin X$. So:

$$
X \cup {e} \subseteq T \cup \{e\} - \{e'\}
$$

and the right-hand side is the edge set of an MST, so $X \cup \{e\}$ is part of some MST. 

---

# Using the Cut Property to Prove Prim's and Kruskal's

The cut property is what lets us prove both [[Prim's Algorithm|Prim's]] and [[Kruskal's Algorithm|Kruskal's]] correct, by induction.

**Claim:** After each iteration of **Prim's** / **Kruskal's**, the set of edges $X$ is a subset of some MST.

**Base Case:** Both algorithms start with $X$ empty, so vacuously, $X$ is a subset of some MST.

## Prim's

Part way through Prim's, $X$ is a tree, and the next edge selected is the lightest edge that connects $X$ to the rest of the vertices. On step $n$:

- Prim's essentially partitions the set of vertices based on whether they are in the tree — this is exactly the cut $(S, V-S)$, with $S$ = vertices currently in $X$.
- Then it picks the lightest edge $e$ that connects the two subsets $S$ and $V-S$.

This is precisely the setup of the cut property, so $X \cup \{e\}$ remains a subset of some MST at every step.

## Kruskal's

Part way through Kruskal's, $X$ is a forest, and the next edge selected is the lightest edge that connects two trees in the forest.

- Kruskal's finds the lightest edge $e$ overall, then partitions the vertex set based on that edge (i.e. $S$ = the tree in the forest containing one endpoint of $e$, $V-S$ = everything else).
- Since $e$ is the lightest edge overall, it must also be the lightest edge that connects those two particular subsets $S$ and $V-S$.

So the same cut property applies here too — Kruskal's just discovers the cut _after_ picking $e$, instead of fixing the cut first like Prim's does.

---

# References / Links

- [[Prim's Algorithm]]
- [[Kruskal's Algorithm]]
- [[Minimum Spanning Trees]]