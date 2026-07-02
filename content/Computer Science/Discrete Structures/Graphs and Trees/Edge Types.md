> [!Abstract]
> In a graph, there consists of many types of edges that connects from node to node


---
# Tree Edge (Forward Edge)
A solid edge included in the [[Explore#DFS Output Tree|DFS Output Tree]], in other words, edges that have been visited by [[Depth First Search (DFS)]] or [[Explore]]. This edge leads to a **descendent**

> [!Note]
> Although **Tree Edge** and **Forward Edge** are similar in both directing an edge forward down the graph, the **Forward Edge** specifies any edge that connects to a node that may require 2 or more edges in the tree to reach said node

## Pre/Post Number Determination
$(u,v)$ is a tree/forward edge when

$$
pre(u) < pre(v) < \underbrace{ post(v) < post(u) }_{ \text{determine here} }
$$

---
# Back Edge
An edge that is not included in the [[Explore#DFS Output Tree|DFS Output Tree]] that leads to an **ancestor**

> [!Note]
> **Back Edge** is slightly different in **directed** and **undirected** graphs

Back edges in an undirected graph $G$ that has been explored are edges in $G$ that are **not** in the DFS tree of $G$ but connects vertices in the DFS output tree.

> [!Info] Removing a back edge will not disconnect the graph
> Removing an edge that is in a cycle will not disconnect an undirected graph and removing an edge that is not in a cycle will disconnect an undirected graph
## Graph Cycles Theorem
An undirected connected graph $G$ has a **cycle** if and only if it's DFS output tree has a back edge.

**Proof:**
1. If $G$ has a cycle → DFS output has a back edge
2. If DFS has a back edge → $G$ has a cycle
## Pre/Post Number Determination
$(u,v)$ is a back edge when

$$
pre(v) < pre(u) < \underbrace{ post(u) < post(v) }_{ \text{determine here} }
$$

---
# Cross Edge
An edge that is not included in the [[Explore#DFS Output Tree|DFS Output Tree]] that leads to **neither and ancestor nor descendent**

## Pre/Post Number Determination
$(u,v)$ is a cross edge when

$$
	pre(v) < \underbrace{ post(v) }_{ \text{determine here} } < pre(u) < \underbrace{ post(u) }_{ \text{determine here} }
$$

---
# Example
Given this graph:

![[Pasted image 20260629172007.png|757]]

Run [[Depth First Search (DFS)]] starting from vertex $A$, we get the following output:

|     | cc  | post | prev | pre         |
| --- | --- | ---- | ---- | ----------- |
| A   | 1   | 16   | 1    | $\emptyset$ |
| B   | 1   | 6    | 3    | C           |
| C   | 1   | 15   | 2    | A           |
| D   | 1   | 5    | 4    | B           |
| E   | 1   | 14   | 7    | C           |
| F   | 1   | 9    | 8    | E           |
| G   | 1   | 13   | 10   | E           |
| H   | 1   | 12   | 11   | G           |


From this table, we find the following [[Depth First Search (DFS)#DFS Output Forest|DFS Output Forest]]
![[Pasted image 20260629145720.png|757]]