> [!Abstract]
> In a graph, there consists of many types of edges that connects from node to node


---
# Tree Edge (Forward Edge)
A solid edge included in the [[Explore#DFS Output Tree|DFS Output Tree]], in other words, edges that have been visited by [[Depth First Search (DFS)]] or [[Explore]]. This edge leads to a **descendent**

## Pre/Post Number Determination
$(u,v)$ is a tree/forward edge when

$$
pre(u) < pre(v) < post(v) < post(u)
$$

---
# Back Edge
An edge that is not included in the [[Explore#DFS Output Tree|DFS Output Tree]] that leads to an **ancestor**

---
# Cross Edge