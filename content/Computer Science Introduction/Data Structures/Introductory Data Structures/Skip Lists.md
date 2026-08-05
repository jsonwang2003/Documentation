---
description: "A layered probabilistic linked search structure that uses randomized express layers to achieve logarithmic performance over linked lists."
aliases:
  - Skip List
  - Probabilistic Linked List
tags:
  - data-structures
  - probabilistic
  - search-optimization
---
> [!abstract] Abstract 
> Invented by William Pugh in 1989, a Skip List is a space-efficient probabilistic data structure that uses multiple layers of forward pointers to simulate a binary search over a [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|linked list]]. It avoids the $O(n)$ data-shifting penalty of sorted [[Array Lists|array lists]] while bypassing the $O(n)$ search constraints of standard sequential linked lists.
> 
> - **Category:** Probabilistic Linked Architecture
> - **Structural Composition:** Multiple stacked layers of sorted nodes connected via skip pointers.
> - **Average-Case Search Complexity:** $O(\log n)$ performance.

---

# Core Layered Topology

A Skip List organizes sorted data nodes into a vertical hierarchy of express lanes:

*   **Layer 0 (The Base Link):** The bottom-most layer is a complete, standard, sorted [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Linked List]] tracking every element in the collection.
*   **Express Layers (Layers 1 to $h$):** Each higher level tracks a sparser subset of the nodes below it, acting as shortcuts to skip over wide blocks of data during searches.
*   **The Head Node Array:** The sentinel head node holds an array of forward-facing pointers, with one pointer dedicated to each level of the hierarchy.

---

# Algorithmic Operations

## `Find(element)`
Starts at the highest level of the head node sentinel. It steps forward along the current layer until the next node's key is larger than the target or hits `NULL`, at which point it drops down one level to repeat the process.

- **Time Complexity:** $O(\log n)$ average-case; degrades to $O(n)$ in the worst case if coin-flip distributions fail.

```pseudo
	\begin{algorithm}
	\caption{Skip List Search Routine}
	\begin{algorithmic}
		\Procedure{Find}{$element, head$}
			\State $current \gets head$
			\State $layer \gets head.height$
			\While{$layer \ge 0$}
				\If{$current.key == element$}
					\Return $\text{true}$
				\EndIf
				\If{$current.next[layer] == \text{NULL}$ \or $current.next[layer].key > element$}
					\State $layer \gets layer - 1$
				\Else
					\State $current \gets current.next[layer]$
				\EndIf
			\EndWhile
			\Return $\text{false}$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Insert(key)` & `Remove(key)`
*   **Insertion Steps:** The system runs the search routine to identify the correct insertion slot at the base layer. It then uses a randomized coin-flip game to determine the vertical height of the new node, updating the forward-facing pointers of preceding neighbors across all assigned levels.
*   **Removal Steps:** The system uses the search routine to locate the target node, tracks its predecessors across all levels it occupies, and updates their pointers to bypass the removed item.

---

# Probability Mechanics & Node Height Sizing

The vertical node height distribution is managed dynamically via an internal randomized coin-flip routine to avoid requiring complex structural rebalancing operations:

### The Coin-Flip Game Strategy
1.  Start tracking at a baseline height of 0.
2.  Flip a random coin with a success probability metric $p$ (Heads).
3.  If Heads manifests, increment the tracking height by 1 and execute another flip step.
4.  If Tails manifests, stop flipping and assign the accumulated height value to the node.

This operational process fits a **Geometric Distribution**: 

$$ P(X = k) = p^k \cdot (1 - p) $$

where $k$ represents the number of sequential successes completed before encountering the first failure.

### Selecting the Probability Metric $p$
*   **Average Search Boundary:** $O(\log n)$ time.
*   **Worst-Case Boundary:** $O(n)$ time (manifests if coin flips fail to generate express layers, leaving only the base layer).
*   **Optimal Height Bound:** Typically defined as $\log_{1/p} n$.
*   **Design Tuning:** While $p = 0.5$ is standard, reducing $p$ saves memory by decreasing pointer overhead at the cost of slightly increasing the average number of search comparisons.

---

# Architectural Performance Matrix

| Metric Performance | Sorted Array List | Standard Linked List | Skip List (Average) |
|---|---|---|---|
| **Search / Find** | $O(\log n)$ | $O(n)$ | $O(\log n)$ |
| **Insert Operation** | $O(n)$ | $O(1)^*$ | $O(\log n)$ |
| **Remove Operation** | $O(n)$ | $O(1)^*$ | $O(\log n)$ |
| **Space Overhead** | $O(n)$ | $O(n)$ | $O(n)$ |

> [!note] Core Comparison Details
> Standard [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Linked List]] insertions and removals run in $O(1)$ constant time only if the system already holds a direct pointer to the target edit location. Locating that specific node profile using standard list searching still introduces an $O(n)$ linear traversal cost. The Skip List avoids this bottleneck, matching the fast search speed of a sorted array while keeping modifications efficient.

---

# Related Notes

- [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Linked List]]
- [[Array Lists|Array Lists]]