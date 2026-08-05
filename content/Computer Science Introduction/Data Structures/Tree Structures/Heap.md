---
description: "A complete binary tree structure governed by a strict priority ordering property, optimized for constant-time root access and logarithmic updates."
aliases:
  - Heap
  - Binary Heap
  - Min-Heap
  - Max-Heap
tags:
  - data-structures
  - heaps
  - priority-queues
---
> [!abstract] Abstract 
> A Heap is a specialized tree-based data structure that satisfies strict structural shape invariants and relative priority ordering properties. Bypassing the pointer overhead of traditional dynamic trees, it serves as the standard physical implementation layer for the [[Priority Queue|Priority Queue ADT]].
> 
> - **Category:** Complete Tree Index Structure
> - **Primary Backbone:** Bounded flat contiguous arrays without explicit link pointers.
> - **Key Advantage:** Fast $O(1)$ constant-time root access paired with predictable $O(\log n)$ mutation paths.

---

# Core Structural Constraints

A valid Binary Heap structure must simultaneously satisfy three core geometric rules:

1.  **Binary Tree Property:** Every individual node inside the hierarchy is restricted to a maximum of two children (maintaining 0, 1, or 2 outgoing branches).
2.  **Heap Property:** For any two nodes $A$ and $B$, if $A$ functions as the structural parent of child node $B$, then the priority value of $A$ must be higher than or equal to the priority value of $B$.
3.  **Shape Property:** The structure must operate as a **Complete Tree**. Every horizontal level of the tree must be fully populated with nodes except for the bottom-most active level, which must be packed sequentially from left to right without internal gaps.

---

# Heap Topological Classifications

Priority rankings are determined directly by evaluating element key weights. Because relative priorities define all branch paths, duplicate keys are permitted throughout the layout; internal priority ties are resolved arbitrarily.

| Architectural Dimension | Min-Heap Configuration | Max-Heap Configuration |
|---|---|---|
| **Structural Ordering** | $\text{Parent} \le \text{Children}$ | $\text{Parent} \ge \text{Children}$ |
| **Root Node Assignment** | Minimum global value (Highest Priority) | Maximum global value (Highest Priority) |
| **Priority Processing Logic** | Smaller key weights assume higher rank | Larger key weights assume higher rank |

---

# Data Structure Operations

## `Peek()`
Identifies and returns the absolute highest-priority element tracking within the collection.

- **Time Complexity:** strictly $O(1)$ constant time.
- **Algorithmic Logic:** The Heap Property guarantees that the highest-priority element always resides at the root position.

## `Push(element)` (Element Insertion)
Appends a new value to the structure while maintaining the Shape Property and the Heap Property.

- **Time Complexity:** $O(\log n)$ worst-case boundary path.

![[Pasted image 20260112154506.png]]

```pseudo
	\begin{algorithm}
	\caption{Heap Element Insertion (Bubble Up)}
	\begin{algorithmic}
		\Procedure{Push}{$element, heap, n$}
			\State $\text{heap}[n] \gets element$
			\State $curr \gets n$
			\State $n \gets n + 1$
			\While{$curr > 0$}
				\State $parent \gets \lfloor \frac{curr - 1}{2} \rfloor$
				\If{\Call{HasHigherPriority}{$\text{heap}[curr], \text{heap}[parent]$}}
					\State \Call{Swap}{$\text{heap}[curr], \text{heap}[parent]$}
					\State $curr \gets parent$
				\Else
					\Break
				\EndIf
			\EndWhile
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Pop()` (Highest-Priority Extraction)
Removes the highest-priority element from the container while preserving heap properties.

- **Time Complexity:** $O(\log n)$ worst-case boundary path.

![[Pasted image 20260112154458.png]]

```pseudo
	\begin{algorithm}
	\caption{Heap Root Extraction (Trickle Down)}
	\begin{algorithmic}
		\Procedure{Pop}{$heap, n$}
			\If{$n == 0$}
				\Return
			\EndIf
			\State $\text{heap}[0] \gets \text{heap}[n - 1]$
			\State $n \gets n - 1$
			\State $curr \gets 0$
			\While{$2 \cdot curr + 1 < n$}
				\State $left \gets 2 \cdot curr + 1$
				\State $right \gets 2 \cdot curr + 2$
				\State $target \gets left$
				\If{$right < n$ \and \Call{HasHigherPriority}{$\text{heap}[right], \text{heap}[left]$}}
					\State $target \gets right$
				\EndIf
				\If{\Call{HasHigherPriority}{$\text{heap}[target], \text{heap}[curr]$}}
					\State \Call{Swap}{$\text{heap}[curr], \text{heap}[target]$}
					\State $curr \gets target$
				\Else
					\Break
				\EndIf
			\EndWhile
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!important] Child Swap Selection Logic
> When trickling an element down, the engine **must** swap with the highest-priority child branch. Swapping with the weaker child would violate the Heap Property by leaving a child node with higher priority than its newly assigned parent.

---

# Flat Array Implementation Mapping

Because heaps satisfy the strict structural definition of a complete tree, they map directly into sequential hardware memory arrays without requiring pointers or leaving empty slots between items.

![[Pasted image 20260112154018.png]]

For an entry element located at array coordinate position $i$ under standard 0-based indexing, coordinate translations map to the following mathematical formulas:

*   **Parent Offset Location:** 
    $$\text{Parent}(i) = \left\lfloor \frac{i - 1}{2} \right\rfloor$$
*   **Left Child Offset Location:** 
    $$\text{LeftChild}(i) = 2i + 1$$
*   **Right Child Offset Location:** 
    $$\text{RightChild}(i) = 2i + 2$$
*   **Next Available Shape Slot:** Coordinates directly to array index $n$ (where $n$ represents the total active element count).

---

# Architectural Complexity Summary

*   **`Peek()` Operational Latency:** $O(1)$ constant time execution.
*   **`Push()` Insertion Latency:** $O(\log n)$ bounding log steps.
*   **`Pop()` Extraction Latency:** $O(\log n)$ bounding log steps.
*   **Total Structural Space Footprint:** Exactly $O(n)$ flat contiguous allocations.

---

# Related Notes

- [[Priority Queue|Priority Queue]]
- [[Tree Structures/Binary Search Tree|Binary Search Tree]]
- [[Binary Tree|Binary Tree]]