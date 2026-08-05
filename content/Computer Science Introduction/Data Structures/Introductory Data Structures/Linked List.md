---
description: "A dynamic memory collection composed of sequential node objects linked via explicit pointer addresses to provide nimble memory growth."
aliases:
  - Linked List
  - Singly-Linked List
  - Doubly-Linked List
tags:
  - data-structures
  - dynamic-memory
  - linked-lists
---
> [!abstract] Abstract 
> Developed in 1955 by Allen Newell, Cliff Shaw, and Herbert A. Simon at RAND Corporation, the linked list is a dynamically allocated data structure that grows as needed in memory. It bypasses the contiguous allocation constraints of standard [[Array Lists|Array Lists]] by linking scattered node containers via explicit system pointers.
> 
> - **Category:** Dynamic Linked Structures
> - **Core Node Anatomy:** Formed of an internal data value paired with directional address pointers.
> - **Entry Constraints:** Direct access is restricted to boundary `head` and `tail` pointers; finding interior elements requires sequential traversal.

---

# Structural Variations

Linked Lists are configured into two primary architectural variants based on pointer depth:

| Architectural Feature | Singly-Linked List | Doubly-Linked List |
|---|---|---|
| **Pointers per Node** | 1 (Points exclusively forward to the next node) | 2 (Points symmetrically to next and previous nodes) |
| **Traversal Direction** | Unidirectional (Forward only) | Bidirectional (Forward and backward) |
| **Termination Bounds** | Final node's `next` reference points to `NULL` | `head.prev` and `tail.next` point to `NULL` |

### Structural Illustrations

![[Pasted image 20260103224206.png]]

![[Pasted image 20260103224211.png]]

> [!warning] Access Limitations Complexity
> If direct structural references are limited to `head` or `tail` markers, finding a node inside a Linked List containing $n$ elements incurs an $O(n)$ linear time complexity, as the system must step through the pointer chain node-by-node.

---

# Core Operations

## Searching & Value Traversal
Finding an item or resolving an index requires sequential traversal from boundary references.

- **Time Complexity:** $O(n)$ worst-case.
- **Optimization:** In a Doubly-Linked List, if the requested index sits closer to the trailing margin, the routine can start at the `tail` and step backward to halve traversal overhead.

```pseudo
	\begin{algorithm}
	\caption{Linked List Search Algorithms}
	\begin{algorithmic}
		\Procedure{FindByElement}{$element, head$}
			\State $current \gets head$
			\While{$current \neq \text{NULL}$}
				\If{$current.data == element$}
					\Return $\text{true}$
				\EndIf
				\State $current \gets current.next$
			\EndWhile
			\Return $\text{false}$
		\EndProcedure

		\Procedure{FindByIndex}{$index, head, n$}
			\If{$index < 0$ \or $index \ge n$}
				\Return $\text{NULL}$
			\EndIf
			\State $current \gets head$
			\For{$i \gets 0 \text{ to } index - 1$}
				\State $current \gets current.next$
			\EndFor
			\Return current
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Element Insertion
Inserting an element requires locating the node preceding the target position and updating neighboring pointers to splice in the new node container.

- **Time Complexity:** $O(1)$ at boundary margins (`head`/`tail`); $O(n)$ for internal positions due to the traversal cost of locating the insertion site.

![[Pasted image 20260103230232.png]]

```pseudo
	\begin{algorithm}
	\caption{Doubly Linked List Insertion}
	\begin{algorithmic}
		\Procedure{Insert}{$newnode, index, head, tail, size$}
			\If{$index == 0$}
				\State $newnode.next \gets head$
				\State $head.prev \gets newnode$
				\State $head \gets newnode$
			\ElseIf{$index == size$}
				\State $newnode.prev \gets tail$
				\State $tail.next \gets newnode$
				\State $tail \gets newnode$
			\Else
				\State $curr \gets head$
				\For{$i \gets 0 \text{ to } index - 2$}
					\State $curr \gets curr.next$
				\EndFor
				\State $newnode.next \gets curr.next$
				\State $newnode.prev \gets curr$
				\State $curr.next \gets newnode$
				\State $newnode.next.prev \gets newnode$
			\EndIf
			\State $size \gets size + 1$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Element Removal
Bypasses a targeted node by linking its preceding and succeeding neighbors directly to each other.

- **Time Complexity:** $O(1)$ at boundary edges; $O(n)$ for internal nodes.

![[Pasted image 20260103232512.png]]

```pseudo
	\begin{algorithm}
	\caption{Doubly Linked List Removal}
	\begin{algorithmic}
		\Procedure{Remove}{$index, head, tail, n$}
			\If{$index == 0$}
				\State $head \gets head.next$
				\State $head.prev \gets \text{NULL}$
			\EndIf
			\If{$index == n - 1$}
				\State $tail \gets tail.prev$
				\State $tail.next \gets \text{NULL}$
			\Else
				\State $curr \gets head$
				\For{$i \gets 0 \text{ to } index - 2$}
					\State $curr \gets curr.next$
				\EndFor
				\State $curr.next \gets curr.next.next$
				\State $curr.next.prev \gets curr$
			\EndIf
			\State $n \gets n - 1$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!note] Memory Cleanup Realities
> In the removal diagram, the decoupled node remains stranded in system space. From a strict data structure interface perspective, this does not break functionality because the item is unreachable. However, in non-garbage-collected environments (like C++), you must explicitly delete the unlinked node to avoid memory leaks.

---

# Architectural Comparison Matrix

| Technical Feature | Linked List Implementation | Array List Implementation |
|---|---|---|
| **Access / Search Cost** | $O(n)$ linear pointer sequence traversal | $O(1)$ random access / $O(\log n)$ sorted binary search |
| **Head Insert / Delete** | $O(1)$ quick pointer reassignment swap | $O(n)$ linear data block shifting |
| **Tail Insert / Delete** | $O(1)$ direct pointer assignment | Amortized $O(1)$ capacity shifting |
| **Memory Footprint** | Dynamic growth layout; no empty pre-allocated slots | Bounded continuous chunks; can leave unused margins |
| **Pointer Overhead** | Higher cost due to storing address references | Minimal cost; tracks data elements only |

---

# Related Notes

- [[Array Lists|Array Lists]]
- [[Circular Arrays|Circular Arrays]]
- [[Skip Lists|Skip Lists]]