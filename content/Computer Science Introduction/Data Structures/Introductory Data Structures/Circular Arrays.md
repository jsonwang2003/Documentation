---
description: "An array optimization utilizing modular index arithmetic to simulate a continuous ring, providing constant-time end operations."
aliases:
  - Circular Array
  - Ring Buffer
  - Circular Buffer
tags:
  - data-structures
  - arrays
  - memory-management
---
> [!abstract] Abstract 
> A Circular Array is a regular [[Array Lists|Array List]] with an implementation that mimics the boundary properties of a [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Linked List]]. It treats the underlying linear array as a continuous ring by tracking `head` and `tail` indices.
> 
> - **Head/Tail Indices:** Instead of starting at index 0, the first element sits at the `head` index and the last sits at the `tail` index.
> - **Contiguity:** Elements remain contiguous in logical sequence, even when wrapping across physical array boundaries.

---

# Wrapping Logic

*   **Add to End:** Increment `tail`. If it hits the physical end of the array, it wraps back to index 0.
*   **Add to Front:** Decrement `head`. If it hits `-1`, it wraps to the final index ($\text{capacity} - 1$).

---

# System Representations

There are two primary ways to conceptualize a Circular Array. Both are equally valid and describe the same underlying logic.

### 1. The Physical (Linear) View
This representation illustrates how data literally sits in computer memory addresses. The `head` and `tail` indices move across the flat array, wrapping around upon hitting boundaries.

*   *Wrap to Front:* If `tail` reaches capacity, it wraps to index 0.
*   *Wrap to Back:* If `head` drops below 0, it wraps to index $\text{capacity} - 1$.

![[Pasted image 20260103235624.png]]

### 2. The Logical (Circular) View
Since operational focus centers on an element's location relative to `head` and `tail`, it is often visualized as a continuous ring structure.

![[Pasted image 20260104003529.png]]

---
# Operations
## Insertion and Resizing

When the backing array becomes fully saturated, it must be resized. Similar to a standard [[Array Lists|Array List]], the engine doubles array capacity and copies existing entries.

![[Pasted image 20260104015800.png]]

During a resize operation, the circular layout must be "unrolled" so the new array begins with the `head` element aligned at physical index 0.

```pseudo
	\begin{algorithm}
	\caption{Circular Array Operations}
	\begin{algorithmic}
		\Procedure{CheckSize}{array, head, tail, n}
			\If{$n == array.\text{length}$}
				\State $newArray \gets \text{Allocate empty array of length } 2 \cdot array.\text{length}$
				\For{$i \gets 0 \text{ to } n - 1$}
					\State $newArray[i] \gets array[(head + i) \pmod{array.\text{length}}]$
				\EndFor
				\State $array \gets newArray$
				\State $head \gets 0$
				\State $tail \gets n - 1$
			\EndIf
		\EndProcedure

		\Procedure{InsertFront}{element, array, head, n}
			\State \Call{CheckSize}{array, head, tail, n}
			\State $head \gets head - 1$
			\If{$head == -1$}
				\State $head \gets array.\text{length} - 1$
			\EndIf
			\State $array[head] \gets element$
			\State $n \gets n + 1$
		\EndProcedure

		\Procedure{InsertBack}{element, array, tail, n}
			\State \Call{CheckSize}{array, head, tail, n}
			\State $tail \gets tail + 1$
			\If{$tail == array.\text{length}$}
				\State $tail \gets 0$
			\EndIf
			\State $array[tail] \gets element$
			\State $n \gets n + 1$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Removal Operations

*   **Remove Front:** Erase the element at `head` and increment the `head` index (wrapping if necessary).
*   **Remove Back:** Erase the element at `tail` and decrement the `tail` index (wrapping if necessary).

```pseudo
	\begin{algorithm}
	\caption{Circular Array Removal}
	\begin{algorithmic}
		\Procedure{RemoveFront}{array, head, n}
			\State $head \gets (head + 1) \pmod{array.\text{length}}$
			\State $n \gets n - 1$
		\EndProcedure

		\Procedure{RemoveBack}{array, tail, n}
			\State $tail \gets tail - 1$
			\If{$tail == -1$}
				\State $tail \gets array.\text{length} - 1$
			\EndIf
			\State $n \gets n - 1$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!note] Memory Cleanup Considerations
> Explicitly clearing unlinked array slots during removal is usually unnecessary because values are overwritten by future insertions. However, if the array stores raw pointers in non-garbage-collected environments, elements must be explicitly deallocated to prevent memory leaks.
## Finding and Random Access

Unlike a [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Linked List]], a Circular Array retains $O(1)$ Random Access capabilities. To access logical element $i$, calculate its physical index via modular arithmetic:

$$ \text{Physical Index} = (head + i) \pmod{array.\text{length}} $$

![[Pasted image 20260104013933.png]]

> [!tip] Modulo Omission Optimization
> Modulo arithmetic can be bypassed whenever $(head + i) < array.\text{length}$, speeding up raw index calculations.

---

# Performance Summary

*   **Random Access:** $O(1)$ achieved via modular offset arithmetic.
*   **Insert/Remove Front:** $O(1)$ requiring zero data shifting.
*   **Insert/Remove Back:** $O(1)$ direct index pointer adjustment.
*   **Resize Complexity:** $O(n)$ occurring infrequently (amortized $O(1)$).

**Conclusion:** Circular Arrays provide the efficient boundary manipulation of a [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Linked List]] while retaining the constant-time random access of an [[Array Lists|Array List]]. This makes them ideal for backing [[Deques|Deques]] and [[Queues|Queues]].

---

# Related Notes

- [[Array Lists|Array Lists]]
- [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|Linked List]]
- [[Deques|Deques]]
- [[Queues|Queues]]