---
description: "A sequential probing collision management strategy that keeps elements confined inside the table array boundaries."
aliases:
  - Linear Probing
  - Open Addressing
  - Closed Hashing
tags:
  - data-structures
  - hashing
  - collision-resolution
---
> [!abstract] Abstract 
> Linear Probing is a collision resolution strategy within the Open Addressing (or Closed Hashing) family. When a key's natural hash index is occupied, the algorithm "probes" the very next sequential slot in the array. This continues until an empty slot is found or the table is determined to be full.
> 
> - **Category:** Open Addressing (Closed Hashing)
> - **Solves:** Collision resolution within the boundaries of the backing array.
> - **Typical use cases:** Fast lookup tables optimizing hardware sequential memory access.

---

# Core Concepts

### Open Addressing vs. Closed Hashing
*   **Open Addressing:** The address of a key is not fixed; it is "open" to moving to a different index than its original hash value.
*   **Closed Hashing:** The key must stay "closed" within the physical boundaries of the backing array.

### Primary Clustering
The core disadvantage of Linear Probing. As array slots fill up, sequential clumps of adjacent keys form. These clumps are statistically more likely to grow because any key that hashes anywhere inside the clump is forced to step to the end of it, degrading constant-time operations into linear scans.

---

# How It Works

Linear Probing follows a simple deterministic path: if $H(k)$ is occupied, try $(H(k) + 1) \pmod M$, then $(H(k) + 2) \pmod M$, and so on:

$$
\text{index} = (\text{index} + 1) \pmod M 
$$

![[Pasted image 20260206100526.png]]

> [!tip] Key Idea
> Probing sequential slots yields excellent hardware cache locality, but it directly increases the probability of primary clustering as the table load factor scales.

---

# Algorithm Walkthroughs

## `Insert(k)`
Scans linearly through consecutive array cells until an empty slot, a tombstone, or a duplicate key is found.

```pseudo
	\begin{algorithm}
	\caption{Linear Probing Insertion}
	\begin{algorithmic}
		\Procedure{InsertLinearProbe}{$k, arr, m$}
			\State $index \gets$ \Call{H}{$k$}
			\State $start \gets index$
			\While{$\text{true}$}
				\If{$arr[index] == k$}
					\Return $\text{false}$
				\EndIf
				\If{$arr[index] == \text{NULL or } arr[index] == \text{TOMBSTONE}$}
					\State $arr[index] \gets k$
					\Return $\text{true}$
				\EndIf
				\State $index \gets (index + 1) \pmod m$
				\If{$index == start$}
					\State \Call{ResizeAndRehash}{$arr$}
					\State $index \gets$ \Call{H}{$k$}
					\State $start \gets index$
				\EndIf
			\EndWhile
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# The Deletion Problem: Lazy Deletion

You cannot simply set a slot to `NULL` when deleting a key because doing so would break the probe chain for other collided keys positioned further down the sequence.

*   **Solution (Tombstones):** Instead of clearing the slot directly, the index is marked with a special `TOMBSTONE` marker.
*   **Behavior:** A `Find` operation treats a tombstone as occupied and continues probing downstream. An `Insert` operation treats a tombstone as empty and can overwrite it with a new key.

---

# Performance Summary

| Feature Parameter | Linear Probing Specification |
|---|---|
| **Average Time Complexity** | $O(1)$ |
| **Worst-Case Time Complexity** | $O(N)$ |
| **Cache Locality** | Excellent (Contiguous sequential memory access) |
| **Main Weakness** | Primary Clustering |
| **Deletion Strategy** | Lazy Deletion via Tombstones |

---

# Related Notes

- [[Closed Addressing (Separate Chaining)|Separate Chaining]]
- [[Double Hashing]]
- [[Random Hashing]]
- [[Cuckoo Hashing]]