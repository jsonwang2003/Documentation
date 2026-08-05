---
description: "A collision resolution strategy where each table slot references an external data structure, keeping keys at their natural hash address."
aliases:
  - Separate Chaining
  - Closed Addressing
  - Open Hashing
  - Bucket Hashing
tags:
  - data-structures
  - hashing
  - collision-resolution
---
> [!abstract] Abstract 
> Separate Chaining (also known as Closed Addressing) is a collision resolution strategy where each slot in the hash table points to a separate data structure—most commonly a Linked List. Unlike Open Addressing, where collisions force keys into different array slots, Separate Chaining keeps keys at their original hashed index.
> 
> - **Category:** Hash-based Priority Structure
> - **Stores:** Dynamic key-value buckets grouped by primary hash addresses.
> - **Built on top of:** Arrays and Linked Lists.
> - **Typical use cases:** High-load lookup environments, dictionary collections, symbol tables where tombstones are undesirable.

---

# Core Structure

In Separate Chaining, the primary array does not house the raw keys directly. Instead, it stores buckets which are pointers to Linked Lists.

```
Hash Table Array
[ Slot 0 ] ---> [ Key A ] ---> [ Key B ] ---> NULL
[ Slot 1 ] ---> NULL
[ Slot 2 ] ---> [ Key C ] ---> NULL
```

![[Pasted image 20260206095334.png]]

> [!tip] Key Idea
> The key is closed to its original hashed address, meaning it never moves to a different index. Conversely, the hashing is open because the data is stored outside the primary array structure.

---

# Structural Properties

*   **Invariant:** Every element $k$ matches the identity $\text{index} = H(k)$, meaning keys are kept at their original hashed index.
*   **Shape Guarantee:** Elements scale dynamically outside the table layout. Average chain length is governed by the load factor $\alpha = \frac{N}{M}$.
*   **Space Complexity:** $O(M + N)$ where $M$ is the primary array capacity and $N$ represents the total inserted nodes across all chains.
*   **Cache Property:** Does NOT guarantee immediate contiguous cache locality, as linked list nodes are scattered across system memory addresses.

---

# Data Structure Operations

## `Insert(k)`
Calculates the hash index, checks for duplicates in the list, and appends the element if no duplicate exists.

- **Time Complexity:** $O(1)$ average; $O(N)$ worst-case when all keys collide into a single chain.
- **Notes:** If the load factor threshold is breached, the structure expands using a larger prime-sized array and rehashes all elements.

```pseudo
	\begin{algorithm}
	\caption{Separate Chaining Insertion}
	\begin{algorithmic}
		\Procedure{InsertSeparateChaining}{$k, arr, n, m, loadFactorThreshold$}
			\State $index \gets$ \Call{H}{$k$}
			\If{\Call{Contains}{$arr[index], k$} == $\text{false}$}
				\State \Call{Append}{$arr[index], k$}
				\State $n \gets n + 1$
				\If{$n / m > loadFactorThreshold$}
					\State $m_{new} \gets$ \Call{NextPrime}{$2 \cdot m$}
					\State $arr_{new} \gets \text{Allocate array of size } m_{new}$
					\State \Call{RehashAll}{$arr, arr_{new}$}
					\State $arr \gets arr_{new}$
					\State $m \gets m_{new}$
				\EndIf
				\Return $\text{true}$
			\EndIf
			\Return $\text{false}$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Find(k)`
Traces the bucket chain at $H(k)$ to verify key presence.

- **Time Complexity:** $O(1 + \alpha)$ average, where $\alpha$ is the load factor.
- **Notes:** Performance slows down gracefully as lists grow longer, but lookups remain functional.

---

# Common Pitfalls

*   **Duplicate Strategy Overlooks:** Forgetting that an insert-time check slows down insertion but speeds up deletion, whereas always inserting at the head speeds up insertion but slows down deletion.
*   **Cache-Miss Penalties:** Assuming chained lists perform comparably to array-based methods on high-performance frameworks; node memory scatter can induce extensive hardware cache misses.

---

# Trade-offs Compared to Other Data Structures

| Structure Choice | Max Load Factor ($\alpha$) | Deletion Method | Cache Performance |
|---|---|---|---|
| **Separate Chaining** | Can be $> 1.0$ (Table never fills) | Simple (Standard list removal) | Poor (Nodes scattered in memory) |
| **Open Addressing** | Must be $< 1.0$ (Strictly limited) | Complex (Requires Tombstones) | Excellent (High locality) |

---

# When to Reach for This Structure

Implement Separate Chaining over Open Addressing when quick, clean deletions are required without managing lazy tombstone markers, or when table saturation must be avoided through graceful degradation.

---

# Related Notes

- [[Open Addressing (Linear Probing)|Open Addressing]]
- [[Double Hashing]]