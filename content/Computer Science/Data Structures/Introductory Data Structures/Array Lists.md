---
description: "A contiguous, homogeneous memory structure that packs elements sequentially to provide constant-time random access alongside dynamic expansion."
aliases:
  - Array List
  - Vector
  - Dynamic Array
tags:
  - data-structures
  - arrays
  - memory-management
---
> [!abstract] Abstract 
> An Array List (often called a Vector) is a dynamically resizing array where no empty slots are present between elements. Users can only insert elements at contiguous indices between $0$ and $n$ inclusive (where $n$ represents the total element count), optimizing the structure for high-speed constant-time index lookups.
> 
> - **Category:** Bounded Contiguous Structures
> - **Backbone Layout:** Contiguous blocks of homogeneous computer memory.
> - **Key Advantage:** Constant-time data location arithmetic using raw offsets.

---

# Properties of a Backing Array

An array is a homogeneous data structure where each element is stored in adjacent memory locations. Homogeneous means all elements are of the exact same data type (e.g., `int`, `double`) and share an identical byte size $b$.

### Random Access Arithmetic
Because each cell shares the same size $b$ and layout blocks are perfectly contiguous in hardware memory, the system calculates the location of any element $i$ in constant time given the starting base address $x$:

$$ \text{Address}(i) = x + b \cdot i $$

> [!note] Memory Address Calculation Example
> Suppose an array of integers ($b = 4 \text{ bytes}$) is initialized at base address $1000$ in decimal memory. The physical start address of cell $6$ under 0-based indexing is:
> 
> $$ 1000 + 4 \cdot 6 = 1024 $$

### Handling Variable Data
If array structures require all elements to be the exact same size, how can they contain strings of varying lengths?
*   **Answer:** The array does not store the string characters directly. Instead, it stores a fixed-size pointer indicating the independent external memory address containing that unique string data.

---

# Dynamic Capacity Resizing

When initializing an Array List with an unknown number of total elements, the structure manages memory allocation via a dynamic growth loop:

1.  Allocates a default "large" capacity array in memory initially.
2.  Inserts elements into this backing array while tracking the count $n$.
3.  Once $n$ equals the array length capacity, it allocates a new backing array of double size ($2 \cdot \text{capacity}$).
4.  Copies all elements from the old array into the new array sequentially, updates the reference, and frees the old space.

---

# Algorithmic Operations

## `Insert(element, index)`
Inserts an item at a specific target index. If adding to the front, all existing elements must slide forward to open a space.

- **Time Complexity:** $O(1)$ Amortized Best Case (appending to the back); $O(n)$ Worst Case (inserting at index 0 or triggering an internal array resize).

```pseudo
	\begin{algorithm}
	\caption{Array List Insertion}
	\begin{algorithmic}
		\Procedure{Insert}{$element, index, array, n$}
			\If{$index < 0$ \or $index > n$}
				\Return $\text{false}$
			\EndIf
			\If{$n == array.\text{length}$}
				\State $newArray \gets \text{Allocate empty array of length } 2 \cdot array.\text{length}$
				\For{$i \gets 0 \text{ to } n - 1$}
					\State $newArray[i] \gets array[i]$
				\EndFor
				\State $array \gets newArray$
			\EndIf
			\If{$index == n$}
				\State $array[index] \gets element$
			\Else
				\For{$i \gets n - 1 \text{ down to } index$}
					\State $array[i + 1] \gets array[i]$
				\EndFor
				\State $array[index] \gets element$
			\EndIf
			\State $n \gets n + 1$
			\Return $\text{true}$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Find(element)`
Performs a linear scan from index 0 across the structure to match the target element.

- **Time Complexity:** $O(1)$ Best Case (first slot match); $O(n)$ Worst Case (item missing or sitting in the final slot).
- **Optimization:** If the underlying array is maintained in a sorted sequence, search speeds improve to $O(\log n)$ using Binary Search.

```pseudo
	\begin{algorithm}
	\caption{Linear Array Search}
	\begin{algorithmic}
		\Procedure{Find}{$element, array, n$}
			\For{$i \gets 0 \text{ to } n - 1$}
				\If{$array[i] == element$}
					\Return $\text{true}$
				\EndIf
			\EndFor
			\Return $\text{false}$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Remove(index)`
Removes an entry at a given index and shifts all trailing elements left by one index to avoid leaving structural gaps.

- **Time Complexity:** $O(1)$ Best Case (deleting the last item); $O(n)$ Worst Case (deleting from index 0).

```pseudo
	\begin{algorithm}
	\caption{Array List Removal}
	\begin{algorithmic}
		\Procedure{Remove}{$index, array, n$}
			\If{$index < 0$ \or $index \ge n$}
				\Return $\text{false}$
			\EndIf
			\If{$index < n - 1$}
				\For{$i \gets index \text{ to } n - 2$}
					\State $array[i] \gets array[i + 1]$
				\EndFor
			\EndIf
			\State $n \gets n - 1$
			\Return $\text{true}$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# Performance Summary

*   **Random Access:** $O(1)$ constant time lookup.
*   **Search Complexity:** $O(\log n)$ if sorted via Binary Search; $O(n)$ if unsorted.
*   **Insert/Delete Mechanics:** $O(n)$ general cost due to sequential cell shifting.
*   **Memory Footprint:** Continuous block safety; can trade off potential memory overhead if pre-allocated block capacity goes unused.
*   **Optimal Environment:** Fixed-size contexts or architectures dominated by index lookups.
*   **Weakest Environment:** Systems tracking heavy insertion or removal routines targeted at the front of the sequence.

---

# Related Notes

- [[Introductory Data Structures/Abstract Data Types (ADT)|Abstract Data Types (ADT)]]
- [[Introductory Data Structures/Circular Arrays|Circular Arrays]]
- [[Introductory Data Structures/Linked List|Linked List]]