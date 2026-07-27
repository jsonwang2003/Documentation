---
description: "A space-efficient probabilistic data structure used to test set membership with zero false negatives and a controlled margin of false positives."
aliases:
  - Bloom Filter
  - Probabilistic Set
tags:
  - data-structures
  - hashing
  - probabilistic
---
> [!abstract] Abstract 
> A Bloom Filter is a space-efficient probabilistic data structure used to test whether an element is a member of a set. Unlike a standard Hash Table, it can return False Positives but never False Negatives. It is the ideal solution when memory is limited and a small margin of error is acceptable.
> 
> - **Category:** Probabilistic Bit Structure
> - **Stores:** Binary membership markers across overlapping bit indices.
> - **Built on top of:** A flat bit array and a bank of independent hash functions.
> - **Typical use cases:** Browser malicious URL tracking filters, database LSM-tree disk-read filters (e.g., Cassandra, RocksDB), cache filtering layers.

---

# Core Structure

The filter does not retain the actual cleartext elements or structural keys within memory. Instead, it maintains a compact, flat array of bits initialized to zero. Multiple independent hash functions map elements to specific bit positions.

```
Bit Array Structure (Size m)
[ 0 ] [ 1 ] [ 0 ] [ 1 ] [ 1 ] [ 0 ] [ 0 ] [ 1 ]
  ^     ^           ^     ^                 ^
  |     |___________|_____|_________________|
  |             Hash Function Hits (k functions)
[ Input Element x ]
```

> [!tip] Key Idea
> By abandoning key storage entirely and representing additions strictly as scattered bits, memory requirements drop from megabytes down to kilobytes. If any bit in an element's probe sequence is 0, it is mathematically impossible for that element to have been inserted, ensuring **zero false negatives**.

---

# Data Structure Operations

## `Insert(x)`
Feeds the element through all $k$ hash functions sequentially and sets every resolved bit coordinate to true.

- **Time Complexity:** $O(k)$ where $k$ matches the fixed count of hash functions.

```pseudo
	\begin{algorithm}
	\caption{Bloom Filter Insertion}
	\begin{algorithmic}
		\Procedure{Insert}{$x, \text{bit\_array}, m, k$}
			\For{$i \gets 1 \text{ to } k$}
				\State $index \gets$ \Call{Hash}{$x, i$} $\pmod m$
				\State $\text{bit\_array}[index] \gets \text{true}$
			\EndFor
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Find(x)`
Re-evaluates the $k$ hash coordinates for the target value. If any bit position along the generated trail holds a value of false, the element is definitely not present.

- **Time Complexity:** $O(k)$ operational steps.
- **Notes:** If all bits return true, the element is marked as *possibly present*. Trailing bit overlap caused by other keys can induce false positive errors.

```pseudo
	\begin{algorithm}
	\caption{Bloom Filter Membership Query}
	\begin{algorithmic}
		\Procedure{Find}{$x, \text{bit\_array}, m, k$}
			\For{$i \gets 1 \text{ to } k$}
				\State $index \gets$ \Call{Hash}{$x, i$} $\pmod m$
				\If{$\text{bit\_array}[index] == \text{false}$}
					\Return $\text{false}$
				\EndIf
			\EndFor
			\Return $\text{true}$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# Mathematical Optimization

The probability of a false positive ($\epsilon$) is directly tied to the size of the bit array ($m$), the number of elements expected ($n$), and the total hash functions deployed ($k$):

$$\epsilon \approx \left(1 - e^{-kn/m}\right)^k$$

To minimize error rates when designing a practical filter envelope, configuration sizing uses these optimal equations:

*   **Optimal Array Sizing:**
    $$m = -\frac{n \ln(\epsilon)}{(\ln(2))^2}$$
*   **Optimal Hash Count:**
    $$k = \frac{m}{n} \ln(2)$$

---

# Common Pitfalls

*   **Attempting Dynamic Deletions:** You cannot un-set a bit to 0 during a remove operation because multiple independent elements share overlapping bit index slots. Erasing one element's footprints will corrupt membership states for unrelated keys.
*   **Under-sizing the Bit Spectrum:** Cramming more elements into the filter than the array capacity $m$ accommodates saturates the bits to 1, causing the false positive rate to rapidly decay toward 100%.

---

# Trade-offs Compared to Other Data Structures

| Feature Metric | Hash Tables | Bloom Filter |
|---|---|---|
| **Memory Allocation** | High ($O(n \times \text{key\_size})$) | Very Low ($O(m)$ flat bit bounds) |
| **Search Performance** | $O(1)$ Average | $O(k)$ Constant function overhead |
| **False Positive Risks** | No | Yes |
| **False Negative Risks** | No | No |
| **Item Erasure / Deletion** | Simple array or list adjustments | Impossible without rebuilding |

---

# Related Notes

- [[Hashing/Collision Resolution/index|Collision Resolution Strategies]]
- [[Hashing/Collision Resolution/Open Addressing (Linear Probing)|Open Addressing (Linear Probing)]]
- [[Hashing/Count-Min Sketches|Count-Min Sketches]]