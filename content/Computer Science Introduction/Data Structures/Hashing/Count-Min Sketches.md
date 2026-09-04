---
description: "A space-efficient probabilistic frequency table structure mapping data streams onto a fixed-size 2D matrix to track item frequencies under bounded error limits."
aliases:
  - Count-Min Sketch
  - CM Sketch
  - Probabilistic Frequency Table
tags:
  - data-structures
  - hashing
  - streaming-algorithms
  - probabilistic
---
> [!abstract] Abstract 
> A Count-Min Sketch is a space-efficient, probabilistic data structure that functions like a frequency table. While a Hash Map stores every individual key-value pair, the Count-Min Sketch uses a fixed-size 2D array to provide an over-estimate of an element's frequency. It functions as a frequency-tracking extension of a Bloom Filter, trading exact precision for massive memory savings in high-volume, massive-scale data streams.
> 
> - **Category:** Probabilistic Frequency Structure
> - **Stores:** Bounded frequency approximations using cell counting arrays.
> - **Built on top of:** A 2D matrix layout paired with a bank of independent hash functions.
> - **Typical use cases:** Heavy-hitter stream identification, network packet frequency tracking, high-volume media view count estimation layers.

---

# Core Structure

The structure discards the source data keys entirely to save memory space. It maintains a 2D matrix holding numerical counters with $k$ independent horizontal rows and $m$ vertical columns. Each row is assigned its own independent hash function.

![[Pasted image 20260202102456.png]]

> [!tip] Key Idea 
> Because distinct keys can map to overlapping cell locations, hash collisions only ever increase or bloat the counters inside individual cells. Therefore, the **minimum value** across all $k$ hashed positions is guaranteed to be the cleanest, least-corrupted estimate. The true frequency will never exceed this returned minimum.

---

# Data Structure Operations

## `Increment(x)`
Passes the input through each row's hash function to resolve specific column coordinates, incrementing the counter at every targeted matrix cell by 1.

- **Time Complexity:** $O(k)$ where $k$ matches the fixed row depth count.

```pseudo
	\begin{algorithm}
	\caption{Count-Min Sketch Counter Increment}
	\begin{algorithmic}
		\Procedure{Increment}{$x, \text{matrix}, k, m$}
			\For{$i \gets 0 \text{ to } k - 1$}
				\State $column \gets$ \Call{HashFunc}{$i, x$} $\pmod m$
				\State $\text{matrix}[i][column] \gets \text{matrix}[i][column] + 1$
			\EndFor
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Estimate(x)`
Queries the $k$ hashed matrix cell coordinates for the key and screens out inflated error noise by isolating the absolute minimum value among them.

- **Time Complexity:** $O(k)$ operational calculations.
- **Notes:** While the isolated count can occasionally over-estimate due to collision footprints, it will never under-estimate the true frequency.

```pseudo
	\begin{algorithm}
	\caption{Count-Min Sketch Frequency Estimation}
	\begin{algorithmic}
		\Procedure{Estimate}{$x, \text{matrix}, k, m$}
			\State $min\_val \gets \infty$
			\For{$i \gets 0 \text{ to } k - 1$}
				\State $column \gets$ \Call{HashFunc}{$i, x$} $\pmod m$
				\State $current\_val \gets \text{matrix}[i][column]$
				\If{$current\_val < min\_val$}
					\State $min\_val \gets current\_val$
				\EndIf
			\EndFor
			\Return $min\_val$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# Mathematical Design

To limit estimation error margins, the layout dimensions of the matrix grid are derived from a chosen error tolerance threshold ($\epsilon$) alongside a targeted confidence level ($1 - \delta$):

*   **Matrix Width (Columns $m$):** Dictates the range bounds of the hashing calculations. More columns compress the numerical probability of a collision occurring inside any single row:
    $$
     m = \left\lceil \frac{e}{\epsilon} \right\rceil 
     $$
*   **Matrix Depth (Rows $k$):** Dictates the number of independent hash functions. More rows reduce the probability that every row will sustain a significant collision overlap for a specific item:
    $$
     k = \left\lceil \ln\left(\frac{1}{\delta}\right) \right\rceil 
     $$

---

# Common Pitfalls

*   **Assuming Absolute Counting Precision:** Using a sketch structure when exact counts are mandatory. The structure is inherently lossy and tailored to identifying broad trends or heavy-hitters rather than ledger accounting entries.
*   **Neglecting to Size Matrix Grids to Stream Volumes:** If the chosen column width count $m$ is too narrow for the total aggregate frequency volume of the stream, cells saturate uniformly, causing estimation errors to exceed the planned $\epsilon$ limit.

---

# Trade-offs Compared to Other Data Structures

| Evaluation Parameter | Hash Map Structure | Count-Min Sketch Structure |
|---|---|---|
| **Accuracy Standard** | 100% Precise Exact Results | Probabilistic Estimates (Over-estimates) |
| **Memory Footprint Scaling** | $O(n)$ — Expands with every unique key added | $O(m \times k)$ — Bounded flat matrix layout size |
| **Explicit Key Preservation** | Yes | No |
| **Optimal Use Cases** | Bounded local datasets | Heavy-hitter stream mining in massive streams |

---

# Related Notes

- [[Bloom Filters|Bloom Filters]]
- [[Open Addressing (Linear Probing)|Open Addressing (Linear Probing)]]
- [[Hash Maps (Maps)|Hash Maps (Maps)]]