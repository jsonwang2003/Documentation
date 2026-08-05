---
description: "An array-based data structure leveraging hash functions to map keys onto discrete indices, achieving average constant-time operations independent of collection sizes."
aliases:
  - Hash Table
  - Unordered Set
tags:
  - data-structures
  - hashing
---
> [!abstract] Abstract 
> A Hash Table is an array-based data structure that leverages a hash function to map keys to specific array indices. Its defining characteristic is that its average-case performance is independent of the total number of elements ($n$) it stores, allowing for $O(1)$ operations in most practical scenarios.
> 
> - **Category:** Array-Based Mapping Structure
> - **Stores:** Hashable key sets distributed across randomized array boundaries.
> - **Built on top of:** Contiguous arrays and compression functions.
> - **Typical use cases:** High-speed lookup tables, duplicate elimination collections, unique set membership tracking systems.

---

# The Constant-Time Disclaimer

In computer science, we frequently label Hash Table operations as $O(1)$. However, it is important to understand what this measurement excludes:

*   **Ignoring the Hash Function:** The $O(1)$ designation refers strictly to the immediate array slot access *after* the hash value has already been calculated.
*   **The Cost of $k$ Elements:** For complex variable data types like strings or nested lists, a robust hash function must iterate over all $k$ items in that collection to avoid catastrophic collision rates. Hashing a string of length $k$ is technically an $O(k)$ operation.
*   **Why We Say $O(1)$ Anyway:** We use $O(1)$ because the time complexity does not scale as you add more elements to the table. Unlike a Binary Search Tree (where search times degrade as the tree grows taller), a Hash Table's structural jump to an index remains constant regardless of the total entry volume.

---

# Core Structure

A Hash Table consists of a physical backing array of size $M$ (representing table capacity) combined with a chosen hash function $H(k)$ that compresses keys down to fit inside valid bounds ($0 \le \text{index} < M$).

```
       Key Input Context
          [ "Giraffe" ] 
                |
                v
       ( Hash Function H(k) )
                |
                v
       [ Compressed Index ] ---> Backing Array: arr[index] = key
```

> [!tip] Key Idea
> Because hash functions utilize mathematical operations (such as $H(k) = 2^k \pmod M$) to randomize where keys land to minimize conflicts, the physical layout sequence of items in memory bears no relationship to their actual input values.

---

# Structural Properties

*   **Invariant:** All element locations are deterministically linked to their hash outcomes. An item can only reside in a slot determined by its probe path or collision chain sequence.
*   **The Unordered Property:** There is no efficient way to iterate or print a Hash Table in chronological, alphabetical, or numerical order.
*   **The Collision Challenge:** Because the universe of possible key values is vastly larger than any practical table capacity $M$, different keys ($k_1 \neq k_2$) will inevitably yield identical index outputs ($H(k_1) = H(k_2)$). Managing these collisions is the primary bottleneck of hash performance.

---

# Data Structure Operations

## `Insert(key)`
Hashes the entry key to resolve its designated index space and positions the element if the cell is open.

- **Time Complexity:** $O(1)$ average; $O(n)$ worst-case under severe clumping.

```pseudo
	\begin{algorithm}
	\caption{Hash Table Insertion (Collision-Free Primitive Paradigm)}
	\begin{algorithmic}
		\Procedure{Insert}{$key, arr$}
			\State $index \gets$ \Call{H}{$key$}
			\If{$arr[index] == \text{NULL}$}
				\State $arr[index] \gets key$
			\EndIf
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `Find(key)`
Re-computes the coordinate mapping to instantly check if the storage cell matches the target key value.

- **Time Complexity:** $O(1)$ average case.

```pseudo
	\begin{algorithm}
	\caption{Hash Table Find (Collision-Free Primitive Paradigm)}
	\begin{algorithmic}
		\Procedure{Find}{$key, arr$}
			\State $index \gets$ \Call{H}{$key$}
			\Return $arr[index] == key$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---

# C++ Language Implementation Detail

In C++, the standard implementation of a Hash Table set is `std::unordered_set`. The standard library deliberately prepends `unordered_` to explicitly remind developers that element sequences are non-deterministic and can shift during dynamic table adjustments.

```cpp
#include <iostream>
#include <unordered_set>
#include <string>

int main() {
    std::unordered_set<std::string> animals = {"Giraffe", "Polar Bear", "Toucan"};
    
    // Output sequence is non-deterministic based on hash bucket assignments
    for (const auto& animal : animals) {
        std::cout << animal << std::endl;
    }
}
```

---

# Performance & Operations Summary

| Operation | Average Complexity | Functional Search Mechanics |
|---|---|---|
| **Find** | $O(1)$ | Computes target index $\to$ executes direct array pointer jump. |
| **Insert** | $O(1)$ | Computes destination index $\to$ places key into array structure. |
| **Remove** | $O(1)$ | Computes target index $\to$ clears array slot or appends tombstone. |

---

# Related Notes

- [[Hash Functions|Hash Functions]]
- [[Hash Maps (Maps)|Hash Maps (Maps)]]
- [[Computer Science Introduction/Data Structures/Hashing/Collision Resolution/index|Collision Resolution Strategies]]
- [[Probability of Collisions|Probability of Collisions]]