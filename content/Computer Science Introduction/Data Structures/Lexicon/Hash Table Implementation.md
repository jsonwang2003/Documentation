---
description: "A fast, unordered lexicon implementation utilizing string hash computation to deliver near-constant time lookups."
aliases:
  - Hash Lexicon
  - Unordered Lexicon
tags:
  - lexicon
  - data-structures
  - hashing
---
> [!abstract] Abstract 
> A [[Hash Tables|Hash Table]] implementation offers the fastest average-case performance for exact word queries within the [[Computer Science Introduction/Data Structures/Lexicon/index|Lexicon ADT]]. By transforming a string word into a distinct numerical index via a type-specific [[Hash Functions|Hash Function]], the system can achieve $O(1)$ average-case lookup, insertion, and removal operations.
> 
> - **Category:** Unordered Hashed Lexicon
> - **Average Lookup Bound:** $O(k)$ where $k$ matches word string length.
> - **Core Trade-off:** Abandons alphabetical ordering properties to optimize search speeds.

---

# Mechanics: From Words to Array Indices

To store a word inside a Hash Table lexicon, the architecture processes the element through a two-step mapping pipeline:

1.  **String Hashing:** A non-commutative polynomial function evaluates the characters of the string (word) of length $k$ to generate an integer hash value. This step takes $O(k)$ time.
2.  **Compression Indexing:** The system uses a modulo function to compress that wide integer down to fit within the physical array size: 

$$
\text{index} = \text{hashValue} \pmod{\text{array\_size}}
$$

---

# Performance Analysis

While we routinely define general hash table actions as constant time ($O(1)$), we must explicitly account for the processing time spent hashing the variable-length string itself ($k$) when evaluating text lexicons:

| Lexicon Operation | Complexity (Average Case) | Algorithmic Logic |
|---|---|---|
| **`find(word)`** | $O(k)$ | Time to hash a string of length $k$ + direct $O(1)$ array pointer jump. |
| **`insert(word)`** | $O(k)$ | Time to hash string + direct $O(1)$ cell entry placement. |
| **`remove(word)`** | $O(k)$ | Time to hash string + direct $O(1)$ slot erasure or tombstone stamp. |
| **Space Overhead** | $O(n)$ | Requires extra padding capacity to preserve a low load factor ($\alpha \le 0.70$). |

---

# Transitioning from Lexicon to Full Dictionary

By upgrading our backing architecture from a standard Hash Table (which tracks unique keys) to an associative [[Hash Maps (Maps)|Hash Map]], we transition from a simple word list verification engine into a fully functional Dictionary:

*   **The Key:** The text word (processed by the string hashing function).
*   **The Value:** The definition string, etymology records, or metadata objects.
*   **The Result:** We gain comprehensive dictionary utility with no significant loss in operational lookup speed.

---

# Architectural Trade-offs Evaluation

The Hash Table is a powerful contender for modern digital lexicons, but it introduces specific structural trade-offs:

*   **Speed Superiority:** On average, it runs faster than [[Array Lists|Binary Search]]. A word with 7 letters takes roughly 7 mathematical operations to hash, regardless of whether the tracking dictionary contains 100 or 1,000,000 words. It decouples lookup speed from total word volume $n$.
*   **Ordering Failure:** Unlike [[Array Implementation#Structural Comparison Linked List vs. Sorted Array|Sorted Arrays]] or [[Binary Search Tree (BSTs)|Binary Search Tree]], **Hash Tables** are completely unordered. You cannot easily print the lexicon in alphabetical order or query the immediate "next" word in alphabetical sequence.
*   **Memory Waste Requirements:** To prevent structural collisions and preserve $O(1)$ speeds, the table must maintain empty safety buffers, leaving roughly 30% of the allocated capacity empty.
*   **Worst-Case Vulnerability:** In the worst-case scenario where many words collide into the same slot, performance can degrade to an $O(n)$ linear scan.

---

# Structural Comparison: Sorted Array vs. Hash Table

| Evaluation Parameter | Sorted Array Implementation | Hash Table Implementation (Average) |
|---|---|---|
| **Search Speed** | $O(\log n)$ binary search | $O(k)$ string character hash traversal |
| **Alphabetical Ordering** | Supported natively | Unsupported (Requires external sorting) |
| **Space Efficiency** | High (100% compact layout) | Lower (Requires $\approx 30\%$ empty padding space) |
| **Worst-Case Search** | Guaranteed $O(\log n)$ | $O(n)$ (Occurs under total collision collapse) |

---

# Related Notes

- [[Hash Tables|Hash Tables]]
- [[Hash Maps (Maps)|Hash Maps (Maps)]]
- [[Multiway Trie Implementation|Multiway Trie Implementation]]