---
description: "A compact Lexicon architecture leveraging sorted contiguous slots to provide logarithmic binary search lookups."
aliases:
  - Array Lexicon
  - Sorted Array Lexicon
tags:
  - lexicon
  - data-structures
  - arrays
---
> [!abstract] Abstract 
> An Array implementation of a Lexicon relies on hardware Random Access to enable high-speed Binary Search algorithms. While it incurs a high cost for word list modifications ($O(n)$ to shift elements), it is a superior choice for a Lexicon where word lookups dominate and the underlying dictionary remains relatively static.
> 
> - **Category:** Contiguous Sorted Lexicon
> - **Key Advantage:** Instant random access to any relative index coordinate.
> - **Optimal Environment:** Static, read-heavy word list validation.

---

# Why Sorting and Random Access Matter

In a Lexicon context, an unsorted array is no more efficient than a [[Computer Science Introduction/Data Structures/Introductory Data Structures/Linked List|linear linked list]], requiring an $O(n)$ linear scan. However, keeping the contiguous array in a sorted alphabetical sequence fundamentally shifts the operational math:

*   **Random Access:** Because array slots sit perfectly contiguous in hardware memory blocks, the platform calculates the exact address of any index position in $O(1)$ constant time.
*   **Binary Search:** Utilizing random access, the search engine samples the middle element, discards the unmatching half of the list, and repeats the split. This compresses the search space from $n$ elements down to a single element in just $\log_2 n$ steps.

---

# Performance Analysis

Because we maintain the backing array in a tightly sorted, compact arrangement with no internal gaps, our operational complexity reflects the cost of maintaining that order:

| Lexicon Operation | Complexity | Algorithmic Logic |
|---|---|---|
| **`find(word)`** | $O(\log n)$ | Enabled by rapid Binary Search splits. |
| **`insert(word)`** | $O(n)$ | Requires shifting trailing elements right to open an alphabetical slot. |
| **`remove(word)`** | $O(n)$ | Requires shifting trailing elements left to close the gap of the deleted word. |
| **Space Complexity** | $O(n)$ | Tracks $n$ slots for words, plus transient buffers for dynamic resizing. |

---

# Evaluation for the Lexicon ADT

The Sorted Array implementation aligns cleanly with our core [[Computer Science Introduction/Data Structures/Lexicon/index|Lexicon ADT]] design rules:

*   **Fast Lookups:** An $O(\log n)$ boundary is a massive improvement over linear lists. For a standard lexicon containing 170,000 words, Binary Search resolves a lookup in roughly 18 comparisons, whereas a linked list could exhaust all 170,000 pointer records.
*   **Infrequent Updates:** While $O(n)$ element shifting is computationally slow, it is acceptable here because we rarely introduce or remove terms from a language dictionary in everyday use.
*   **Memory Efficiency:** Arrays offer excellent space efficiency, though dynamic vectors may temporarily double their memory footprint ($2n$) during reallocation steps to accommodate growth.

---

# Structural Comparison: Linked List vs. Sorted Array

| Technical Parameter | Linked List Implementation | Sorted Array Implementation |
|---|---|---|
| **Search Speed** | $O(n)$ linear traversal | $O(\log n)$ binary search |
| **Random Access** | Impossible | Native ($O(1)$ address math) |
| **Insertion Mechanism** | Pointer redirection ($O(n)$ sorted) | Contiguous cell shifting ($O(n)$) |
| **Memory Overhead** | High (Node pointers tracking elements) | Low (Contiguous data block allocation) |

---

# Related Notes

- [[Array Lists|Array Lists]]
- [[Binary Search Tree Implementation|Binary Search Tree Implementation]]
- [[Linked List Implementation|Linked List Implementation]]