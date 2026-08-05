---
title: "String Searching Data Structures"
description: "A central directory of high-performance substring matchers, full-text indexes, and automata optimized for large-scale string search patterns."
aliases:
  - String Searching Directory
  - String Matchers Index
  - String Searching Hub
  - Text Indexing Hub
tags:
  - index
  - string-searching
  - algorithms
  - bioinformatics
---
> [!abstract] Overview
> String searching structures resolve the complex task of locating substring segments or short query reads within massive database texts. These architectures form the core processing foundations for modern sequence aligners, text miners, and intrusion detection frameworks.

---

# Key Search Frameworks

### Preprocessed Query Models
Optimized for matching a large collection of short signature terms against a fluid sequence:

*   **[[Aho-Corasick Automaton\|Aho-Corasick Automaton]]:** Joins pattern structures inside an optimized finite state machine to discover multiple keywords simultaneously in a single linear pass.

### Preprocessed Database Models
Optimized for searching fluid query terms against a massive, fixed text database:

*   **[[Suffix Arrays\|Suffix Arrays]]:** Replaces costly character duplicate tracking with a compact sorted integer array mapping suffix offsets to enable logarithmic binary search lookups.
*   **[[Burrows-Wheeler Transformation\|Burrows-Wheeler Transformation]]:** Sorts cyclic shifts into a reversible text layout, using an FM-Index and Backward Search to achieve fast search speeds that scale independent of database size.

---

# Operational Performance Summary

| Architecture Scheme | Preprocessing Allocation Target | Search Time Complexity | Memory Management Footnote |
|---|---|---|---|
| **Aho-Corasick Automaton** | Multiple Short Patterns ($m$) | $O(n + \text{matches})$ | [[Multiway Trie Implementation\|Multiway Trie]] tracking with failure link nodes. |
| **Suffix Arrays** | Reference Database String ($n$) | $O(k \cdot \log n)$ | Flat sorted array storing compact integer offsets. |
| **Burrows-Wheeler Transformation** | Reference Database String ($n$) | $O(k)$ | Run-Length Encoded block matching via L2F steps. |

---

# Notes in This Section

| Note Link | Description |
|---|---|
| [[Aho-Corasick Automaton\|Aho-Corasick Automaton]] | Implements a linear-time finite state machine tracking overlapping keywords via failure links. |
| [[Suffix Arrays\|Suffix Arrays]] | Sorted index structure providing logarithmic binary search matching over static database blocks. |
| [[Burrows-Wheeler Transformation\|Burrows-Wheeler Transformation]] | Reversible permutation engine enabling fast query search lookups via the L2F property. |

---

# Related Categories

- [[Computer Science Introduction/Data Structures/Lexicon/index\|Lexicon ADT Implementations]]
- [[Computer Science Introduction/Data Structures/Hashing/index\|Hashing and Probabilistic Sketches]]
- [[Computer Science Introduction/Data Structures/Tree Structures/index\|Tree Structures]]