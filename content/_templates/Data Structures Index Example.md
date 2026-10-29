---
title: "Data Structures" 
description: "Data structures for organizing information to support fast queries and updates — hashing strategies, disjoint sets, and their trade-offs." 
tags:
- CS/index
- CS/data-structures 
aliases: ["Data Structures"] 
---

> [!abstract] Overview 
> Data structures for organizing information to support fast queries and updates. Unlike the Algorithms section, notes here focus on _what a structure stores and guarantees_ rather than a single procedure's input/output behavior.

---

# Foundational Concepts

## Collision Resolution (Hashing)

When two keys hash to the same slot, a hash table needs a strategy to resolve the conflict — open addressing (probe for another slot) or chaining (store multiple keys per slot) are the two general families; [[Cuckoo Hashing]] is a more aggressive open-addressing variant covered in this section.

---

# Notes in This Section

|Note|One-line description|
|---|---|
|[[Disjoint Sets & Up-Trees]]|Union-Find ADT for tracking a partition of elements into disjoint subsets; near-constant amortized time with union-by-size/rank and path compression|
|[[Cuckoo Hashing]]|Collision resolution strategy where inserting keys evict existing ones; guarantees worst-case O(1) find/delete at the cost of occasional expensive rehashing|

---

# Related Categories

- [[Algorithms]] — several algorithms in that section (e.g. [[8. Kruskal's Algorithm]]) rely on structures documented here.