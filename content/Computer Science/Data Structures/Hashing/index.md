> [!ABSTRACT]
> 
> While search structures like [[AVL Tree]] and [[Array Lists|Sorted Array]] offer $O(\log n)$ performance, **Hash Tables** aim for the "holy grail" of data structures: **$O(1)$ average-case time complexity**. This is achieved by transforming a key into an array index via a mathematical process called **Hashing**.

---
## 1. The Core Motivation: Beyond $O(\log n)$

To understand the power of a **Hash Table**, consider the efficiency of a standard array. If you already know that your data is stored at index `i`, accessing it with `array[i]` takes **constant time ($O(1)$)**.

The challenge is that we usually only have a **key** (like a name or ID). Hashing provides a way to map that key directly to a specific index.

---
## 2. What is Hashing?

1. **The Hash Function:** A mathematical function that takes a key $k$ and computes a numerical **hash value**.
2. **The Hash Table:** An array-based data structure that uses that value to determine exactly where $k$ should be stored.

---
## 3. Key Design Challenges

Creating a functional and fast [[Hash Tables|Hash Table]] requires solving three fundamental problems:

### A. [[Hash Functions|Designing a Good Hash Function]]

The function must be fast and distribute keys uniformly. If many keys map to the same index, the performance will collapse.

### B. [[Probability of Collisions#|Determining Table Size]]

The size of the backing array affects both memory usage and the **frequency of collisions**. A table that is too small becomes crowded, while one that is too large wastes memory.

### C. [[Computer Science/Data Structures/Hashing/Collision Resolution/index|Collision Resolution]]

Because array indices are *finite*, two different keys will eventually map to the same index. This is called a **collision**.
- [[Closed Addressing (Separate Chaining)]]: Storing multiple elements in a list at the same index.
- [[Open Addressing (Linear Probing)]]: Searching for the next available empty slot in the array.

---
## 4. Advanced Probabilistic Structures

In cases where we need to check if an element exists but have very little memory, we use structures based on similar hashing principles:
- [[Bloom Filters]]: Used for fast set-membership checks.
- [[Count-Min Sketches]]: Used for frequency estimation in data streams.

---
## 5. Performance Summary

When these design choices are handled correctly, the Hash Table provides nearly instantaneous access.

|**Operation**|**Worst-Case BST**|**Average-Case Hash Table**|
|---|---|---|
|**Find**|$O(\log n)$|**$O(1)$**|
|**Insert**|$O(\log n)$|**$O(1)$**|
|**Remove**|$O(\log n)$|**$O(1)$**|
