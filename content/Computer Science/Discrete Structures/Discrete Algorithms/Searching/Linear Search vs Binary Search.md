> [!ABSTRACT]
> 
> The choice between Linear and Binary search is a trade-off between data preparation and execution speed. While Linear Search is flexible, Binary Search is asymptotically faster, meaning its efficiency advantage grows exponentially as the dataset scales.

---
## Comparative Analysis

### 1. Linear Search
- **Mechanism**: Compares the target value with every element in the list, one by one.
- **Requirement**: Works on any data (sorted or unsorted).
- **Efficiency**: $O(n)$. The time taken grows in a 1:1 ratio with the input size.

### 2. Binary Search
- **Mechanism**: Leverages the **sorting property** to eliminate half of the remaining list with every single comparison.
- **Requirement**: Data **must** be sorted.
- **Efficiency**: $O(\log n)$. The time taken grows extremely slowly as the input size increases.

---
## Visualizing the Gap

![[Pasted image 20251217190052.png]]

As shown in the graph, Binary Search is not just faster; its runtime becomes a **smaller and smaller fraction** of the Linear Search runtime as $n$ grows.

> [!INFO]
> 
> Mathematically, we say [[Binary Search]] is **asymptotically faster** than [[Linear Search]]. Even if *Linear Search* was running on a supercomputer and *Binary Search* on a calculator, for a large enough $n$, **Binary Search will always win**.

---
## Summary Table

|**Feature**|**Linear Search**|**Binary Search**|
|---|---|---|
|**Recurrence**|$T(n) = T(n-1) + c$|$T(n) = T(n/2) + c$|
|**Complexity**|$O(n)$|$O(\log n)$|
|**Best Case**|$O(1)$ (First element)|$O(1)$ (Middle element)|
|**Data Order**|Unsorted|**Must be Sorted**|

---
## Related Notes
- [[Asymptotic Notation]] – Understanding the math behind the "Gap."
- [[Merge Sort]] – Often used as a prerequisite to enable Binary Search.
- [[Lower Bounds for Searching]] – Why we can't do better than $O(\log n)$ for comparisons.