---
title: Divide and Conquer
---
> [!ABSTRACT]
> 
> Divide and Conquer is a recursive paradigm that breaks a problem into independent sub-problems, solves them recursively, and combines the results. It is the primary strategy for breaking the "quadratic barrier" ($O(n^2)$) in fundamental algorithms.

---
## The Three Pillars of D&C
1. **Divide**: Break the problem into smaller instances of the same problem.
2. **Conquer**: Solve sub-problems recursively. If they are small enough (base case), solve them directly.
3. **Combine**: Merge the sub-problem solutions into the final answer.

---
## Knowledge Map

### [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Divide and Conquer/Merge Sort]]
- **Concept**: An $O(n \log n)$ sorting algorithm that halves the input and merges sorted results.
- **Key Logic**: Uses a linear-time `RMerge` helper.
- **Formal Verification**: Requires **Strong Induction** for the main sort (due to $\frac{n}{2}$ shrinkage) and **Regular Induction** for the merge helper.

### [[Fast Multiplication]]
- **Concept**: Moving beyond the $O(n^2)$ "grade-school" multiplication method.
- **The Karatsuba Breakthrough**: Reduces the number of recursive multiplications from 4 to 3 using algebraic identities.
- **Complexity**: Achieves $O(n^{1.585})$ via the Master Theorem Case 3 ($a=3, b=2$).

---
## Complexity Overview

|**Algorithm**|**Recurrence**|**Big-O**|**Strategy**|
|---|---|---|---|
|**Binary Search**|$T(n) = T(n/2) + O(1)$|$O(\log n)$|Decrease and Conquer|
|**Merge Sort**|$T(n) = 2T(n/2) + O(n)$|$O(n \log n)$|Balanced splitting|
|**Naive Mult.**|$T(n) = 4T(n/2) + O(n)$|$O(n^2)$|Simple splitting|
|**Karatsuba**|$T(n) = 3T(n/2) + O(n)$|$O(n^{1.585})$|Optimized splitting|

---
## Related Toolkits
- [[Recursive Proofs|Recursive Proofs]]: Understanding why D&C requires Strong Induction.
- [[Master Theorem|Master Theorem]]: The standard tool for solving D&C recurrences.