---
title: Searching Algorithms
---
> [!ABSTRACT]
> 
> Searching is the process of finding the location of a target value within a data structure. The efficiency of a search is largely dictated by whether the underlying data is sorted or unsorted.

---
## The Recursive Strategy: Decrease and Conquer

While Divide and Conquer splits a problem into multiple sub-problems, searching often uses **Decrease and Conquer**, where the problem is reduced to a single, smaller sub-problem.
- **Linear Search**: Reduces the problem size by 1 ($n-1$) each step.
- **Binary Search**: Reduces the problem size by half ($n/2$) each step.

---
## Knowledge Map
### [[Linear Search]]
- **The Big Picture**: Check the first element; if it's not the target, search the remaining $n-1$ elements.
- **Data Requirement**: None (works on unsorted data).
- **Analysis**: $T(n) = T(n-1) + c \implies \mathbf{O(n)}$.
### [[Binary Search]]
- **The Big Picture**: Compare the target with the middle element of a **sorted** list. Eliminate the half that cannot contain the target and recurse on the other half.
- **Verification**: Proven using **Strong Induction** because the search space is halved.
- **Analysis**: $T(n) = T(n/2) + c \implies \mathbf{O(\log n)}$.
### [[Lower Bounds for Searching]]
- **Concept**: A proof of why we cannot search an unsorted list faster than $O(n)$ or a sorted list faster than $O(\log n)$.
- **Logic**: Uses **Decision Trees** to show that the height of the tree (number of comparisons) represents the best-case complexity.

---
## Complexity Comparison

|**Algorithm**|**Data State**|**Recurrence**|**Complexity**|
|---|---|---|---|
|**Linear Search**|Unsorted|$T(n) = T(n-1) + O(1)$|$O(n)$|
|**Binary Search**|Sorted|$T(n) = T(n/2) + O(1)$|$O(\log n)$|

---
## Related Toolkits
- [[Time Analysis|Time Analysis]]: How we derive the $O(\log n)$ bound for Binary Search.
- [[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Divide and Conquer/Merge Sort|Merge Sort]]: Why we often sort data first (to enable $O(\log n)$ searching later).