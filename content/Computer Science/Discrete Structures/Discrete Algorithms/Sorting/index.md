---
title: Sorting
---
> [!ABSTRACT]
> 
> Sorting is the process of arranging elements according to a total ordering (e.g., $a_1 \leq a_2 \leq \dots \leq a_n$). It is a fundamental building block in Computer Science that makes searching, finding duplicates, and set operations significantly more efficient.

---
## Why Sort?
- **Efficiency**: Sorted data enables faster operations (e.g., [[Computer Science/Discrete Structures/Discrete Algorithms/Searching/Binary Search]] or finding Min/Max).
- **Organization**: Helps in ranking and finding duplicates.
- **Algorithmic Variety**: Sorting is the perfect playground for learning:
    - **Iterative** (Bubble, Selection)
    - **Recursive / Divide & Conquer** ([[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Divide and Conquer/Merge Sort]])
    - **Randomized** (Quick Sort)

---
## The Valet Problem: Minimum Swaps
Imagine you and a friend are valets. You have 6 cars in a row and must order them by value (cheapest on the left, most expensive on the right). With no extra parking spots, your only operation is **swapping two cars**.

![[Pasted image 20251027181447.png]]

**The Goal**: Sort the lot in the **fewest number of swaps**.
**The Solution (Selection Sort Logic)**:
1. Identify the car that _should_ be at the current position (e.g., the minimum value).
2. Swap it with the car currently occupying that spot.
3. Repeat for the next position until the lot is sorted.

![[Pasted image 20251027181809.png]]

> [!TIP]
> 
> Swap Limits: For a list of size $n$, the Best Case is $0$ swaps (already sorted). The Worst Case is $n-1$ swaps (e.g., when the elements are cyclically shifted).

---
## Sorting Specification
Given a list $a_1, a_2, \dots, a_n$, we rearrange them such that:
1. **Sorted Property**: $a_1 \leq a_2 \leq \dots \leq a_n$.
2. **Permutation Property**: The output contains the exact same set of values as the input.

**Basic Operations**:
- **Comparison**: Checking if $a_i < a_j$.
- **Swap**: Exchanging the positions of two elements.

---
## Knowledge Map
### [[Bubble Sort]]
- **Logic**: Repeatedly steps through the list, compares **adjacent** elements, and swaps them if they are in the wrong order.
- **Vibe**: The largest elements "bubble up" to the end of the list.
### [[Selection Sort (Min Sort)]]
- **Logic**: Divides the list into a "sorted" and "unsorted" region. It repeatedly **selects** the minimum element from the unsorted region and moves it to the end of the sorted region.
- **Vibe**: Direct placement based on the Valet Problem strategy.
### [[Insertion Sort]]
- **Logic**: Builds the final sorted array one item at a time by **inserting** each new element into its correct relative position among the already-sorted elements.

---
## Related Toolkits
- [[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Divide and Conquer/Merge Sort|Merge Sort]] – Scaling sorting to $O(n \log n)$.
- [[Asymptotic Notation]] – Comparing why some sorts are better for large $n$.