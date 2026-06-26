> [!ABSTRACT]
> 
> Selection Sort is an in-place comparison sorting algorithm. It works by dividing the input list into two parts: a sorted sublist which is built up from left to right, and an unsorted sublist. In each step, it selects the smallest element from the unsorted sublist and swaps it into the next available position in the sorted sublist.

---
## The Strategy
For a list $[a_1, a_2, \dots, a_n]$:
1. **Scan**: Search the unsorted portion of the list for the smallest element $x$.
2. **Swap**: Exchange $x$ with the element at the beginning of the unsorted portion.
3. **Repeat**: Move the boundary of the sorted portion one element to the right and repeat until the entire list is processed.

![[Pasted image 20251027192326.png]]

---
## Efficiency Case Studies

### 1. Which list requires the fewest swaps?
- **Sorted List**: If the list is already in order, the algorithm identifies that the element already at the current index is the minimum.
- **Result**: $0$ swaps.

### 2. Which list requires the greatest swaps?
- **Cyclically Shifted**: If the list is shifted (e.g., $[2, 3, 4, 1]$), the algorithm must perform a swap at nearly every step.
- **Result**: $n-1$ swaps.

---

## Time Analysis
On a list of length $n$, Selection Sort is characterized by its predictable comparison count but highly efficient swap count.
### 1. Number of Swaps
- **Best Case**: $0$
- **Worst Case**: $n-1$
- **Big-O**: $O(n)$ — _This makes Selection Sort superior to [[Bubble Sort]] in scenarios where write operations (swaps) are expensive._

### 2. Number of Comparisons
Unlike Bubble Sort, Selection Sort **always** performs the same number of comparisons because it must scan the entire unsorted portion to guarantee it has found the true minimum.
- **Formula**: $(n-1) + (n-2) + \dots + 1 = \frac{n(n-1)}{2}$
- **Big-O**: $\boxed{O(n^2)}$ for both Best and Worst cases.

---
## Pros and Cons

|**Strengths**|**Weaknesses**|
|---|---|
|**Minimal Swaps**: Performs at most $n-1$ swaps.|**Unstable**: May swap equal elements out of their original relative order.|
|**Memory Efficient**: In-place algorithm ($O(1)$ extra space).|**Inefficient Comparisons**: Does not "notice" if a list is already sorted ($O(n^2)$ even in best case).|

---
## Related Notes
- [[Computer Science/Discrete Structures/Discrete Algorithms/Sorting/index|Sorting Index]] — Compare Selection Sort to other $O(n^2)$ algorithms.
- [[Sum of an Arithmetic Series]] — The mathematical proof for why the comparisons total $\frac{n(n-1)}{2}$.
- [[Bubble Sort]] — Another $O(n^2)$ algorithm that performs significantly more swaps.