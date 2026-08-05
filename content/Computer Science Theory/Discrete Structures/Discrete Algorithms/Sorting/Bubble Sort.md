> [!ABSTRACT]
> 
> Bubble Sort is a simple, comparison-based sorting algorithm. It works by repeatedly stepping through the list, comparing adjacent elements, and swapping them if they are in the wrong order. The algorithm gets its name because the largest elements "bubble up" to their correct positions at the end of the list.

---
## The Strategy
For a list $[a_1, a_2, \dots, a_n]$:
1. **Compare & Swap**: Compare $a_i$ and $a_{i+1}$. If $a_i > a_{i+1}$, swap them.
2. **Pass Completion**: Continue this all the way to the end of the current unsorted range.
    - _Result:_ The largest element in that range is now at the final position.
3. **Reduce Range**: Repeat the process, reducing the "end" of the list by one each time until no more swaps are needed.

![[Pasted image 20251106162704.png]]

---
## Optimization: Early Exit
Standard Bubble Sort is "blind"—it continues iterating even if the list becomes sorted mid-way. By adding a **boolean flag** (e.g., `swapped`), we can terminate the algorithm early.
- **Logic**: If a full pass is completed without a single swap occurring, the list is guaranteed to be sorted.
- **Impact**: This improves the **Best Case** time complexity significantly.

![[Pasted image 20251106160837.png]]

---
## Proof of Correctness (Sketch)
- **Invariant**: After $k$ passes, the $k$ largest elements are correctly sorted at the end of the list.
- **Base Case**: After 1 pass, the largest element has "bubbled" to position $n$.
- **Inductive Step**: If the $k$ largest elements are in place, the $(k+1)^{th}$ pass will find the largest element in the remaining $n-k$ unsorted elements and move it to position $n-k$.

---
## Time Analysis
On a list of length $n$:
### 1. Number of Swaps
- **Best Case**: $0$ (The list is already sorted).
- **Worst Case**: $\frac{n(n-1)}{2}$ (The list is in reverse order).

### 2. Number of Comparisons
- **Best Case (with Early Exit)**: $n-1$ (Only one pass is needed to verify it is sorted).
- **Worst Case**: $\frac{n(n-1)}{2}$ (Requires all passes).

> [!NOTE]
> 
> In Big-O notation, Bubble Sort is $\mathbf{O(n^2)}$ in the worst and average cases. Because it only swaps adjacent elements, it is generally less efficient than [[Selection Sort (Min Sort)]] or [[Insertion Sort]].

---
## Pros and Cons

|**Strengths**|**Weaknesses**|
|---|---|
|**Stable**: Maintains relative order of equal elements.|Very slow ($O(n^2)$) for large datasets.|
|**In-place**: Requires $O(1)$ extra memory.|Performs many more swaps than Selection Sort.|
|**Simple**: Extremely easy to implement.||

---
## Related Notes
- [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Sorting/index|Sorting Index]] — Compare Selection Sort to other $O(n^2)$ algorithms.
- [[Selection Sort (Min Sort)]] — A similar $O(n^2)$ sort that minimizes swaps.
- [[Asymptotic Notation]] — Understanding why $n^2$ scales poorly.
