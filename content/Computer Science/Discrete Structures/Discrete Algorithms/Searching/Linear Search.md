> [!ABSTRACT]
> 
> Linear Search is the most fundamental searching algorithm. It works by inspecting every element in a list sequentially until the target is found or the end of the list is reached. Because it makes no assumptions about the order of the data, it is the only option for unsorted lists.

---
## The Strategy
The algorithm follows a simple iterative logic:
1. **Iterate**: While there are items remaining in the list:
2. **Inspect**: Look at the current item.
3. **Match**: Is this the item looking for?
    - **Yes** → Return the position (Index).
    - **No** → Move to the next item.
4. **Terminate**: If you reach the end of the list without a match, report that the item is **not found**.

![[Pasted image 20251106171845.png]]

---
## Proof of Correctness
**Claim:** Linear Search returns the index of the target if it exists, otherwise it returns a "not found" indicator.
- **Base Case ($n=0$):** The list is empty. The loop condition is never met, and the algorithm correctly reports the item is not found.
- **Inductive Step:** Assume the algorithm correctly searches a list of size $k$. For a list of size $k+1$, the algorithm checks the first element. If it matches, it's correct. If not, it recursively (or iteratively) searches the remaining $k$ elements. By the hypothesis, the search on the remaining $k$ elements is correct.

---
## Time Analysis
On a list of length $n$, the performance is measured by the number of comparisons:
### 1. Best Case ($O(1)$)
- **Scenario**: The target element is the **first item** in the list.
- **Comparisons**: 1.

### 2. Worst Case ($O(n)$)
- **Scenario**: The target element is the **last item** in the list or is **not present** at all.
- **Comparisons**: $n$.

### 3. Average Case ($O(n)$)
- **Scenario**: The target is found somewhere in the middle.
- **Comparisons**: Approximately $n/2$.

![[Pasted image 20251106172228.png]]

---
## Pros and Cons

| **Strengths**                                     | **Weaknesses**                                                        |
| ------------------------------------------------- | --------------------------------------------------------------------- |
| Works on **unsorted** data.                       | Very slow for large datasets ($n=1,000,000$ takes $1,000,000$ steps). |
| Simple to implement and requires no extra memory. | Inefficient compared to [[Computer Science/Discrete Structures/Discrete Algorithms/Searching/Binary Search]] for sorted data.            |

---
## Related Notes
- [[Linear Search vs Binary Search]] – Comparing asymptotic growth.
- [[Time Analysis]] – General rules for $O(n)$ complexity.
