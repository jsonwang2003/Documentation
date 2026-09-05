> [!ABSTRACT]
> 
> Insertion Sort is an intuitive, comparison-based sorting algorithm that builds the final sorted list one element at a time. It is analogous to the way most people sort a hand of playing cards: you take one card at a time and "insert" it into its correct relative position among the cards already in your hand.

---
## The Strategy
For a list $[a_1, a_2, \dots, a_n]$:
1. **Divide**: Imagine the list is split into a "sorted" side (left) and an "unsorted" side (right). Initially, the first element is considered sorted.
2. **Pick**: Take the first element from the unsorted side (the "key").
3. **Shift**: Compare the key with elements in the sorted side from right to left. Shift elements that are larger than the key one position to the right.
4. **Insert**: Once you find the correct spot (or reach the beginning), drop the key into the empty slot.
5. **Repeat**: Continue until the unsorted side is empty.

```pseudo
	\begin{algorithm}
	\caption{Insertion Sort}
	\begin{algorithmic}
	\Procedure{InsertionSort}{$A[0 \dots n-1$: an array of $n$}
		\For{$i=1$ to $n-1$}
			\State $v = A[i]$
			\State $j = i-1$
			\While{$j \geq 0$ and $A[j] > v$}
				\State $A[j+1] = A[j]$
				\State $j = j-1$
            \EndWhile
            \State $A[j+1] = v$
        \EndFor
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

---
## Proof of Correctness (Loop Invariant)
- **Invariant**: At the start of each iteration $i$, the sub-list $A[1 \dots i-1]$ consists of the original elements but in sorted order.
- **Base Case**: When $i=1$, the sub-list has only one element, which is trivially sorted.
- **Maintenance**: If $A[1 \dots i-1]$ is sorted, the algorithm finds the correct position for $A[i]$ by shifting larger elements. After inserting $A[i]$, the sub-list $A[1 \dots i]$ is sorted.
- **Termination**: When $i$ reaches $n+1$, the entire list is sorted.

---
## Time Analysis
On a list of length $n$:
### 1. Best Case ($O(n)$)
- **Scenario**: The list is **already sorted**.
- **Logic**: The algorithm only performs one comparison for each element and zero shifts. It realizes immediately that each element is already in the correct place.

### 2. Worst Case ($O(n^2)$)
- **Scenario**: The list is in **reverse order**.
- **Logic**: For every new element, the algorithm must compare and shift against _every_ element already in the sorted portion.
- **Formula**: $1 + 2 + 3 + \dots + (n-1) = \frac{n(n-1)}{2}$.

### 3. Average Case ($O(n^2)$)
- **Logic**: On average, each element will be compared/shifted with half of the sorted sub-list.

---
## Pros and Cons

| **Strengths**                                                       | **Weaknesses**                                                  |
| ------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Adaptive**: Very fast for lists that are already "nearly sorted." | Inefficient ($O(n^2)$) for large, randomly ordered datasets.    |
| **Stable**: Does not change the relative order of equal elements.   | Performs many "shifts" (writes), though fewer than Bubble Sort. |
| **Online**: Can sort a list as it receives it (streaming data).     |                                                                 |

---
## Related Notes
- [[Bubble Sort]] – Another $O(n^2)$ sort, but usually less efficient than Insertion Sort.
- [[Selection Sort (Min Sort)]] – $O(n^2)$ but performs fewer swaps (at most $n-1$).
- [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Divide and Conquer/Merge Sort|Merge Sort]] – The $O(n \log n)$ alternative for larger datasets.