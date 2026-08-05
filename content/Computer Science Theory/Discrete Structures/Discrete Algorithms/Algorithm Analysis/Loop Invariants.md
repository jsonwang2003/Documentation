> [!ABSTRACT]
> 
> A loop invariant is a property that remains true before and after each execution of a loop's body. It is the primary tool used to prove the correctness of iterative algorithms through mathematical induction.

---
## 1. The Three-Step Proof Process
To prove an algorithm is correct using an invariant, you must follow this logical flow:
1. **State the Invariant**: Define the property precisely in terms of the loop variables and iterations ($t$).
2. **Prove the Invariant ([[Induction]])**:
    - **Base Case**: Show it is true before the loop starts ($t=0$).
    - **Inductive Step**: Show that if it's true after $t$ iterations, it remains true after the $(t+1)^{th}$ iteration.
3. **Prove Correctness**: Show that the invariant, combined with the **loop termination condition**, logically leads to the desired solution.

---
## 2. Selection Sort (MinSort) Example
Selection sort works by repeatedly finding the minimum element from the unsorted portion and moving it to the front.

![[Pasted image 20251108144130.png]]
### The Invariants (after $t$ iterations)
1. **Sorted-ness**: The first $t$ elements are in non-decreasing order.
2. **Minimality**: The first $t$ elements are the $t$ smallest elements of the entire array.
### Proof Sketch
- **Base Case ($t=0$)**: The first $0$ elements are **vacuously true**; no elements exist to be out of order or "not the smallest."
- **Inductive Step**: Assume the first $t$ are sorted and smallest. In iteration $t+1$, the algorithm finds the minimum of the remaining $n-t$ elements and places it at index $t+1$. Since this new element is $\geq$ the previous $t$ elements (by minimality) but $\leq$ all remaining elements, the first $t+1$ elements are now sorted and smallest.
- **Termination**: After $n-1$ iterations, the first $n-1$ elements are sorted and smallest. This forces the $n^{th}$ element to be the largest, meaning the entire array is sorted.

---
## 3. Binary Search Decision Example
Binary search is a **Decision Algorithm** (returns TRUE/FALSE). To prove it correct, we use a **Bi-Directional Proof**.

![[Pasted image 20251108195815.png]]
### Claim 1: If the algorithm returns TRUE, then $x$ is in the list.
- **Proof**: The algorithm only returns TRUE if it encounters a line where `target == list[mid]`. By definition, $x$ is in the list.
### Claim 2: If $x$ is in the list, then the algorithm returns TRUE.
This requires the **Loop Invariant**: $a_i \leq x \leq a_j$ (where $i$ and $j$ are the current search boundaries).
- **Base Case**: Initially $i=1, j=n$. Since the list is sorted and $x$ is in the list, it must be that $a_1 \leq x \leq a_n$.
- **Inductive Step**:
    - If $x > a_m$, we move $i$ to $m+1$. Since $x$ was $\leq a_j$ and now $x \geq a_{m+1}$, the invariant holds.
    - If $x < a_m$, we move $j$ to $m-1$. Since $x$ was $\geq a_i$ and now $x \leq a_{m-1}$, the invariant holds.
- **Termination**: When the loop ends ($i=j$), the invariant $a_i \leq x \leq a_j$ implies $a_i \leq x \leq a_i$. Therefore $x = a_i$, and the algorithm returns TRUE.

---
## 4. Key Definitions

|**Term**|**Definition**|
|---|---|
|**Vacuously True**|A statement that is true because its antecedent cannot be satisfied (e.g., "All elements in an empty set are purple").|
|**Decision Problem**|A problem with a simple YES/NO (True/False) answer.|
|**Iteration Variable ($t$)**|A counter representing how many times the loop body has fully executed.|
