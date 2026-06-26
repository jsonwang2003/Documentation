> [!ABSTRACT]
> 
> Merge Sort is a comparison-based sorting algorithm that uses the Divide and Conquer strategy to achieve a guaranteed $O(n \log n)$ time complexity. It relies on the fact that merging two sorted lists is significantly more efficient ($O(n)$) than sorting an unsorted one from scratch.

---
## The Strategy
1. **Divide**: Split the unsorted list into two sub-lists of roughly $\frac{n}{2}$ size.
2. **Recursively Sort**: Call `MergeSort` on both sub-lists until base cases are reached.
3. **Conquer (Merge)**: Combine the two sorted sub-lists into one sorted result using the `RMerge` helper.

![[Pasted image 20251112202703.png]]

---
## Formal Proof of Correctness

### Part 1: The Merge Helper (`RMerge`)
We prove the helper function using **Regular Induction** because the total number of elements ($k+l$) decreases by exactly 1 in each recursive call.

![[Pasted image 20251112194733.png]]

- **Base Case**: If both lists are empty ($n=0$), it returns an empty list, which is sorted.
- **Inductive Hypothesis**: Assume `RMerge` correctly merges any two sorted lists with combined size $n-1$.
- **Inductive Step**: For a combined size $n$, we compare the heads ($a_1, b_1$). The smaller element is prepended to the result of `RMerge` called on the remaining $n-1$ elements. By the hypothesis, the sub-call is correct; thus, the final list is sorted.

### Part 2: The Main Algorithm
We prove `MergeSort` using **Strong Induction** because each subsequent call **halves the input size** ($\frac{n}{2} < n-1$).

![[Pasted image 20251207173654.png]]

- **Base Case**:
    - $n=0$: Returns an empty list (trivially true).
    - $n=1$: Returns $a_1$, a trivially sorted list containing all elements.
- **Inductive Hypothesis**: Assume `MergeSort` correctly sorts all lists with $k$ elements for any $0 \leq k < n$, where $n > 1$.
- **Inductive Step**:
    1. Divide the list of size $n$ into two halves of size $m = \lfloor n/2 \rfloor$ and $n-m$.
    2. By the **Strong Inductive Hypothesis**, since both halves have size $< n$, the recursive calls $L_1 = MergeSort(\text{Left})$ and $L_2 = MergeSort(\text{Right})$ return correctly sorted lists.
    3. By the correctness of `RMerge`, $RMerge(L_1, L_2)$ results in a sorted list of all $n$ elements.

---
## Time Analysis
### 1. Recurrence Extraction

Let $T(n)$ be the runtime of `MergeSort` on a list of size $n$.
$$
\begin{align*}
T(0) &= c_0\\
T(1) &= c_1\\
T(n) &= 2T(n/2) + T_{merge}(n)
\end{align*}
$$

Since $T_{merge}(n) = O(n)$, the expression is:

$$
T(n) = 2T(n/2) + O(n)
$$

### 2. Method A: Master Theorem
Comparing to $T(n) = aT(n/b) + O(n^d)$:
- $a = 2$ (number of recursive calls)
- $b = 2$ (factor by which size is reduced)
- $d = 1$ (exponent of non-recursive work $O(n^1)$)

Comparison: $a = 2$ and $b^d = 2^1 = \boxed{2}$.

Since $a = b^d$, we use Case 2:

$$
\boxed{T(n) = O(n \log n)}
$$

### 3. Method B: Unraveling (Iteration)
We substitute the recurrence into itself to find the pattern:

$$
\begin{align*} T(n) &= 2T(n/2) + cn \\ 
&= 2[2T(n/2^2) + c(n/2)] + cn = 2^2T(n/2^2) + 2cn \\ 
&= 2^2[2T(n/2^3) + c(n/2^2)] + 2cn = 2^3T(n/2^3) + 3cn \\ 
&\dots \\ 
&= 2^k T(n/2^k) + kcn 
\end{align*}
$$

To reach the base case $T(1)$, we set $n/2^k = 1 \implies \mathbf{k = \log_2 n}$:

$$
\begin{align*}
T(n) &= nT(1) + (\log_2 n)cn \\ 
&= n c_1 + cn \log_2 n \\ 
&= \boxed{O(n \log n)} 
\end{align*}
$$

---
## Related Notes

- [[Fast Multiplication|Fast Multiplication]] – Another application of D&C.
- [[Master Theorem|Master Theorem]] – Deep dive into the cases used here.
- [[Recursive Proofs|Recursive Proofs]] – General framework for Strong vs. Regular Induction.