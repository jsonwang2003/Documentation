
> [!IMPORTANT]
> 
> Binary Search is only applicable if the list is sorted. It uses the "Decrease and Conquer" strategy to eliminate half of the search space with every comparison.

---
## The Procedure
1. **Divide**: Identify the **mid-point** of the current list. Divide the search space into two conceptual halves:
    - $[0, \text{mid-point})$ — The left half.
    - $[\text{mid-point} + 1, n)$ — The right half.
2. **Compare**: Check the target element against the value at the **mid-point**:
    - **Smaller**: Search the left half.
    - **Greater**: Search the right half.
    - **Equal**: Return the current position (Target found).
3. **Recurse**: Continue splitting and searching until only one element remains.
4. **Terminate**: If the search space is exhausted without a match, report that the item is **not found**.

![[Pasted image 20251106174826.png]]

---
## Time Analysis
The efficiency of Binary Search comes from how quickly it shrinks the input. Each "split" reduces the remaining work by half.
### 1. Comparison Scaling
As shown in the table below, as the input size $n$ roughly doubles, the number of required comparisons only increases by 1.

|**n**|**# of splits**|**# of comparisons (k)**|
|---|---|---|
|1|0|1|
|3|1|2|
|7|2|3|
|15|3|4|
|31|4|5|

### 2. Deriving the Complexity
In general, if the list size is **$n = 2^{k}-1$**, you need **$k$** comparisons. To find the runtime in terms of $n$, we solve for $k$:

$$
\begin{align*}
n &\leq 2^k - 1 \\
n+1 &\leq 2^k \\
\log_{2}(n+1) &\leq k \\
k &= \boxed{\lceil\log_{2}(n+1)\rceil} 
\end{align*}$$

> [!CHECK]
> 
> This confirms that Binary Search grows at a Logarithmic rate, $O(\log n)$, making it incredibly efficient for large datasets.

![[Pasted image 20251106180059.png]]

---
## Formal Proof (Strong Induction)
**Claim:** Binary Search correctly finds the target in a sorted list of size $n$.
- **Base Case ($n=1$):** The mid-point is the only element. The algorithm checks it and correctly returns the index or "not found."
- **Inductive Hypothesis:** Assume Binary Search is correct for all sorted lists of size $m < n$.
- **Inductive Step:** For a list of size $n$, the algorithm compares the mid-point.
    - If equal, it returns correctly.
    - If not equal, it calls itself on a sub-list of size roughly $n/2$.
    - Since $n/2 < n$, the **Inductive Hypothesis** applies, and the sub-search is guaranteed to be correct.

---
## Related Notes
- [[Linear Search vs Binary Search]] — Visualizing why $O(\log n)$ is asymptotically faster.
- [[Merge Sort]] — The most common way to ensure a list is sorted before searching.
- [[Asymptotic Notation]] — More on the $O$ notation used here.