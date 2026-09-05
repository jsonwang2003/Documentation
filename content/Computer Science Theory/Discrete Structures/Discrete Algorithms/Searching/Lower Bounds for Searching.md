> [!ABSTRACT]
> 
> A lower bound proof establishes the absolute minimum number of operations (usually comparisons) required to solve a problem, regardless of how clever the algorithm is. For searching in a sorted list, the lower bound is $\Omega(\log n)$, proving that Binary Search is asymptotically optimal.

---
## The Decision Tree Model
To prove a lower bound for any comparison-based search, we use a **Decision Tree**. This is a conceptual model where:
- **Nodes** represent a comparison between the target $x$ and an element $a_i$.
- **Edges** represent the outcome ($<$ or $>$).
- **Leaves** represent the final result (the index of the element or "Not Found").

### Properties of the Tree
For a list of size $n$, there are $n+1$ possible outcomes:
1. The item is at index $1$.
2. The item is at index $2$.
    $\dots$
3. The item is at index $n$.
4. The item is **not in the list**.

Therefore, any valid decision tree for searching must have at least **$L = n+1$ leaves**.

---
## The Mathematical Proof
We use the relationship between the number of leaves ($L$) and the height of a binary tree ($h$). The height $h$ represents the **worst-case number of comparisons**.
1. Binary Tree Property: A binary tree of height $h$ can have at most $2^h$ leaves.
    $$
    L \leq 2^h
    $$
    
2. Substitution: Since we need at least $n+1$ leaves:
    $$
    n+1 \leq 2^h
    $$
    
3. Solve for $h$:
    $$
    \begin{align*}
    \log_2(n+1) \leq h\\
    h \geq \lceil \log_2(n+1) \rceil
    \end{align*}
    $$
    

> [!IMPORTANT]
> 
> This proves that any algorithm that searches by comparing elements must perform at least $\approx \log_2 n$ comparisons in the worst case.

---
## Optimal vs. Sub-optimal
We can now categorize algorithms based on how close they get to this theoretical floor:

|**Algorithm**|**Worst Case**|**Lower Bound**|**Status**|
|---|---|---|---|
|**Linear Search**|$O(n)$|$\Omega(\log n)$|**Sub-optimal** (for sorted data)|
|**Binary Search**|$O(\log n)$|$\Omega(\log n)$|**Asymptotically Optimal**|

### Why is Linear Search $O(n)$?
Linear Search does not utilize the sorted property. It effectively builds a "degenerate" decision tree (a long chain) where the height is $n$ rather than $\log n$.

![[Pasted image 20251217193305.png]]

---
## Limitations: Can we go faster?
Can we ever search faster than $\log n$?
- **Non-Comparison Sorts**: If we use the _value_ of the data as an index (like a **Hash Table**), we can achieve $O(1)$ average time.
- **The Constraint**: The $\Omega(\log n)$ bound only applies to **comparison-based** algorithms where we only learn information through "is $x < a_i$?".

---
## Related Notes
- [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Searching/Binary Search]] — The algorithm that meets this lower bound.
- [[Asymptotic Notation]] — Understanding the $\Omega$ (Big-Omega) notation for lower bounds.
- [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Sorting/index|Sorting Algorithms]] — Sorting also has a lower bound of $\Omega(n \log n)$ based on decision trees.