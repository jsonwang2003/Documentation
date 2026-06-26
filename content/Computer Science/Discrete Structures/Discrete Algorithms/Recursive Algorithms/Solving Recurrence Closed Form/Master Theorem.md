> [!ABSTRACT]
> 
> The Master Theorem provides a "cookbook" solution for the asymptotic analysis of divide-and-conquer recurrences. It allows us to determine the [[Asymptotic Notation#3. Big-O ($O$)|Big-O]] complexity by simply comparing the rate of subproblem proliferation to the rate of work done at each level.

---
## The Formal Template

If $T(n) = aT(\frac{n}{b}) + f(n)$, where $f(n) = O(n^d)$, and:
- $a \geq 1$: The number of subproblems (how many branches the tree has).
- $b > 1$: The factor by which the subproblem size is reduced.
- $d \geq 0$: The exponent of the work done outside the recursive calls (the "cost of combining").

---
## The Three Cases (The Intuition)

The solution depends on the relationship between $a$ and $b^d$.

$$
T(n) = \begin{cases}
	O(n^d) &\text{if } a < b^d\\
	O(n^d\log(n)) &\text{if } a = b^d\\
	O(n^{\log_{b}(a)}) &\text{if } a > b^d
\end{cases}
$$
### Case 1: $a < b^d$ (The "Top-Heavy" Case)
- **Complexity:** $O(n^d)$
- **Meaning:** The cost of the work at the root (the "combine" step) is so high that it dominates the total runtime. The recursive calls are relatively "cheap."
- **Example:** $T(n) = 2T(n/2) + n^2 \implies O(n^2)$.
### Case 2: $a = b^d$ (The "Balanced" Case)
- **Complexity:** $O(n^d \log n)$
- **Meaning:** The work is distributed evenly across all levels of the recursion tree. Each level does the same amount of work, so we multiply the work per level by the number of levels ($\log n$).
- **Example:** $T(n) = 2T(n/2) + n \implies O(n \log n)$ (Merge Sort).
### Case 3: $a > b^d$ (The "Bottom-Heavy" Case)
- **Complexity:** $O(n^{\log_b a})$
- **Meaning:** The number of subproblems grows so quickly that the work at the "leaves" of the recursion tree dominates the total runtime.
- **Example:** $T(n) = 4T(n/2) + n \implies O(n^2)$.

---
## Step-by-Step Application
To solve any Master Theorem problem:
1. **Identify $a, b,$ and $d$** from the recurrence.
2. **Calculate $b^d$**.
3. **Compare $a$ vs $b^d$**:
    - If $a < b^d \rightarrow$ Answer is $O(n^d)$.
    - If $a = b^d \rightarrow$ Answer is $O(n^d \log n)$.
    - If $a > b^d \rightarrow$ Answer is $O(n^{\log_b a})$.

---
## Important Limitations
The Master Theorem **cannot** be used if:
- $a$ is not a constant (e.g., $T(n) = nT(n/2)$).
- $f(n)$ is not a polynomial (e.g., $f(n) = 2^n$ or $f(n) = \sin n$).
- $b$ is not a constant (e.g., $T(n) = T(\sqrt{n})$).
- **The Gap Case:** In advanced cases, if $f(n)$ is "just barely" smaller or larger than $n^{\log_b a}$ (e.g., by a factor smaller than a polynomial), the basic Master Theorem fails.

---
## Related Notes
- [[Unraveling]] – How we derive the Master Theorem.
- [[Computer Science/Discrete Structures/Discrete Algorithms/Algorithm Analysis/index|Algorithm Analysis]] – Understanding why $O(n \log n)$ is better than $O(n^2)$.
- [[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Divide and Conquer/index|Divide and Conquer]] – The algorithm design pattern that creates these recurrences.