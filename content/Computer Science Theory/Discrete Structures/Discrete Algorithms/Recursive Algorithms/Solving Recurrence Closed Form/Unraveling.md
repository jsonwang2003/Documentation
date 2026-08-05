
> [!ABSTRACT]
> 
> Unraveling (also known as the Iteration Method) involves repeatedly substituting a recurrence relation into itself until a clear pattern emerges. This pattern is then summed up—usually as a series—to find the closed-form solution.

---
## The Step-by-Step Process
1. **Substitution**: Replace the $T(\dots)$ term on the right-hand side with the definition of the recurrence itself.
2. **Pattern Recognition**: Do this $k$ times until you can write a general expression for the $k^{th}$ iteration.
3. **Base Case Convergence**: Determine what value of $k$ makes the input reach the **base case** (usually $n=1$ or $n=0$).
4. **Summation**: Substitute that $k$ back into your general expression and solve the resulting finite series.

---
# Examples
## **Problem 1:** 
Solve $T(n) = T(n-1) + n$ where $T(1) = 1$.

### 1. Unravel the first few steps
$$
\begin{align*}
T(n) &= T(n-1) + n\\
T(n) &= [T(n-2) + (n-1)] + n\\
T(n) &= [T(n-3) + (n-2)] + (n-1) + n\\
\end{align*}
$$
### 2. Generalize for $k$ steps
$$
T(n) = T(n-k) + \sum_{i=0}^{k-1} (n-i)
$$

### 3. Reach the Base Case
We want the inner term to be the base case: $n - k = 1$.

Therefore, let $k = n - 1$.
### 4. Solve the Series
Substitute $k$ into the general form:

$$
T(n) = T(1) + \sum_{i=0}^{n-2} (n-i)
$$

This expands to $1 + 2 + 3 + \dots + n$, which is the famous Arithmetic Series:

$$
\boxed{T(n) = \frac{n(n+1)}{2}}
$$

---

## **Problem 2:** 
Solve $T(n) = 2T(n/2) + n$ (The Merge Sort recurrence).
### 1. Unraveling
$$
\begin{align*}
T(n) &= 2[2T(n/4) + n/2] + n \\
&= 4T(n/4) + n + n \\
&= 4T(n/4) + 2n\\
T(n) &= 4[2T(n/8) + n/4] + 2n \\
&= 8T(n/8) + n + 2n \\
&= 8T(n/8) + 3n
\end{align*}
$$
### 2. Generalize ($k^{th}$ iteration)

$$
T(n) = 2^k T\left(\frac{n}{2^k}\right) + kn
$$

### 3. Reach Base Case

Set $\frac{n}{2^k} = 1 \implies n = 2^k \implies \mathbf{k = \log_2 n}$.

### 4. Final Solution

$$
T(n) = n T(1) + n \log_2 n
$$

Assuming $T(1) = 1$:

$$
\boxed{T(n) = n + n \log_2 n \approx O(n \log n)}
$$

---
## Common Pitfalls
- **Algebraic Errors**: It is very easy to lose a coefficient (like the $2$ in $2T(n/2)$) during the second or third expansion. Always simplify at each step.
- **Off-by-one errors**: Be careful whether your summation ends at $k$ or $k-1$.
- **Non-constant work**: If the "extra work" (the $+n$ part) changes in a complex way, the summation can become difficult to solve.

---
## Related Notes
- [[Master Theorem]] - A faster way to solve the Divide & Conquer example above.
- [[Sum of an Arithmetic Series]] - Essential for solving linear unraveling.
- [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Recursive Algorithms/index|Recursive Algorithms]] - The logic behind why we unravel.