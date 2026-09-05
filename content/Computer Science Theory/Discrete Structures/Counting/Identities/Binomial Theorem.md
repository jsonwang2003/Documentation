> [!ABSTRACT]
> 
> The Binomial Theorem provides a powerful algebraic method for expanding expressions of the form $(x + y)^n$. In combinatorics, it serves as a fundamental identity that links algebra to counting, specifically through the use of Combinations (binomial coefficients).

---
## 1. The Theorem
For any non-negative integer $n$, the expansion of $(x + y)^n$ is given by:

$$
(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k
$$

Expanding this summation, we get:

$$
\begin{align*}
&(x + y)^n \\&= \binom{n}{0}x^n y^0 + \binom{n}{1}x^{n-1} y^1 + \binom{n}{2}x^{n-2} y^2 + \dots + \binom{n}{n}x^0 y^n
\end{align*}
$$

![[Pasted image 20251003132047.png]]

---
## 2. Combinatorial Interpretation
The coefficient $\binom{n}{k}$, often read as "$n$ choose $k$," represents the number of ways to choose $k$ items from a set of $n$.

In the context of the expansion $(x + y)(x + y)\dots(x + y)$, to obtain the term $x^{n-k}y^k$, we must choose the variable $y$ from exactly $k$ of the $n$ available binomial factors. The remaining $n-k$ factors must contribute an $x$.
- There are $\binom{n}{k}$ ways to make this selection.
- This is why binomial coefficients are also the entries in **[[Pascal's Identity|Pascal's Triangle]]**.

---
## 3. Useful Identities derived from the Theorem
By substituting specific values for $x$ and $y$, we can derive several important counting identities:
### The Sum of Coefficients
Let $x = 1$ and $y = 1$:

$$
\begin{align*}
&(1 + 1)^n = \sum_{k=0}^{n} \binom{n}{k} \\
\implies &2^n = \binom{n}{0} + \binom{n}{1} + \dots + \binom{n}{n}
\end{align*}
$$

- **Counting Meaning**: The total number of subsets of a set of size $n$ (the **[[Power Rule|Power Set]]**) is $2^n$.
### Alternating Sum
Let $x = 1$ and $y = -1$:

$$
\begin{align*}
&(1 - 1)^n = \sum_{k=0}^{n} \binom{n}{k} (-1)^k \\
\implies &0 = \binom{n}{0} - \binom{n}{1} + \binom{n}{2} - \dots
\end{align*}
$$

- **Counting Meaning**: For any non-empty set, the number of subsets of even size is exactly equal to the number of subsets of odd size.

---
## 4. Connection to De Morgan's Law and PIE
While the Binomial Theorem describes how to expand a union-like algebraic structure, it is often used to simplify the terms found in the **[[Inclusion Exclusion|Principle of Inclusion-Exclusion (PIE)]]**.

When applying PIE to $n$ properties where each intersection of $k$ properties has the same size (symmetry), the formula simplifies into a binomial expansion pattern:

$$
\begin{align*}
&|S| - \sum |A_i| + \sum |A_i \cap A_j| \dots \\
\implies &\sum_{k=0}^n (-1)^k \binom{n}{k} N(k)
\end{align*}
$$

This structure mirrors the alternating sum derived from the Binomial Theorem, often allowing complex counting problems to be reduced to a single compact expression.