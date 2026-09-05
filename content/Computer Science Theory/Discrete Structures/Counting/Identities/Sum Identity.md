> [!ABSTRACT]
> 
> The Sum Identity, also known as the Sum of Binomial Coefficients, states that the sum of all entries in the $n$-th row of Pascal's Triangle is exactly $2^n$. This identity provides the fundamental link between binomial expansion and the total number of subsets in a power set.

---
## The Identity

$$
\sum_{k = 0}^{n}\binom{n}{k} = \binom{n}{0} + \binom{n}{1} + \binom{n}{2} + \dots + \binom{n}{n} = 2^n
$$

---
## 1. Algebraic Proof
Using the **[[Binomial Theorem|Binomial Theorem]]**, we can expand $(x+y)^n$:

$$
(x+y)^n = \sum_{k=0}^{n}\binom{n}{k}x^{n-k}y^k
$$

By substituting $x = 1$ and $y = 1$:

$$
\begin{align*} 
(1+1)^n &= \sum_{k = 0}^{n} \binom{n}{k}(1)^{n-k}(1)^k \\ 
2^n &= \sum_{k = 0}^n \binom{n}{k} \cdot 1 \cdot 1 \\ 
2^n &= \binom{n}{0} + \binom{n}{1} + \dots + \binom{n}{n} 
\end{align*}
$$

---
## 2. Combinatorial Proof
We can prove this by showing that both sides of the equation count the same set of objects: **the total number of binary strings of length $n$.**
- **RHS ($2^n$)**: For a binary string of length $n$, each of the $n$ positions has 2 choices (`0` or `1`). By the **[[Product Rule|Product Rule]]**, there are $2 \times 2 \times \dots \times 2 = 2^n$ total strings.
- **LHS ($\sum \binom{n}{k}$)**: We can partition the set of all binary strings by their _weight_ (the number of `1`s they contain).
    - Strings with zero `1`s: $\binom{n}{0}$
    - Strings with one `1`: $\binom{n}{1}$
    - Strings with $k$ 1s: $\binom{n}{k}$        

> [!INFO]
> By the [[Sum Rule|Sum Rule]], the total number of strings is the sum of these disjoint cases from $k=0$ to $n$.

---
## 3. Connection to Power Sets
This identity also counts the total number of subsets of a set $S$ where $|S| = n$:
- $\binom{n}{k}$ represents the number of ways to choose a subset of size $k$.
- Summing from $k=0$ (the empty set) to $k=n$ (the set itself) gives every possible subset.
- Therefore, the size of the **[[Power Rule|Power Set]]** is always $2^n$.

---
## Example ($n = 4$)

$$
\begin{align*} 
2^4 &= \binom{4}{0} + \binom{4}{1} + \binom{4}{2} + \binom{4}{3} + \binom{4}{4}\\ 
16 &= 1 + 4 + 6 + 4 + 1\\ 
16 &= 16 
\end{align*}
$$

- **1** empty set $\emptyset$
- **4** subsets of size 1
- **6** subsets of size 2
- **4** subsets of size 3
- **1** subset of size 4 (the set itself)