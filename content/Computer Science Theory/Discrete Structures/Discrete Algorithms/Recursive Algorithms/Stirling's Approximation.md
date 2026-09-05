> [!ABSTRACT]
> 
> Factorials ($n!$) grow extremely fast, making them difficult to compute directly for large $n$. Stirling's Approximation provides a way to estimate these values using continuous functions, while recursive partitioning helps us understand their combinatorial origin in permutations.

---
## 1. Stirling's Approximation
When analyzing algorithm complexity (especially in [[Asymptotic Notation]]), we often need a "smooth" function to represent $n!$. Stirling's formula provides a precise approximation:

$$
n! \approx \sqrt{2 \pi n} \left(\frac{n}{e}\right)^n
$$

### Accuracy Comparison

As $n$ increases, the relative error of the approximation decreases, making it invaluable for large-scale system analysis.

|**n**|**n! (Actual)**|**Stirling's Approximation**|**Relative Error**|
|---|---|---|---|
|1|1|0.92|~8.0%|
|5|120|118.02|~1.6%|
|10|3,628,800|3,598,695.62|~0.8%|

> [!IMPORTANT]
> 
> Because this is an approximation, it is not a closed-form equivalent for discrete calculations, but it is used to prove that $\log(n!) \in \Theta(n \log n)$.

---
## 2. Permutations: The Combinatorial Origin

To understand why $n!$ appears in algorithms, we look at the number of ways to arrange a set of size $n$, denoted as $S(n)$.

### Recursive Partitioning

We can calculate $S(n)$ by partitioning the set of all permutations based on their starting element. For a set $\{1, 2, \dots, n\}$:
1. There are **$n$ possible starting elements**.
2. Once the first element is chosen, we must arrange the remaining **$n-1$ elements**.
3. This creates the recurrence relation: $S(n) = n \cdot S(n-1)$.
### Unraveling the Recurrence
By "unrolling" this recursive definition, we arrive at the factorial:

$$
\begin{align*} 
S(n) &= n \cdot S(n-1) \\
S(n) &= n \cdot (n-1) \cdot S(n-2) \\
S(n) &= n \cdot (n-1) \cdot (n-2) \cdots 1 \cdot S(0) \\
S(n) &= n! 
\end{align*}
$$

_(Note: By convention, $S(0) = 0! = 1$, representing the single way to arrange an empty set.)_

---
## 3. Computational Impact

In [[Runtime of Algorithms]], factorials represent a "complexity wall."
- **Polynomial Time**: $O(n^k)$ is usually manageable.
- **Factorial Time**: $O(n!)$ is catastrophic. Even for $n=20$, $n!$ is approximately $2.4 \times 10^{18}$. On a 1 GHz processor, an $O(n!)$ algorithm would take over **70 years** to complete for $n=20$.

---
## 4. Connection to Other Identities
The factorial is the backbone of many [[Binomial Theorem|Binomial Identities]]:
- **[[Symmetry Identity]]**: Uses factorials to show $\binom{n}{k} = \binom{n}{n-k}$.
- **[[rPermutations|r-Permutations]]**: Defines the number of ways to pick and arrange $r$ elements from $n$ as $\frac{n!}{(n-r)!}$.