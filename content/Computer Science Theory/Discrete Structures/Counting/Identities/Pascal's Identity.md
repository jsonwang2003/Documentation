> [!ABSTRACT]
> 
> Pascal's Identity states that $\binom{n+1}{k} = \binom{n}{k-1} + \binom{n}{k}$. Combinatorially, this represents splitting a selection process into two disjoint cases based on whether a specific "special" element is included or excluded.

---
## Proof 1: Algebraic Manipulation
We can prove the identity by expanding the binomial coefficients into their factorial forms and finding a common denominator.

**LHS:**

$$
\frac{(n+1)!}{k!((n+1)-k)!}
$$

**RHS:**

$$
\binom{n}{k-1} + \binom{n}{k} = \frac{n!}{(k-1)!(n-k+1)!} + \frac{n!}{k!(n-k)!}
$$

1. **Find Common Denominator**: The common denominator is $k!(n-k+1)!$.    
2. **Multiply to match**:
    - Multiply the first term by $\frac{k}{k}$.
    - Multiply the second term by $\frac{n-k+1}{n-k+1}$.
        $$
        \frac{n! \cdot k}{k!(n-k+1)!} + \frac{n! \cdot (n-k+1)}{k!(n-k+1)!}
        $$
        
3. **Combine and Simplify**:
    $$
    \frac{n!(k + n - k + 1)}{k!(n-k+1)!} = \frac{n!(n+1)}{k!(n-k+1)!} = \frac{(n+1)!}{k!(n+1-k)!}
    $$
    
    **LHS = RHS**

---
## Proof 2: Combinatorial Interpretation
As you noted, we can think of this as counting binary strings of length $n+1$ with exactly $k$ ones. By the **[[Sum Rule|Sum Rule]]**, we can partition all such strings into two mutually exclusive sets based on the value of the **last bit**.
### LHS: Total Count
The total number of binary strings of length $n+1$ with $k$ ones is $\binom{n+1}{k}$.

### RHS: The Partition
- **Case 1**: The string ends in 1
    If the last bit is a 1, we have $n$ remaining positions to fill and we only need $k-1$ more 1s.
    Number of ways: $\binom{n}{k-1}$
- **Case 2**: The string ends in 0
    If the last bit is a 0, we still have $n$ positions to fill, but we still need to place all $k$ of our 1s.
    Number of ways: $\binom{n}{k}$

Since a string must end in either `1` or `0`, the total is the sum of these two cases.

---
## Pascal's Triangle Connection
This identity explains the structure of **Pascal's Triangle**. 

![[Pasted image 20251003133435.png]]

Each interior number is the sum of the two numbers above it because those two numbers represent the "Include" and "Exclude" cases for a set of size $n$.
### Visual Example: $\binom{5}{3} = \binom{4}{2} + \binom{4}{3}$
- **$\binom{5}{3}$**: Ways to pick 3 people for a team out of 5 friends (Alice, Bob, Charlie, Dan, Eve).
- **$\binom{4}{2}$**: Teams that **include** Alice. (We must pick 2 more from the remaining 4).
- **$\binom{4}{3}$**: Teams that **exclude** Alice. (We must pick all 3 from the remaining 4).