> [!ABSTRACT]
> 
> The Symmetry Identity states that $\binom{n}{k} = \binom{n}{n-k}$. This principle highlights the inherent balance in Pascal's Triangle, reflecting the fact that choosing a group to "include" is mathematically identical to choosing a group to "exclude."

---
## 1. Algebraic Proof
By applying the factorial definition of a binomial coefficient, the symmetry becomes clear through the commutative property of multiplication.

$$\text{LHS: } \binom{n}{k} = \frac{n!}{k!(n-k)!}$$

$$\text{RHS: } \binom{n}{n-k} = \frac{n!}{(n-k)!(n-(n-k))!} = \frac{n!}{(n-k)!k!}$$

Since the denominators are identical, the two expressions are equal.

---
## 2. Combinatorial Proofs
### The "Selection is Rejection" Argument
- **LHS**: Represents the number of ways to choose $k$ objects from a set of $n$ to be in a committee.
- **RHS**: Represents the number of ways to choose $n-k$ objects from a set of $n$ to be left out of the committee.
- **Logic**: Every time you pick $k$ people to be "in," you are simultaneously and uniquely picking $n-k$ people to be "out." Because every selection of a subset uniquely determines its complement, the number of ways to do both must be equal.

### The Bijection (Bit-Flipping) Argument
- $\binom{n}{k}$ counts the number of fixed-density binary strings of length $n$ with $k$ ones.
- $\binom{n}{n-k}$ counts the number of fixed-density binary strings of length $n$ with $n-k$ ones.
- **Proof**: We can define a **bijection** (a one-to-one mapping) between these two sets by **flipping every bit** (turning every `1` into a `0` and every `0` into a `1`).
    - Example: For $n=3, k=1$, flipping the string `100` (from $\binom{3}{1}$) gives `011` (from $\binom{3}{2}$).
    - Since every string in the first set maps to exactly one unique string in the second set, the two quantities must be equal.

---
## 3. Visualizing Symmetry
This identity explains why Pascal's Triangle is a mirror image of itself. For any row $n$:
- The 1st element equals the last element: $\binom{n}{0} = \binom{n}{n} = 1$
- The 2nd element equals the second-to-last: $\binom{n}{1} = \binom{n}{n-1} = n$
### Example ($n=4$)
Using the row $1, 4, 6, 4, 1$:
- $\binom{4}{1} = 4$ (Choosing 1 object to keep)
- $\binom{4}{3} = 4$ (Choosing 3 objects to throw away)

---
## 4. Connection to De Morgan's Law

The Symmetry Identity is the combinatorial cousin of **[[Identities/Demorgan's Law|De Morgan's Law]]**. While De Morgan's deals with the logic of complements ($\overline{A \cup B} = \overline{A} \cap \overline{B}$), the Symmetry Identity deals with the _counting_ of complements. Both rely on the fundamental relationship between a set and everything not in that set.