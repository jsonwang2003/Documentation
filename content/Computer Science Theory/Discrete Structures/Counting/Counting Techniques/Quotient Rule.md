---
tags:
  - Quotient-rule
  - Counting
---

> [!INFO]
> If a set $A$ can be partitioned into disjoint subsets $X_1, X_2, \dots, X_k$ each having the **same number of elements**,  
> that is, $|X_1| = |X_2| = \dots = |X_k|$,  
> then the number of subsets is:

$$
k = \frac{|A|}{|X_1|}
$$

---

### When to Use the Quotient Rule

This rule is useful when a set can be expanded into **equally sized subsets**, and each subset is easy to count. It helps eliminate overcounting due to symmetry or repeated structure.

Common applications include:
- **Anagram counting**: dividing total permutations by repeated character counts  
  → See [[Anagram Counting]]
- **Counting objects with orientation**: e.g., necklaces, bracelets, or rotational symmetries
- **Combinations of objects**: when multiple arrangements map to the same outcome
- **Fixed-density binary strings**: e.g., strings with exactly $k$ ones and $n-k$ zeros

---

### Example: Anagram

How many distinct anagrams of the word REASSESS?

- Total letters: 8  
- Frequencies: R (1), E (2), A (1), S (4)

$$
\frac{8!}{1! \cdot 2! \cdot 1! \cdot 4!}
$$

> [!NOTE]
> The denominator accounts for indistinguishable permutations due to repeated letters.