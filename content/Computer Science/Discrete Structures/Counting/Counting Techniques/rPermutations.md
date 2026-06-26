---
tags:
  - Counting
  - Multiset
  - Permutations
---
## r-Permutations

> [!INFO]
> The number of ways to arrange $r$ objects out of $n$  
> Rearrangement or ordering of $n$ **distinct objects** so that each object appears **exactly once**

$$
P(n, r) = nPr = n(n - 1)(n - 2)\dots(n - r + 1) = \frac{n!}{(n - r)!}
$$

> [!IMPORTANT]
> $0! = 1$

---

## Multisets

> [!INFO]
> A **multiset** is a collection that allows for repeated elements.  
> Example: $\{1, 1, 2, 3\}$ is a multiset.

Multisets are essential when counting permutations of objects that are **not all distinct**.