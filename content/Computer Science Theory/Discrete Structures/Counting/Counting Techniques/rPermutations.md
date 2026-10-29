---
title: r-Permutations
description: Counting the number of ways to arrange r distinct objects selected from a set of n objects.
tags:
  - discrete-structures
  - counting
  - permutations
aliases:
  - "nPr"
  - "Permutations of r out of n"
---
> [!abstract] Abstract
> **Definition:** The number of ways to arrange $r$ objects out of $n$ distinct objects.
> **Context:** Ordered arrangements where position matters.
> **Prerequisites:** [[Product Rule]], [[Combinations]]

## Concepts

An $r$-permutation is an ordered arrangement of $r$ elements taken from a set of $n$ distinct elements.

$$
\begin{align*}
P(n, r) &= nPr \\
&= n(n - 1)(n - 2)\dots(n - r + 1) \\
&= \frac{n!}{(n - r)!}
\end{align*}
$$

> [!IMPORTANT]
> By convention, $0! = 1$.

## Examples

1. **Setup:** Selecting a 1st, 2nd, and 3rd place winner from 10 runners ($n=10, r=3$).
2. **Apply:** $P(10, 3) = \frac{10!}{(10-3)!} = \frac{10!}{7!} = 10 \cdot 9 \cdot 8$.
3. **Result:** 720 possible outcomes.

## Edge Cases & Pitfalls

- If $r > n$, $P(n, r) = 0$ (you cannot choose more items than exist without replacement).
- Order matters in permutations, unlike combinations.

---
# Related Notes

- [[Combinations]]
- [[Product Rule]]
- [[MultiSet]]
