---
title: Sum Rule
description: Fundamental counting rule for the union of disjoint sets or alternative choices.
tags:
  - discrete-structures
  - counting
aliases:
  - "Addition Rule"
---
> [!abstract] Abstract
> **Definition:** For any **disjoint** sets $A$ and $B$, $|A \cup B| = |A| + |B|$.
> **Context:** Used to count alternative choices (keyword: **OR**).
> **Prerequisites:** [[Counting and Probability]]

## Concepts

To count the number of alternative disjoint choices:
1. Count the number of choices in the **first disjoint set**.
2. Count the number of choices in the **second disjoint set**.
3. **Add** those two counts.

$$ |A \cup B| = |A| + |B| \quad \text{when } A \cap B = \emptyset $$

> [!CAUTION]
> This rule only applies when the sets are **disjoint**—i.e., they share no elements. If sets overlap, use the **Principle of Inclusion-Exclusion** instead.

## Examples

Suppose you have:
- 5 red shirts
- 7 blue shirts  
If you want to know how many shirts you can choose **either red OR blue**, the total is:
$$ 5 + 7 = 12 $$

## Edge Cases & Pitfalls

If sets are **not disjoint**, then:
$$ |A \cup B| = |A| + |B| - |A \cap B| $$
This is a basic form of the **[[Inclusion Exclusion|Inclusion-Exclusion Principle]]**.

---
# Related Notes

- [[Product Rule]]
- [[Inclusion Exclusion]]
