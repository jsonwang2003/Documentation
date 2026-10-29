---
title: Product Rule
description: Fundamental counting rule for determining the size of Cartesian products or sequential choices.
tags:
  - discrete-structures
  - counting
aliases:
  - "Multiplication Rule"
---
> [!abstract] Abstract
> **Definition:** For any sets $A$ and $B$, $|A \times B| = |A| |B|$.
> **Context:** Used to count pairs or sequential independent choices.
> **Prerequisites:** [[Counting and Probability]]

## Concepts

To count the **number of pairs of objects** or sequential choices:
1. Count the number of choices for **selecting the first object**.
2. Count the number of choices for **selecting the second object**.
3. **Multiply** these two counts.

$$ |A_1 \times A_2 \times \dots \times A_k| = |A_1| \cdot |A_2| \cdot \dots \cdot |A_k| $$

> [!CAUTION]
> Selecting the second object will only work if the number of choices for the second object **does not depend** on which first object we choose.

## Examples

Walk through a concrete example or derivation step-by-step.

1. **Setup:** Choose a shirt (3 options) and pants (4 options).
2. **Apply:** $|Shirt| \cdot |Pants| = 3 \cdot 4$.
3. **Result:** 12 total outfits.

## Edge Cases & Pitfalls

If the choices for subsequent steps depend on prior choices (dependent variables), the Product Rule cannot be applied directly; you must condition the choices or use a tree model.

---
# Related Notes

- [[Sum Rule]]
- [[Quotient Rule]]
- [[Power Rule]]
