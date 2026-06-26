---
title: Counting Techniques
---
# The Combinatorics Toolbox: Counting Techniques

> [!ABSTRACT]
> 
> This module covers the operational mechanics of counting. The goal is to transform word problems into mathematical expressions by identifying the underlying constraints of the set.

---

## The Four Fundamental Rules

_The "Bread and Butter" of counting. Use these to break down complex problems into smaller parts._
- ### **[[Sum Rule]]**
    - **When to use:** When you have a choice between mutually exclusive tasks (**OR**).
    - **Core Logic:** If Task A has $n$ ways and Task B has $m$ ways, and they cannot happen together, total = $n + m$.
- ### **[[Product Rule]]**
    - **When to use:** When a procedure consists of a sequence of tasks (**AND**).
    - **Core Logic:** The fundamental counting principle; multiply the number of ways for each independent step.1
- ### **[[Quotient Rule]]**
    - **When to use:** When your counting method overcounts identical outcomes (Symmetry).
    - **Core Logic:** Total = (Total with overcounting) / (Number of times each item is repeated).
- ### **[[Power Rule]]**
    - **When to use:** For sequences where repetition is allowed (e.g., bit strings or passwords).
    - **Core Logic:** $n^k$ (choosing from $n$ items $k$ times).

---
## Permutations & Combinations

_The core logic of selection. The "Golden Rule" here is: **Does order matter?**_

|**Technique**|**Order Matters?**|**Repetition?**|**Formula**|
|---|---|---|---|
|**[[rPermutations]]**|Yes|No|$P(n, r) = \frac{n!}{(n-r)!}$|
|**[[Combinations]]**|No|No|$C(n, r) = \frac{n!}{r!(n-r)!}$|
|**[[MultiSet]]**|No|Yes|Often referred to as "Combinations with repetition."|

---
## Advanced Selection Strategies

_Specialized "patterns" for tricky constraints._

### Arrangement & Distribution
- **[[Anagram Counting]]**: Using permutations of multisets to find arrangements of words with repeating letters.
- **[[Binary Strings]]**: Techniques for counting bit patterns (0s and 1s) under specific constraints (e.g., "no two 0s adjacent").
- **[[Stars and Bars]]**: The go-to method for distributing 2$n$ identical objects into 3$k$ distinct bins.4

### Logical Filters
- **[[Inclusion Exclusion]]**: A technique to calculate the size of the union of multiple sets by accounting for their intersections.5
    - _Essential for problems involving "at least one" or "none of the following."_