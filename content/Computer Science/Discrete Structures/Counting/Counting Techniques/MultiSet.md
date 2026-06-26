---
tags:
  - Counting
  - Multiset
  - Repetition
  - Combinatorics
  - Multichoose
  - Stars-And-Bars
---
> [!INFO]
> A **multiset** is a collection of elements where **repetition is allowed**.  
> Unlike a traditional set, a multiset can contain multiple instances of the same element.

Example:

$$
\{1, 1, 2, 3\}
$$

This multiset contains:
- Two copies of 1
- One copy of 2
- One copy of 3

---

### Key Properties

- **Order does not matter**, but **multiplicity does**.
- Multisets are often used in:
  - **Anagram counting**
  - **Combinatorics with repetition**
  - **Partitioning problems**
  - **Multichoose formulas**

---

### Multichoose: Combinations with Repetition

To count the number of multisets (unordered selections **with** repetition) of size $k$ from a set of $n$ distinct elements:

$$
\text{Multichoose}(n, k) = \binom{n + k - 1}{k}
$$

> [!NOTE]
> This is often visualized using the **Stars and Bars** method. See [[Stars and Bars]].

---

### Example: Choosing 3 Fruits from {Apple, Banana, Cherry}

You can choose:
- 3 Apples
- 2 Apples + 1 Banana
- 1 Apple + 1 Banana + 1 Cherry
- etc.

Total number of multisets:

$$
\binom{3 + 3 - 1}{3} = \binom{5}{3} = 10
$$

---
### Connection to Anagrams
Multisets are the foundation for counting anagrams of words with repeated letters.  
See [[Anagram Counting]] for worked examples.

---
## Related Topic
- [[Set Theory]] - 