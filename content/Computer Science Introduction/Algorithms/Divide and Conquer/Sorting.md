---
aliases:
  - Sorting Algorithms
description: Foundational concepts for comparison-based sorting — the decision-tree argument establishing the Ω(n log n) lower bound that Merge Sort and Quick Sort operate under.
tags:
  - sorting
  - divide-and-conquer
---
> [!abstract] Overview 
> Before diving into specific sorting algorithms like [[Computer Science Introduction/Algorithms/Divide and Conquer/Merge Sort]] and [[Quick Sort]], it's worth establishing _how fast sorting can possibly be_ — every comparison-based sorting algorithm is bound by the same $\Omega(n\log n)$ limit, which comes from a simple counting argument, not from any particular algorithm's design.

---

# Foundational Concepts

## The Comparison Model

A **comparison-based** sorting algorithm only learns information about the input by comparing pairs of elements (e.g. "is $a < b$?") — it never inspects or exploits the actual values otherwise. [[Computer Science Introduction/Algorithms/Divide and Conquer/Merge Sort]] and [[Quick Sort]] are both comparison-based.

## The Decision Tree Argument

Any comparison sort can be modeled as a binary **decision tree**: each internal node represents one comparison, its two children represent the two possible outcomes, and each leaf represents one fully-determined final ordering.

![[Pasted image 20260709135749.png]]

> [!tip] Key Idea 
> If we must sort things based on comparisons, we must travel down a path in this tree — every run of a comparison-based sort corresponds to exactly one root-to-leaf path, and the depth of that path is the number of comparisons made on that particular input.

To correctly sort $n$ distinct elements, the tree must be able to distinguish all $n!$ possible orderings of them — so it needs at least $n!$ leaves. Since a binary tree of height $h$ has at most $2^h$ leaves, this forces:

$$ 
2^h \geq n! \implies h \geq \log(n!) 
$$

So **any sorting algorithm that relies on comparisons between elements runs in $\Omega(\log(n!))$ time** — no comparison-based algorithm can beat this, regardless of how cleverly it's designed.

## The Ω(n log n) Lower Bound

$\log(n!)$ isn't just some awkward expression — by Stirling's approximation, $\log(n!) = \Theta(n\log n)$. Roughly:

$$
\log(n!) = \sum_{i=1}^{n}\log i \approx \int_1^n \log x , dx = n\log n - n = \Theta(n\log n) 
$$

So **no comparison-based sorting algorithm can do better than $\Omega(n\log n)$ in the worst case.** This is why [[Computer Science Introduction/Algorithms/Divide and Conquer/Merge Sort]]'s $\Theta(n\log n)$ worst-case bound is considered _optimal_ — it's not just a good algorithm, it's asymptotically as good as any comparison sort can ever be.

## Faster-Than-Comparison Sorts Exist — With Caveats

There are sorting algorithms out there that run faster than $O(n\log n)$ (e.g. Counting Sort, Radix Sort), but they rely on prior knowledge about the elements — for example, values are only allowed to come from a small range. These algorithms don't violate the lower bound above, because they aren't purely comparison-based; the $\Omega(n\log n)$ bound only applies to algorithms that learn about the input _exclusively_ through pairwise comparisons.

## Worked Example: Sorting 4 Elements

- **Decision tree lower bound:** sorting 4 elements should take $\lceil \log(4!) \rceil = \lceil \log(24) \rceil = 5$ comparisons at minimum.
- **Naive all-pairs approach:** comparing every pair of elements takes $\binom{4}{2} = 6$ comparisons — one more than necessary. This shows that brute-force pairwise comparison isn't optimal; a well-designed algorithm can sort 4 elements in exactly 5 comparisons by reusing information from earlier comparisons instead of re-deriving it.

## Deterministic vs. Randomized

| |Deterministic|Randomized|
|---|---|---|
|**Sorting**|[[Computer Science Introduction/Algorithms/Divide and Conquer/Merge Sort]] — $O(n\log n)$|[[Quick Sort]] — Best: $O(n\log n)$, Worst: $O(n^2)$, Average: $O(n\log n)$|

See [[Computer Science Introduction/Algorithms/Divide and Conquer/index#Deterministic vs. Randomized Approaches|Deterministic vs. Randomized Approaches]] for the full table including [[Selection]].

---

# Notes in This Section

|Note|One-line description|
|---|---|
|[[Computer Science Introduction/Algorithms/Divide and Conquer/Merge Sort]]|Divide and conquer sort — splits, recursively sorts each half, merges the sorted halves; $\Theta(n\log n)$ worst case, matching the comparison lower bound exactly|
|[[Quick Sort]]|Divide and conquer sort — partitions around a pivot, recursively sorts each side; $O(n\log n)$ expected, $O(n^2)$ worst case|

---

# Related Categories

- [[Computer Science Introduction/Algorithms/Divide and Conquer/index|Divide and Conquer]]