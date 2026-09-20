---
title: Divide and Conquer
description: Overview of the Divide and Conquer paradigm, the classic and Karatsuba multiplication algorithms, and the Master Theorem used to analyze their runtimes.
tags:
  - divide-and-conquer
aliases:
  - Divide and Conquer
---
> [!abstract] Overview 
> Divide and Conquer is an algorithmic paradigm that breaks a problem into similar subproblems, solves each recursively, and combines the results[cite: 70]. Every algorithm in this section produces a recurrence of the form $T(n) = aT(n/b) + O(n^d)$, which the Master Theorem solves in closed form[cite: 70].

---
## Foundational Concepts

**The Paradigm**
- Break a problem into similar subproblems[cite: 70].
- Solve each subproblem recursively[cite: 70].
- Combine the subproblem results into a solution for the original problem[cite: 70].
- Subproblems must be smaller instances of the same problem to ensure the recursion terminates and to enable Master Theorem-style analysis[cite: 70].

**Multiplying $n$-bit Numbers**
Suppose we want to multiply two $n$-bit numbers, where $n$ is a power of 2, by splitting each into left and right halves of $n/2$ bits each[cite: 70]:
$$x = 2^{n/2}x_{L} + x_{R}$$[cite: 70]
$$y = 2^{n/2}y_{L} + y_{R}$$[cite: 70]
$$xy = 2^{n}x_{L}y_{L} + 2^{n/2}(x_{L}y_{R} + x_{R}y_{L}) + x_{R}y_{R}$$[cite: 70]

- **Classic Recursive Approach:** Uses 4 recursive calls ($x_{L}y_{L}$, $x_{L}y_{R}$, $x_{R}y_{L}$, $x_{R}y_{R}$) resulting in $T(n) = 4T(n/2) + O(n)$[cite: 70]. By the Master Theorem, $4 > 2^1$, yielding $O(n^2)$, which is asymptotically no better than grade-school multiplication[cite: 70].
- **Karatsuba's Algorithm (Multiply KS):** Computes $(x_L+x_R)(y_L+y_R)$ once and subtracts off the two direct products to recover the cross terms, turning 4 recursive multiplications into 3[cite: 70]. This produces $T_{KS}(n) = 3T_{KS}(n/2) + O(n)$, yielding $O(n^{\log_2 3}) \approx O(n^{1.585})$[cite: 70].

---
## Master Theorem & Complexity Analysis

> [!info] Master Theorem
> If $T(n) = aT\left(\frac{n}{b}\right) + O(n^d)$ for constants $a>0, b>1, d\geq 0$, then:
> - **Case 1 ($a < b^d$):** $T(n) \in O(n^d)$[cite: 70].
> - **Case 2 ($a = b^d$):** $T(n) \in O(n^d\log n)$[cite: 70].
> - **Case 3 ($a > b^d$):** $T(n) \in O(n^{\log_b a})$[cite: 70].

**Proof via Geometric Series:**
After $k$ levels of recursion, there are $a^k$ subproblems, each of size $n/b^k$[cite: 70]. The work at level $k$ is $O(n^d(\frac{a}{b^d})^k)$[cite: 70]. Summing this over $\log_b n$ levels yields a geometric series with ratio $r = \frac{a}{b^d}$[cite: 70].
- If $r < 1$ (Case 1), the series converges to a constant[cite: 70].
- If $r = 1$ (Case 2), every term equals 1, meaning the sum is just the number of terms[cite: 70].
- If $r > 1$ (Case 3), the sum is exponential and grows proportional to its last term[cite: 70].

**Deterministic vs. Randomized Tradeoffs:**
Sorting and Selection each have deterministic and randomized solutions, trading a worse worst-case bound for simpler algorithms and better typical-case performance[cite: 70].
- **Sorting:** [[5. Sorting/5.1 Merge Sort|Merge Sort]] is deterministic $O(n\log n)$[cite: 70]. [[5. Sorting/5.2 Quick Sort|Quick Sort]] is randomized, expecting $O(n\log n)$ but risking $O(n^2)$ worst-case[cite: 70].
- **Selection:** [[4. Selection/4.2. Deterministic Selection|Deterministic Selection]] (Median of Medians) guarantees $O(n)$[cite: 70]. [[4. Selection/4.1 Quick Select|Quick Select]] expects $O(n)$ but risks $O(n^2)$ worst-case[cite: 70].

---
## Notes in This Section

| Note                                                                                               | Description                                                                                                                                              |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[1. Cook-Toom-k Algorithm\|Cook-Toom-k Algorithm]]                                                | Generalizes Karatsuba's trick to split into $k$ parts instead of 2, trading extra combine-step overhead for fewer recursive multiplications[cite: 70].   |
| [[2. Binary Search\|Binary Search]]                                                                | Halves the search space each comparison on a sorted array; $O(\log n)$[cite: 70].                                                                        |
| [[3. Two Runners\|Two-Runners]]                                                                    | Binary search for the "turning point" where a slower-starting runner overtakes a faster one; $O(\log n)$[cite: 70].                                      |
| [[Computer Science Introduction/Algorithms/III. Divide and Conquer/4. Selection/index\|Selection]] | The general "find the $k^{th}$ smallest element" problem, and the shared in-place `Partition with Pivot` subroutine[cite: 70].                           |
| [[4.1 Quick Select\|Quick Select]]                                                                 | Quick Sort-style partitioning used to find the $k^{th}$ smallest element directly, via a random pivot; $O(n)$ expected, $O(n^{2})$ worst case[cite: 70]. |
| [[4.2. Deterministic Selection\|Deterministic Selection]]                                          | Median-of-medians (BFPRT) — splits into groups of 5 to construct a provably-good pivot, guaranteeing $O(n)$ worst case without randomization[cite: 70].  |
| [[Computer Science Introduction/Algorithms/III. Divide and Conquer/5. Sorting/index\|Soritng]]     | Foundational note for this family ― the decision tree argument for the $\Omega(n \log n)$ comparison-sort lower bound[cite: 70].                         |
| [[5.1 Merge Sort\|Merge Sort]]                                                                     | Splits the array in half, recursively sorts each half, merges the two sorted halves; $O(n\log n)$[cite: 70].                                             |
| [[5.2 Quick Sort\|Quick Sort]]                                                                     | Partitions around a pivot, recursively sorts each side; $O(n\log n)$ expected, $O(n^2)$ worst case[cite: 70].                                            |

---
## Related Categories 
- [[Computer Science Introduction/Algorithms/index\|Algorithms]] 
- [[Computer Science Introduction/Algorithms/V. Dynamic Programming/index\|Dynamic Programming]] 
- [[Computer Science Introduction/Algorithms/II. Greedy Algorithms/index\|Greedy Algorithms]]