---
title: Divide and Conquer
description: Overview of the Divide and Conquer paradigm, the classic and Karatsuba multiplication algorithms, and the Master Theorem used to analyze their runtimes.
tags:
  - divide-and-conquer
aliases:
  - Divide and Conquer
---
> [!Note] Section Overview
> 
> - **Divide and Conquer:** break a problem into similar subproblems, solve each subproblem recursively, then combine the results.
> - Every algorithm in this section produces a recurrence of the form $T(n) = aT(n/b) + O(n^d)$, which the [[#Master Theorem]] below solves in closed form — it's the one tool that ties this whole section together.

---

# Divide and Conquer (The Paradigm)

- Break a problem into similar subproblems.
    
- Solve each subproblem recursively.
    
- Combine the subproblem results into a solution for the original problem.
    
- **Key detail:** subproblems must be _smaller instances of the same problem_ — that's what makes the recursion terminate and what makes Master Theorem-style analysis applicable in the first place.
    

---

# Multiplying n-bit Numbers (Classic Recursive Approach)

Suppose we want to multiply two $n$-bit numbers, $n$ a power of 2. Split each into left/right halves of $n/2$ bits each:

$$ 
\begin{align*}
x &= 2^{n/2}x_{L} + x_{R}\\
y &= 2^{n/2}y_{L} + y_{R}\\ \\ 
xy &= 2^{n}\boxed{x_{L}y_{L}} + 2^{n/2}(\boxed{x_{L}y_{R}} + \boxed{x_{R}y_{L}}) + \boxed{x_{R}y_{R}} 
\end{align*} 
$$

The boxed terms are recursive calls at half the size.

```pseudo
	\begin{algorithm}
	\caption{Algorithm Multiply}
	\begin{algorithmic}
	\Input $n$-bit intergers $x$ and $y$
	\Output the product $xy$
	\Procedure{multiply}{$x,y$}
		\If{$n=1$}
			\Return $xy$
        \EndIf
        \State $x_L, x_R$ and $y_L, y_R$ are the left-most and right-most $n/2$ bits of $x$ and $y$ respectively
        \State $P_1$ = multiply($x_L, y_L$)
        \State $P_2$ = multiply($x_L, y_R$)
        \State $P_3$ = multiply($x_R, y_L$)
        \State $P_4$ = multiply($x_R, y_R$)
        \Return $P_1 \times 2^n + (P_2 + P_3) \times 2^{n/2} + P_4$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

- **Time Complexity:** $T(n) = 4T(n/2) + O(n)$ (3 additions + 2 bit shifts of $O(n)$-bit integers, plus 4 recursive calls) → Master Theorem with $a=4, b=2, d=1$: since $a > b^d$ ($4 > 2$), $T(n) = O(n^{\log_2 4}) = O(n^2)$.
- **Key detail:** this is asymptotically no better than grade-school multiplication — the 4-way recursive split doesn't actually help until you reduce the number of subproblems (see Karatsuba below).

---

# Karatsuba's Algorithm (Multiply KS)

```pseudo
	\begin{algorithm}
	\caption{Multiply KS}
	\begin{algorithmic}
	\Input $n$-bit integers $x$ and $y$
	\Output the product $xy$
	\Procedure{multiplyKS}{$x,y$}
		\If{$n=1$}
			\Return $xy$
        \EndIf
        \State $x_L, x_R$ and $y_L, y_R$ are the left-most and right-most $n/2$ bits of $x$ and $y$ respectively
        \State $R_1$ = multiplyKS($x_L, y_L$)
        \State $R_2$ = multiplyKS($x_R, y_R$)
        \State $R_3$ = multiplyKS($(x_L + x_R)(y_L + y_R)$)
        \Comment{$(x_L + x_R)(y_L + y_R) = x_Ly_L + x_Ly_R + x_Ry_L + x_Ry_R$}
        \Return $R_1 \times 2^n + (R_3 - R_1 - R_2) \times 2^{n/2} + R_2$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

- **Time Complexity:** $T_{KS}(n) = 3T_{KS}(n/2) + O(n)$ → Master Theorem with $a=3, b=2, d=1$: since $a > b^d$ ($3 > 2$), $T_{KS}(n) = O(n^{\log_2 3}) \approx O(n^{1.585})$.
- **Key detail:** the trick is computing $(x_L+x_R)(y_L+y_R)$ **once** and subtracting off $R_1, R_2$ to recover the two cross terms $x_Ly_R + x_Ry_L$ — turning 4 recursive multiplications into 3, which is what drops the exponent below 2.

---

# Master Theorem

If $T(n) = aT\left(\frac{n}{b}\right) + O(n^d)$ for constants $a>0, b>1, d\geq 0$, then

$$ T(n) \in \begin{cases} O(n^{d}) &\text{if } a< b^{d}\ \ O(n^{d}\log n) &\text{if } a = b^{d}\ \ O(n^{\log_{b}a}) &\text{if } a>b^{d} \end{cases} $$

## Solving the Recurrence

After $k$ levels of recursion, there are $a^k$ subproblems, each of size $n/b^k$. Work at level $k$:

$$ O\left(\left(\frac{n}{b^{k}}\right)^{d}\right) a^{k} = \boxed{O\left(n^{d}\left(\frac{a}{b^{d}}\right)^{k}\right)} $$

After $\log_b n$ levels, subproblem size shrinks to 1 (the base case), so the total is the sum over all levels:

$$ T(n) = O\left(n^{d}\sum_{k=0}^{\log_{b} n}\left(\frac{a}{b^{d}}\right)^{k}\right) $$

This is a **geometric series** with ratio $r = \dfrac{a}{b^d}$.

## Proof (Three Cases)

**Case 1 — $a < b^d$ ($r < 1$):** the series converges to a constant, so $T(n) = O(n^d)$.

**Case 2 — $a = b^d$ ($r = 1$):** every term equals 1, so the sum is just the number of terms:

$$ T(n) = O\left(n^{d}\sum_{k=0}^{\log_{b}n} 1^{k}\right) = O(n^{d}\log_{b}n) $$

**Case 3 — $a > b^d$ ($r > 1$):** the sum is exponential and grows proportional to its last term:

$$ T(n) = O\left(n^{d}\left(\frac{a}{b^{d}}\right)^{\log_{b} n}\right) = O(n^{\log_{b}a}) $$

> [!Info] Recall $$\sum_{k=0}^{n} r^{k} = \frac{r^{n+1}-1}{r-1} = O(r^{n})$$

---
# Deterministic vs. Randomized Approaches
**Sorting** and **Selection** each have a deterministic and a randomized solution, trading a worse worst-case bound for a simpler algorithm and better typical-case performance:

|               | Deterministic                                            | Randomized                                                                    |
| ------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Sorting**   | [[Merge Sort]] — $O(n\log n)$                            | [[Quick Sort]] — Best: $O(n\log n)$, Worst: $O(n^{2})$, Average: $O(n\log n)$ |
| **Selection** | [[Deterministic Selection]] (Median of Medians) — $O(n)$ | [[QuickSelect]] — Best: $O(n)$, Worst: $O(n^{2})$, Average: $O(n)$            |

---

# Quick Reference Table

|Algorithm|Recurrence|Master Theorem Case|Closed-Form Runtime|
|---|---|---|---|
|Classic Multiply|$T(n) = 4T(n/2) + O(n)$|$a>b^d$ ($4 > 2$)|$O(n^2)$|
|Karatsuba (Multiply KS)|$T(n) = 3T(n/2) + O(n)$|$a>b^d$ ($3 > 2$)|$O(n^{\log_2 3}) \approx O(n^{1.585})$|

---

# Notes in This Section

| Note                        | One-line description                                                                                                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [[Binary Search]]           | Halves the search space each comparison on a sorted array; $O(\log n)$                                                                                                                                  |
| [[Cook-Toom-k algorithm]]   | Generalizes Karatsuba's trick to split into $k$ parts instead of 2, trading extra combine-step overhead for fewer recursive multiplications                                                             |
| [[Deterministic Selection]] | Median-of-medians (BFPRT) — splits into groups of 5 to construct a provably-good pivot, guaranteeing $O(n)$ worst case without randomization                                                            |
| [[Sorting]]                 | Foundational note for this family ― the decision tree argument for the $\Omega(n \log n)$ comparison-sort lower bound                                                                                   |
| [[Merge Sort]]              | Splits the array in half, recursively sorts each half, merges the two sorted halves; $O(n\log n)$                                                                                                       |
| [[Quick Sort]]              | Partitions around a pivot, recursively sorts each side; $O(n\log n)$ expected, $O(n^2)$ worst case                                                                                                      |
| [[QuickSelect]]             | Quick Sort-style partitioning used to find the $k^{th}$ smallest element directly, via a random pivot; $O(n)$ expected, $O(n^{2})$ worst case                                                           |
| [[Selection]]               | The general "find the $k^{th}$ smallest element" problem, plus the shared in-place `Partition with Pivot` subroutine; [[QuickSelect]] and [[Deterministic Selection]] are the two algorithms solving it |
| [[Two Runners]]             | Binary search for the "turning point" where a slower-starting runner overtakes a faster one — a discrete analogue of the Intermediate Value Theorem; $O(\log n)$                                        |

---

# References / Links

- [[Binary Search]]
- [[Sorting]]
- [[Merge Sort]]
- [[Quick Sort]]
- [[QuickSelect]]
- [[Deterministic Selection]]
- [[Selection]]
- [[Cook-Toom-k algorithm]]
- [[Two Runners]]