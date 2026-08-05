---
description: The k-th order statistic problem — find the k-th smallest element in an unsorted list — solved via a divide-and-conquer partition strategy that beats sort-then-index.
tags:
  - algorithm
  - divide-and-conquer
  - selection
aliases:
  - k-th Order Statistic
  - Selection Problem
---
> [!abstract]
> What if we designed an algorithm that takes as input a list of numbers of length $n$ and an integer $1 \leq k \leq n$, and outputs the $k^{th}$ smallest integer in the list?
> 
> - **Category:** Divide and Conquer / Selection (order statistics)
> - **Input:** A list of $n$ numbers, and an integer $k$ with $1 \leq k \leq n$
> - **Output:** The $k^{th}$ smallest element of the list
> - **Paradigm:** Divide and Conquer (partition-based)
> - **Typical use cases:** finding medians, general order statistics, percentile queries

---

# Problem Specification

- **Instance:** A list $L = [a_1, \dots, a_n]$ of integers, and an integer $k$ with $1 \leq k \leq n$.
- **Solution Format:** A single integer — the $k^{th}$ smallest value in $L$.
- **Constraints:** $1 \leq k \leq n$.
- **Objective / Goal:** Unlike the optimization problems elsewhere in this vault, Selection isn't a "maximize/minimize over many valid solutions" problem — there's exactly one correct answer per instance. The goal is instead to compute it **correctly and quickly**, ideally faster than the $O(n\log n)$ a full sort would cost.

---

# Candidate Strategies / Approaches

## Sort-then-Index

Sort the entire list, then return the element at index $k$. Always correct, but costs $O(n \log n)$ — more work than necessary, since we only need to identify the rank of _one_ element, not fully order all $n$ of them.

## Divide and Conquer (Partition-Based) ✔

Applying the general Divide and Conquer recipe: break into similar subproblems (split the list), solve recursively (select from one sublist), combine (decide how to split again).

Just splitting down the middle doesn't help — instead, pick a random **pivot**, and split the list into all elements smaller than the pivot and all elements larger. Then determine which side the $k^{th}$ smallest element must fall in (note that $k$ itself may need to change depending on which side we recurse into).

> [!tip] Key Idea 
> Unlike Merge Sort, Selection never needs to recurse on _both_ halves — once we know which side of the pivot the answer lives on, the other side can be discarded entirely. That's exactly what allows Selection to beat the $O(n\log n)$ sorting lower bound.

---

# Partition with Pivot (Core Subroutine)

Given a list $L = [a_1, \dots, a_n]$ and a pivot $a_i$, rearrange $L$ so that all elements smaller than $a_i$ are to the left of $a_i$ and all elements larger are to the right. This is the core operation both [[QuickSelect]] and [[Deterministic Selection]] are built on.

**Design goals:** linear time, and ideally **in place** (rearranging the list only by swapping elements, no auxiliary array).

```pseudo
	\begin{algorithm}
	\caption{Partition with Pivot}
	\begin{algorithmic}
	\Input List $L = [a_0, a_1, \dots, a_{n-1}]$
	\Output Rearranged list $L'$ such that the elements smaller than pivot are to left and elements larger than pivot are to the right
	\Procedure{Partition}{$[a_0, a_1, \dots, a_{n-1}]$}
		\State $i=0$
		\State $h = n-1$
		\While{$i \neq h$}
			\If{$i < h$}
				\If{$a_i < a_h$}
					\State $h = h-1$
				\Else
					\State swap $a_i$ and $a_h$
					\State swap $i$ and $h$
					\State $h = h+1$
				\EndIf
			\Else
				\If{$a_i \geq a_h$}
					\State $h = h+1$
				\Else
					\State swap $a_i$ and $a_h$
					\State swap $i$ and $h$
					\State $h = h-1$
                \EndIf
            \EndIf
        \EndWhile
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!note] Reading This Pseudocode 
> This partitions the list around the element that starts at index $i=0$ (i.e. $a_0$ acts as the pivot). Two pointers `i` and `h` scan toward each other from opposite ends; whenever the scanning pointer finds an element on the wrong side of the current reference value, it's swapped into place, and the roles of `i` and `h` swap — which is why the pseudocode swaps the _index variables themselves_, not just the array values. This continues until the two pointers meet, at which point the list is fully partitioned. It answers both of the classic design questions: it's **linear time** (each element is examined a bounded number of times as `i` and `h` converge), and it's **in place** (only swaps are used — no second array).

---

# Worked Example

> [!Example] `Selection([40, 31, 6, 51, 76, 58, 97, 37, 86, 31, 19, 30, 68], 7)` Pick a pivot ($31$). Divide the list into 3 groups:
> 
> - $SL$ — all elements smaller than $31$: $SL = [6, 19, 30]$, size $3$
> - $Sv$ — all elements equal to $31$: $Sv = [31, 31]$, size $2$
> - $SR$ — all elements greater than $31$: $SR = [40, 51, 76, 58, 97, 37, 86, 68]$, size $8$
> 
> Since $k=7$ is bigger than $|SL|=3$, the $k^{th}$ smallest element can't be in $SL$. Since $k=7$ is also bigger than $|SL|+|Sv| = 5$, it can't be in $Sv$ either — so it must be in $SR$.
> 
> Since $5$ elements ($SL \cup Sv$) have already been accounted for as smaller than everything in $SR$, the $7^{th}$ smallest element overall is the $(7-5) = 2^{nd}$ smallest element **within** $SR$. Recurse on $SR$ with the adjusted $k=2$.

---

# Chosen Approach

This note covers the shared problem framing and the `Partition with Pivot` subroutine both concrete algorithms rely on. The actual recursive selection algorithms — and their correctness proofs and complexity analyses — live in their own dedicated notes, since the choice of _how to pick the pivot_ is what distinguishes them:

- [[QuickSelect]] — picks the pivot **randomly**; simple, expected $O(n)$ time, but $O(n^2)$ worst case.
- [[Deterministic Selection]] — picks the pivot via a guaranteed-good strategy (median-of-medians); worst-case $O(n)$ time, at the cost of a larger constant factor.

---

# Time & Space Complexity Analysis

## Partition with Pivot (this note's subroutine)

| |Complexity|Notes|
|---|---|---|
|Time|$O(n)$|Single pass — the two pointers `i`, `h` converge, each element examined a bounded number of times|
|Space|$O(1)$|In place — only element swaps, no auxiliary array|

The complexity of the _full_ selection algorithm depends entirely on how the pivot is chosen — see [[QuickSelect]] and [[Deterministic Selection]] for those analyses.

## Deterministic vs. Randomized

| |Deterministic|Randomized|
|---|---|---|
|**Selection**|[[Deterministic Selection]] (Median of Medians) — $O(n)$|[[QuickSelect]] — Best: $O(n)$, Worst: $O(n^2)$, Average: $O(n)$|

See [[Computer Science Introduction/Algorithms/Divide and Conquer/index#Deterministic vs. Randomized Approaches|Deterministic vs. Randomized Approaches]] for the full table including [[Sorting]].

---

# Drawbacks / Constraints

- **Sort-then-Index wastes work.** Fully sorting costs $O(n\log n)$ when only one element's rank is actually needed.
- **Pivot choice matters enormously.** A poor pivot (e.g. always the min or max) barely shrinks the problem each recursive call — see [[QuickSelect]]'s worst-case analysis for exactly how bad this gets.
- **Not suitable for:** repeated queries for many different $k$ values on the same list — if you need several order statistics from the same data, sorting once ($O(n\log n)$) and then indexing repeatedly ($O(1)$ each) can beat re-running Selection ($O(n)$ each) from scratch every time.

---

# References / Links

- [[QuickSelect]]
- [[Deterministic Selection]]
- [[Computer Science Introduction/Algorithms/Divide and Conquer/index|Divide and Conquer]]