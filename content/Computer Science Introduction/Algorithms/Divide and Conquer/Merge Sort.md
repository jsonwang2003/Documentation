---
title: "Merge Sort" 
description: "Deterministic divide-and-conquer sort that recursively sorts each half of the array and merges the results in O(n log n) worst-case time." 
tags:
- CS/algorithms
- CS/divide-and-conquer
- CS/sorting 
aliases: ["MergeSort"]
---
> [!abstract] Abstract Merge Sort splits the array in half, recursively sorts each half, then merges the two sorted halves back together — the canonical Divide and Conquer sorting algorithm.
> 
> - **Category:** Divide and Conquer / Sorting (Deterministic)
> - **Input:** An array $a[1 \dots n]$
> - **Output:** The array, sorted
> - **Paradigm:** Divide and Conquer
> - **Typical use cases:** general-purpose stable sorting; external/merge-based sorting of data too large to fit in memory; the go-to when a _guaranteed_ $O(n\log n)$ worst case matters more than average-case speed

---

# Core Logic (High-Level)

1. **Divide:** split the array into two halves.
2. **Conquer:** recursively sort each half.
3. **Combine:** merge the two now-sorted halves into one sorted array.

> [!tip] Key Idea 
> All the real work happens in the **merge** step, not the split — splitting an array in half is trivial, but merging two already-sorted lists into one sorted list can be done in linear time by repeatedly comparing the fronts of each list and taking the smaller one. That single linear-time combine step, applied at every level of the recursion, is what gives the whole algorithm its $O(n\log n)$ bound (see [[Sorting]] for why $\Omega(n\log n)$ is also the best any comparison sort can do).

---

# Pseudocode (Mid-Level Implementation)

```pseudo
	\begin{algorithm}
	\caption{Merge Sort}
	\begin{algorithmic}
	\Input array to be sorted
	\Output sorted array
	\Procedure{mergesort}{$a[1 \dots n]$}
		\If{$n > 1$}
			\State $ML = mergesort(a[1 \dots \lfloor \frac{n}{2} \rfloor])$
			\State $MR = mergesort(a[\lfloor \frac{n}{2} + 1, \dots n])$
			\Return $merge(ML, MR)$
		\Else
			\Return $a$
        \EndIf
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`a[1...n]`|Array|The input to be sorted|
|`ML`, `MR`|Sorted arrays|The recursively-sorted left and right halves|

## Helper Functions / Operations Used

- **`merge(ML, MR)`** — combines two already-sorted lists into one sorted list. Repeatedly compares the fronts of `ML` and `MR`, appending whichever is smaller and advancing that list's pointer, until one list is exhausted, then appends the rest of the other. Runs in $O(|ML| + |MR|)$ time.

---

# Proof of Correctness

**Base case:** $n=1$ — `mergesort` returns the original single-element array `a`, which is trivially sorted.

**Inductive Hypothesis:** suppose that for some $n > 1$, `mergesort(a[1...k])` correctly outputs the elements of `a` in sorted order for all inputs of size $k$ where $1 \leq k < n$. We want to show it works for inputs of size $n$.

**Inductive Step:** since $n > 1$, `mergesort(a[1...n])` returns `merge(ML, MR)` where $ML = mergesort(a[1, \dots, \lfloor n/2 \rfloor])$ and $MR = mergesort(a[\lfloor n/2 \rfloor + 1, \dots, n])$. Since $\lfloor n/2 \rfloor < n$ (and the size of the second half is also $< n$), the Inductive Hypothesis ensures both $ML$ and $MR$ are sorted. And `merge` correctly combines two sorted lists into one sorted list, so the algorithm returns the elements of `a` in sorted order. 

---

# Time & Space Complexity Analysis

## General Case

Suppose `mergesort` runs in $T(n)$ time for inputs of length $n$. Each recursive call runs in $T(n/2)$ time, and `merge` runs in $O(k + \ell)$ time where $k, \ell = n/2$, so `merge` runs in $O(n)$ time:

$$ 
\begin{align*} 
T(n) &= 2T\left(\frac{n}{2}\right) + O(n) \\
&= \boxed{O(n\log n)} 
\end{align*} 
$$

By the [[Master Theorem]] ($a=2, b=2, d=1$, so $a = b^d$ — Case 2): $T(n) = O(n^d \log n) = O(n\log n)$.

| |Complexity|Notes|
|---|---|---|
|Time|$O(n\log n)$ — worst, best, and average case are all the same|The split is always exactly in half regardless of input, so there's no "unlucky" input the way there is for [[Quick Sort]]|
|Space|$O(n)$ auxiliary|`merge` needs extra space to hold the merged output before it can overwrite the original array positions|

## Best / Worst / Average Case

- **Best / Worst / Average case:** all $\Theta(n\log n)$ — Merge Sort's split is data-independent (always exactly in half), so unlike Quick Sort, there's no input arrangement that makes it faster or slower.

---

# Drawbacks / Constraints

- **Not in-place.** Requires $O(n)$ auxiliary space for the merge step, unlike [[Quick Sort]]'s partition, which can be done with $O(1)$ extra space (see [[Selection#Partition with Pivot|Partition with Pivot]]).
- **Not adaptive.** Runs in $\Theta(n\log n)$ even on already-sorted input — algorithms like Insertion Sort can detect and exploit partial sortedness to run faster on nearly-sorted data, but Merge Sort always does the same amount of work.
- **Slower in practice than Quick Sort, often**, despite the better worst-case guarantee — Quick Sort's in-place partitioning tends to have better cache locality and a smaller constant factor, so Merge Sort is usually chosen specifically _for_ its guaranteed worst case, not for raw speed.
- **Alternatives to consider:** [[Quick Sort]] when average-case speed matters more than worst-case guarantees; Insertion Sort for small or nearly-sorted inputs (often used as the base case inside a real Merge Sort implementation once the array is small enough).

---

# References / Links

- [[Sorting]]
- [[Quick Sort]]
- [[Master Theorem]]