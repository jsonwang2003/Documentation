---
description: Randomized divide-and-conquer sort that partitions around a random pivot and recursively sorts both sides — O(n log n) expected, O(n^2) worst case.
tags:
  - algorithm
  - divide-and-conquer
  - sorting
aliases:
  - QuickSort
---
> [!abstract]
> Quick Sort picks a random pivot, partitions the list around it, and recursively sorts both sides — the randomized counterpart to [[Computer Science Introduction/Algorithms/Divide and Conquer/Merge Sort]]'s deterministic split.
> 
> - **Category:** Divide and Conquer / Sorting (Randomized)
> - **Input:** A list $a_1, \dots, a_n$
> - **Output:** The list, sorted
> - **Paradigm:** Randomized Divide and Conquer
> - **Typical use cases:** general-purpose in-place sorting; often faster in practice than Merge Sort due to good cache locality and a smaller constant factor, despite a worse worst-case bound

---

# Core Logic (High-Level)

1. Pick a random index $i$ and treat $a_i$ as the pivot.
2. Partition the list into $SL$ (smaller than the pivot), $Sv$ (equal to the pivot), $SR$ (larger than the pivot).
3. Recursively sort $SL$ and $SR$.
4. Concatenate: sorted-$SL$ $\circ$ $Sv$ $\circ$ sorted-$SR$.

> [!tip] Key Idea 
> This is the same partition idea as [[QuickSelect]] and [[Selection#Partition with Pivot|Partition with Pivot]] — but where QuickSelect only ever recurses into _one_ side (since it just needs one rank), Quick Sort must recurse into **both** sides, since every element needs to end up in its correct position, not just the one at rank $k$.

---

# Pseudocode (Mid-Level Implementation)

```pseudo
	\begin{algorithm}
	\caption{Quick Sort}
	\begin{algorithmic}
	\Input list to be sorted
	\Output sorted list
	\Procedure{quickSort}{$a_1, \dots, a_n$}
		\If{$n==1$}
			\Return $a_1$
        \EndIf
        \State Pick a random index $1 \leq i \leq n$
        \State Partition the list into $SL, Sv, SR$ based on $a_i$
        \State $L = QuickSort(SL)$
        \State $R = QuickSort(SR)$
        \Return $L \circ Sv \circ R$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`i`|Random index|Selects the pivot $a_i$ for this call|
|`SL`, `Sv`, `SR`|Sublists|Elements smaller than, equal to, and greater than the pivot|
|`L`, `R`|Sorted sublists|The recursively-sorted versions of $SL$ and $SR$|

## Helper Functions / Operations Used

- **Partition around `a_i`** — same idea as [[Selection#Partition with Pivot|Partition with Pivot]]; can be done in place in $O(n)$ time.
- **`∘` (concatenation)** — joins the three pieces back into one list; $O(n)$.

---

# Proof of Correctness

> [!note] The argument below is added, following the same shape as [[Computer Science Introduction/Algorithms/Divide and Conquer/Merge Sort#Proof of Correctness|Merge Sort's proof]].

**Base case:** $n=1$ — the single element is trivially sorted, returned directly.

**Inductive Hypothesis:** suppose `quickSort` correctly sorts every list of size $< n$.

**Inductive Step:** for a list of size $n$, partitioning around $a_i$ guarantees every element of $SL$ is $\leq$ every element of $Sv$, which is $\leq$ every element of $SR$ (by construction — that's what "partition around the pivot" means). Since $|SL| < n$ and $|SR| < n$ (as $Sv$ contains at least the pivot itself), the Inductive Hypothesis guarantees $L$ and $R$ are correctly sorted versions of $SL$ and $SR$. Concatenating $L \circ Sv \circ R$ then produces a fully sorted list, since each piece is internally sorted and the three pieces are already correctly ordered relative to each other. $\blacksquare$

---

# Time & Space Complexity Analysis

## Expected Runtime

$$ 
\begin{align*} 
ET(n) &= \frac{1}{n} \left[ \sum_{i=1}^{n} ET(i-1) + ET(n-i) \right] + O(n)\\
&= \frac{2}{n}\left[ \sum_{i=1}^{n} ET(i-1) \right] + O(n)\\
&= \boxed{O(n\log n)} 
\end{align*} 
$$

This averages over every possible pivot rank $i$ (each equally likely, since the pivot is chosen uniformly at random): if the pivot lands at rank $i$, the two recursive calls cost $ET(i-1)$ and $ET(n-i)$ respectively, plus $O(n)$ for partitioning.

> [!Important] 
> Like [[Deterministic Selection]]'s recurrence, this can't be solved with the [[Master Theorem]] directly — it's a **full-history recurrence** (it depends on _every_ smaller subproblem size, not just $n/2$ or a fixed fraction of $n$). The standard way to close this: guess $ET(n) \leq an\ln n$ for a suitable constant $a$, substitute the guess back into the sum, bound $\sum_{k=1}^{n-1} k\ln k$ using the integral $\int_1^n x\ln x,dx = \frac{n^2}{2}\ln n - \frac{n^2}{4}$, and verify the resulting expression is $\leq an\ln n$ for large enough $a$ — completing the induction and confirming $ET(n) = O(n\log n)$.

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(n\log n)$ expected, $O(n^2)$ worst case|Randomized pivot choice makes the worst case unlikely but not impossible|
|Space|$O(\log n)$ expected recursion depth (worst case $O(n)$)|Partitioning itself can be done in place ($O(1)$ auxiliary), so space is dominated by the call stack|

## Best / Worst / Average Case

- **Best case:** $O(n\log n)$ — pivot happens to land near the median every time, giving balanced splits (same shape as Merge Sort's recursion).
- **Worst case:** $O(n^2)$ — pivot is repeatedly the min or max (e.g. an already-sorted list paired with unlucky random draws, or a poorly-implemented deterministic pivot rule that an adversary can exploit).
- **Average case:** $O(n\log n)$ — proven via the expected-runtime derivation above.

---

# Drawbacks / Constraints

- **$O(n^2)$ worst case**, unlike Merge Sort's guaranteed $\Theta(n\log n)$ — see [[Computer Science Introduction/Algorithms/Divide and Conquer/Merge Sort]] when a worst-case guarantee matters more than average speed.
- **Not stable** — the partitioning step can reorder equal elements relative to each other, unlike Merge Sort's `merge` step, which naturally preserves relative order.
- **Deterministic pivot rules are riskier** — always picking, say, the first element as pivot makes the worst case _predictable_ and exploitable (e.g. by an already-sorted or reverse-sorted input); randomization exists specifically to prevent an adversary from reliably triggering the $O(n^2)$ case.
- **Alternatives to consider:** [[Computer Science Introduction/Algorithms/Divide and Conquer/Merge Sort]] for a guaranteed worst case; Insertion Sort for small subarrays (often used as a cutoff inside real Quick Sort implementations, same as with Merge Sort).

---

# References / Links

- [[Sorting]]
- [[Computer Science Introduction/Algorithms/Divide and Conquer/Merge Sort]]
- [[Selection]]
- [[QuickSelect]]