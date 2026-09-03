---
description: Randomized divide-and-conquer selection algorithm that finds the k-th smallest element in expected O(n) time by recursing into only one side of a random pivot.
tags:
  - algorithm
  - divide-and-conquer
  - selection
aliases:
  - Quick Select
---
> [!abstract] Abstract 
> QuickSelect solves the [[Selection]] problem — finding the $k^{th}$ smallest element — by picking a **random** pivot at each step and recursing into only the one side that must contain the answer.
> 
> - **Category:** Divide and Conquer / Selection (Randomized)
> - **Input:** A list of $n$ integers, and an integer $k$ with $1 \leq k \leq n$
> - **Output:** The $k^{th}$ smallest element
> - **Paradigm:** Randomized Divide and Conquer
> - **Typical use cases:** median finding, order statistics, any single "find the $k^{th}$ ranked item" query

---

# Core Logic (High-Level)

Recap of [[Selection]]'s partition-based strategy: pick a random pivot $v$, split the list into $SL$ (smaller than $v$), $Sv$ (equal to $v$), and $SR$ (larger than $v$) using [[Selection#Partition with Pivot|Partition with Pivot]]. Compare $k$ against $|SL|$ and $|SL|+|Sv|$ to determine which one group contains the $k^{th}$ smallest element, then recurse into **only that group** (adjusting $k$ if recursing into $SR$).

> [!tip] Key Idea 
> Since we only ever recurse into one side — never both — the total work depends entirely on how unbalanced that one side is. Picking the pivot **randomly** doesn't guarantee a balanced split on any single call, but on average it shrinks the problem fast enough to give linear _expected_ time overall, even though individual unlucky calls can be bad.

---

# Pseudocode (Mid-Level Implementation)

```pseudo
	\begin{algorithm}
	\caption{Quick Select}
	\begin{algorithmic}
	\Input List of integers and integer $k$
	\Output The $k^{th}$ smallest number in the set of integers
	\Procedure{QuickSelect}{$a[1 \dots n], k$}
		\If{$n=1$}
			\Return $a[1]$
        \EndIf
        \State Pick a random integer $v$ in the list
        \State Split the list into sets $SL$, $Sv$, $SR$
        \If{$k \leq |SL|$}
	        \Return QuickSelect($SL, k$)
	    \Elif{$k \leq |SL| + |Sv|$}
		    \Return $v$
		\Else
			\Return QuickSelect($SR, k - |SL| - |Sv|$)
        \EndIf
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`v`|Integer|The randomly chosen pivot for this call|
|`SL`, `Sv`, `SR`|Sublists|Elements smaller than, equal to, and greater than `v`, respectively — see [[Selection#Partition with Pivot\|Partition with Pivot]]|
|`k`|Integer|The target rank — re-adjusted (`k -|

## Helper Functions / Operations Used

- **[[Selection#Partition with Pivot|Partition with Pivot]]** — splits the list around `v` in $O(n)$ time, $O(1)$ extra space (in place).
- **Random pivot selection** — pick `v` uniformly at random from the current list; $O(1)$.

---

# Proof of Correctness

**Claim:** `QuickSelect(a, k)` returns the true $k^{th}$ smallest element of `a`.

**Proof (strong induction on $n = |a|$):**

- **Base case ($n=1$):** the only element is trivially the $1^{st}$ (and only) smallest — correct by inspection.
- **Inductive hypothesis:** assume `QuickSelect` is correct on every list of size $< n$.
- **Inductive step:** for a list of size $n$, partitioning around `v` produces $SL$, $Sv$, $SR$ such that every element of $SL$ is smaller than every element of $Sv$, which is smaller than every element of $SR$. So:
    - If $k \leq |SL|$, the $k^{th}$ smallest overall is exactly the $k^{th}$ smallest within $SL$ — correct by the inductive hypothesis, since $|SL| < n$.
    - If $|SL| < k \leq |SL|+|Sv|$, the $k^{th}$ smallest is one of the (equal-valued) elements of $Sv$, i.e. $v$ itself — returned directly, correctly.
    - If $k > |SL|+|Sv|$, the $k^{th}$ smallest overall is the $(k - |SL| - |Sv|)^{th}$ smallest within $SR$ — correct by the inductive hypothesis, since $|SR| < n$ (as $Sv$ contains at least the pivot itself, so $|SR| \leq n-1$).

**Termination:** every recursive call operates on a strictly smaller list ($|SL| < n$ or $|SR| < n$, since $Sv$ always contains at least the pivot), so the recursion depth is finite and the algorithm terminates.

---

# Time & Space Complexity Analysis

## Naive Best/Worst Case Reasoning

The runtime depends entirely on how big $|SL|$ and $|SR|$ turn out to be relative to $n$ — the recursive call costs $T(|SL|)$ or $T(|SR|)$, plus $O(n)$ for picking the pivot and partitioning.

**Lucky case:** if $v$ happens to land close to the median every time, $|SL| \approx |SR| \approx n/2$, so no matter which side we recurse on:

$$ 
T(n) = T\left(\frac{n}{2}\right) + O(n) 
$$

By the [[Master Theorem]] ($a=1, b=2, d=1$, so $a < b^d$): $T(n) = O(n)$.

**Unlucky case:** if $v$ happens to be the max or min every time, $|SL| = n-1$ (or $|SR| = n-1$), so:

$$ 
T(n) = T(n-1) + O(n) = O(n^2) 
$$

## Expected Runtime (Rigorous)

Selecting the $i^{th}$ element uniformly at random splits the list into pieces of length $(i-1)$ and $(n-i)$. Recursing on the relevant piece costs time proportional to $\max(i-1, n-i)$.

The smallest possible max-size split is at $i = n/2$:

$$ 
\max\left(\frac{n}{2}, n-\frac{n}{2}\right) = \frac{n}{2} 
$$

and the worst case is at $i=1$ or $i=n$:

$$ 
\max(1, n-1) = n-1 
$$

If $\frac{n}{4} \leq i \leq \frac{3n}{4}$ (a "good" pivot), then $\max(i-1, n-i) \leq \frac{3n}{4}$. Otherwise (a "bad" pivot), $\frac{3n}{4} < \max(i-1,n-i) < n$. Since a uniformly random pivot lands in the "good" range with probability $\geq \frac{1}{2}$, this gives an upper bound on the expected runtime:

$$ 
\begin{align*} 
ET(n) &\leq \frac{1}{2}ET\left(\frac{3n}{4}\right) + \frac{1}{2}ET(n) + O(n) \\ ET(n) &\leq ET\left(\frac{3n}{4}\right) + O(n) 
\end{align*} 
$$

Plugging into the [[Master Theorem]] with $a=1$, $b=\frac{4}{3}$, $d=1$: since $a < b^d$,

$$ ET(n) = O(n) $$

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(n)$ expected, $O(n^2)$ worst case|Randomized pivot choice makes the worst case extremely unlikely, but not impossible|
|Space|$O(1)$ auxiliary if partitioning in place; $O(\log n)$ expected recursion depth (worst case $O(n)$)|Follows directly from the same lucky/unlucky split analysis above, applied to call-stack depth instead of time|

## Best / Worst / Average Case

- **Best case:** $O(n)$ — even a single lucky partition near the median-heavy path keeps total work linear (geometric decay: $n + n/2 + n/4 + \dots = O(n)$).
- **Worst case:** $O(n^2)$ — pivot is repeatedly the min or max (e.g. an adversarially chosen or already-sorted input paired with an unlucky random draw every time).
- **Average / Expected case:** $O(n)$ — proven rigorously above via the "good pivot" probability argument.

---

# Drawbacks / Constraints

- **$O(n^2)$ worst case is real, if rare.** Randomization makes an adversary unable to force bad performance _deterministically_, but doesn't eliminate the possibility — pathologically unlucky random draws can still occur.
- **Not suitable for:** situations requiring a _guaranteed_ worst-case bound (e.g. real-time systems where a single slow call is unacceptable) — see [[Deterministic Selection]] for a pivot-selection strategy that guarantees $O(n)$ worst case, at the cost of a larger constant factor.
- **Not suitable for:** repeated queries for many different $k$ on the same list — see [[Selection#Drawbacks / Constraints|Selection's Drawbacks]] for why sorting once can be better in that case.

---

# References / Links

- [[Selection]]
- [[Deterministic Selection]]
- [[Master Theorem]]
- [[Computer Science Introduction/Algorithms/Divide and Conquer/index|Divide and Conquer]]