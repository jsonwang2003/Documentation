---
description: Blum-Floyd-Pratt-Rivest-Tarjan's median-of-medians algorithm — a deterministic divide-and-conquer selection algorithm guaranteeing worst-case O(n) time.
tags:
  - algorithm
  - divide-and-conquer
  - selection
aliases:
  - Median of Medians
  - BFPRT
---
> [!abstract]
> For theoretical computer scientists, it is unsatisfactory to only have a randomized algorithm ([[QuickSelect]]) that could run in quadratic time. Blum, Floyd, Pratt, Rivest, and Tarjan developed a **deterministic** approach to finding the median (or any $k^{th}$ smallest element), guaranteeing worst-case linear time.
> 
> - **Category:** Divide and Conquer / Selection (Deterministic)
> - **Input:** A list $L$, and an integer $k$
> - **Output:** The $k^{th}$ [smallest] element of $L$
> - **Paradigm:** Deterministic Divide and Conquer
> - **Typical use cases:** any selection scenario needing a _guaranteed_ worst-case bound rather than an expected one — adversarial inputs, hard real-time systems, or as a worst-case-safe pivot-picker inside other algorithms (e.g. a guaranteed-$O(n\log n)$ Quick Sort)

---

# Core Logic (High-Level)

[[QuickSelect]]'s randomized pivot works well _on average_, but a run of bad luck can still cost $O(n^2)$. The fix: instead of hoping for a decent pivot, **construct** one that's provably decent every time.

1. Split the list into groups of 5.
2. Find the median of each group (sort each tiny group of 5, or recursively call `MofM(S[i], 3)`).
3. Recursively find the median **of those group-medians** — this is the "median of medians," $M$, used as the pivot.
4. Partition $L$ around $M$ into $SL$, $SM$ (equal to $M$), $SR$ — exactly like [[Selection#Partition with Pivot|Partition with Pivot]] in **QuickSelect**.
5. Recurse into whichever side contains rank $k$ (adjusting $k$ as needed), just like **QuickSelect**.

> [!tip] Key Idea 
> Because $M$ is the median of $n/5$ group-medians, at least half of those groups have a median $\leq M$ — and for each such group, at least 3 of its 5 elements (the median and the two below it) are also $\leq M$. That guarantees roughly $\frac{3n}{10}$ elements are provably $\leq M$ (and symmetrically, $\frac{3n}{10}$ are provably $\geq M$) — no matter how the input is arranged. This turns "hope for a lucky pivot" into "guarantee a decent one," at the cost of doing extra work to compute it.

---

# Pseudocode (Mid-Level Implementation)

```pseudo
	\begin{algorithm}
	\caption{Median of Medians}
	\begin{algorithmic}
	\Input $L$ list of elements
	\Input $k$ the $k^{th}$ smallest element to find
	\Output the $k^th$ element
	\Procedure{MofM}{$L, k$}
		\If{$L$ has $10$ or fewer elements}
			\State Sort($L$)
			\Return $k^{th}$ element
        \EndIf
        \State Partition $L$ into sublists $S[i]$ of five elements each
        \For{$i = 1, \dots, \frac{n}{5}$}
	        \State $m[i] = MofM(S[i], 3)$
        \EndFor
        \State $M = MofM([m[1], \dots, m[\frac{n}{5}]], \frac{n}{10})$
        \State Split the list into sets $SL$, $SM$, $SR$
        \If{$k \leq |SL|$}
	        \Return $MofM(SL, k)$
        \EndIf
        \If{$k \leq |SL| + |Sv|$}
	        \Return $v$
	    \Else
		    \Return $MofM(SR, k - |SL| - |Sv|)$
        \EndIf
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!note] Reading This Pseudocode 
> The split step defines $SM$ (elements equal to the pivot $M$), but the branches below it reference `Sv` and `v` — these are presumably meant to be `SM` and `M` respectively (likely copied over from the **QuickSelect** pseudocode without renaming). Treat `Sv` = `SM` and `v` = `M` when reading this.

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`S[i]`|Sublists of 5|The list partitioned into groups of 5 elements each|
|`m[i]`|Array of group medians|The median (3rd of 5) of each group `S[i]`, found via a recursive call|
|`M`|Element|The "median of medians" — the recursively-found median of the `m[i]` array; used as the pivot|
|`SL`, `SM`, `SR`|Sublists|Elements smaller than, equal to, and greater than `M`, from partitioning $L$ around it|

## Helper Functions / Operations Used

- **Sort (base case only)** — sorting a list of $\leq 10$ elements; $O(1)$ since the size is bounded by a constant.
- **`MofM(S[i], 3)`** — recursively finds the median of a 5-element group by treating "median of 5" as its own selection instance.
- **Partition around `M`** — same in-place partitioning idea as [[Selection#Partition with Pivot|Partition with Pivot]].

---

# Proof of Correctness

The recursive correctness argument (base case + correctly identifying which of $SL$/$SM$/$SR$ contains rank $k$, adjusting $k$ appropriately) mirrors [[QuickSelect#Proof of Correctness|QuickSelect's proof]] exactly, since the partition-and-recurse structure is identical — the only difference is _how the pivot is chosen_. The genuinely new thing to prove here is that the chosen pivot is always good enough:

**Claim:** $|SL| \leq \frac{7n}{10}$ and $|SR| \leq \frac{7n}{10}$.

**Proof:** Consider the $n/5$ group-medians. Since $M$ is their median, at least half of them — $\frac{n}{10}$ groups — have a group-median $\leq M$. For each such group, since its median is $\leq M$, at least 3 of its 5 elements (the median itself, plus the two elements below it in that group) are also $\leq M$. So at least $\frac{n}{10} \times 3 = \frac{3n}{10}$ elements of $L$ are provably $\leq M$ (ignoring at most one partial leftover group, which only affects the bound by a bounded constant).

That means at most $n - \frac{3n}{10} = \frac{7n}{10}$ elements can be $> M$, so $|SR| \leq \frac{7n}{10}$. The symmetric argument (using the $n/10$ groups with median $\geq M$) gives $|SL| \leq \frac{7n}{10}$. 

---

# Time & Space Complexity Analysis

## The Recurrence

By construction, $|SR| \leq \frac{7n}{10}$ and $|SL| \leq \frac{7n}{10}$ (proven above), so no matter which side we recurse on:

$$ T(n) = T\left(\frac{n}{5}\right) + T\left(\frac{7n}{10}\right) + O(n) $$

— the $T(n/5)$ term from finding the median of medians, the $T(7n/10)$ term from the worst-case recursive selection call, and $O(n)$ for partitioning and the group-median computations.

> [!Important] You **cannot** use the [[Master Theorem]] here — it only applies to a single recursive term of the form $aT(n/b)$, not a sum of two differently-sized recursive calls like this. Instead, this is solved directly by induction.

## Proof by Induction: $T(n) = O(n)$

**Claim:** there exists a constant $c$ such that $T(n) \leq cn$ for all $n$.

- **Base case:** for $n \leq 10$, the algorithm just sorts directly, so $T(n) = O(1) \leq cn$ for any $c$ large enough to dominate that constant.
- **Inductive hypothesis:** suppose $T(m) \leq cm$ for all $m < n$.
- **Inductive step:** let $an$ (for some constant $a$) bound the $O(n)$ non-recursive work (partitioning, computing group medians, etc.) at this level. Then:

$$ 
\begin{align*}
T(n) &= T\left(\frac{n}{5}\right) + T\left(\frac{7n}{10}\right) + an \\
&\leq c\cdot\frac{n}{5} + c\cdot\frac{7n}{10} + an \qquad \text{(by the inductive hypothesis, since }
\frac{n}{5}, \frac{7n}{10} < n \text{)}\\ 
&= c\left(\frac{2n}{10} + \frac{7n}{10}\right) + an \\
&= \frac{9cn}{10} + an 
\end{align*} 
$$

We want this $\leq cn$:

$$ \frac{9cn}{10} + an \leq cn \iff an \leq \frac{cn}{10} \iff c \geq 10a $$

So choosing $c = 10a$ (or any constant at least that large, and large enough to also cover the base case) makes the induction go through for every $n$. Therefore:

$$ T(n) = O(n) $$

**Deterministic Selection runs in worst-case linear time.**

> [!note] Why groups of 5, specifically? 
> The group size isn't arbitrary. With groups of size 5, the two recursive fractions sum to $\frac{1}{5} + \frac{7}{10} = \frac{9}{10} < 1$ — strictly less than 1 is what makes the induction above work (the $an$ term has "room" to be absorbed). Groups of 3 would instead give a recurrence like $T(n) = T(n/3) + T(2n/3) + O(n)$, where the fractions sum to _exactly_ 1 — that no longer converges to linear time (it degrades toward $O(n\log n)$ instead), since there's no leftover fraction to absorb the $O(n)$ work at each level.

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(n)$ worst case|Proven via the induction above — no randomness needed, unlike [[QuickSelect]]|
|Space|$O(n)$|Auxiliary storage for the groups of 5 and the array of group-medians at each level of recursion|

## Best / Worst / Average Case

- **Best / Worst / Average case:** all $O(n)$ — this is the entire point of the algorithm. Unlike **QuickSelect**, there's no input arrangement or unlucky randomness that can push this above linear time.

---

# Drawbacks / Constraints

- **Larger constant factor than QuickSelect.** Sorting every group of 5, recursively finding the median of medians, and doing two recursive-sized calls worth of bookkeeping per level adds substantially more overhead per element than **QuickSelect**'s simple random pivot pick — in practice, **QuickSelect** is usually faster on typical (non-adversarial) inputs despite its worse worst case.
- **More complex to implement correctly** — the nested recursive structure (recursing both to find group medians _and_ to find the median of medians _and_ to recurse into $SL$/$SR$) is easy to get subtly wrong compared to **QuickSelect**'s single recursive call.
- **Not suitable for:** everyday use where average-case performance is what matters — reach for [[QuickSelect]] unless you specifically need a worst-case guarantee.

---

# References / Links

- [[Selection]]
- [[QuickSelect]]
- [[Master Theorem]]
- [[Computer Science Introduction/Algorithms/Divide and Conquer/index|Divide and Conquer]]