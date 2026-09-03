---
aliases:
  - Two-Runners
tags:
  - algorithm
  - searching
  - divide-and-conquer
description: Binary search for the 'turning point' where a slower-starting runner overtakes a faster one — a discrete analogue of the Intermediate Value Theorem.
---
> [!abstract] Problem Statement 
> Two runners are racing to a finish line at position $T$. Runner A starts at position 0; runner B has a head start at position $X$ (e.g. $X = 10$). It is given that A wins the race in $n$ seconds. You are given each runner's position at every second of the race, as two arrays: $$A[0] = 0,\ A[1],\ A[2], \dots, A[n] = T$$ $$B[0] = X,\ B[1],\ B[2], \dots, B[n] < T$$ Find an index $j$ where A "passes" B — that is, $A[j] \le B[j]$ and $A[j+1] > B[j+1]$.
> 
> - **Category:** Binary Search
> - **Source:** Algorithm design exercise
> - **Difficulty:** Medium

---

# Problem Specification

- **Input:** Two arrays $A[0 \dots n]$ and $B[0 \dots n]$ such that $A[0] = 0$, $A[0] \le B[0]$, and $A[n] > B[n]$.
- **Output:** An index $j$ such that $A[j] \le B[j]$ and $A[j+1] > B[j+1]$ (the "turning point").
- **Constraints:** $0 \le j < n$; positions are given for every integer second $0$ through $n$.
- **Assumptions:** A starts behind or tied with B ($A[0] \le B[0]$) and finishes strictly ahead ($A[n] > B[n]$), so a turning point is guaranteed to exist (discrete Intermediate Value Theorem).
- **Edge cases to consider:** $n = 1$ (only one possible turning point, $j = 0$); A passes B exactly once vs. multiple times (algorithm only guarantees finding _one_ valid $j$, not necessarily the first).

---

# Core Logic / Strategy / Approach

1. Maintain two pointers `lo` and `hi` such that the invariant $A[lo] \le B[lo]$ and $A[hi] > B[hi]$ always holds.
2. At each step, probe the midpoint `m` between `lo` and `hi`.
3. If `m` itself is the turning point ($A[m] \le B[m]$ and $A[m+1] > B[m+1]$), return it immediately.
4. Otherwise, use the sign of the comparison at `m` (and `m+1`) to discard half the search space — shrinking the range `[lo, hi]` while preserving the invariant.
5. Repeat until `lo` and `hi` are adjacent (`lo + 1 = hi`), at which point they must be the turning point by the invariant.

> [!tip] Key Idea 
> The condition "$A \le B$" transitions to "$A > B$" exactly once as we sweep left to right in the sense that matters here: at any index `lo` with $A[lo] \le B[lo]$ and any index `hi` with $A[hi] > B[hi]$, there is guaranteed to be a turning point somewhere in $[lo, hi]$. This lets us binary search on the _comparison sign_ the same way we'd binary search on a monotonic predicate, even though $A$ and $B$ themselves need not be monotonic.

---

# Solution in Pseudocode

```pseudo
	\begin{algorithm}
	\caption{Two Runners}
	\begin{algorithmic}
	\Input Two lists $A[0 \dots n]$ and $B[0 \dots n]$ such that $A[0] = 0$, $A[0] \leq B[0]$ and $A[n] > B[n]$
	\Output Index $j$ such that $A[j] \leq B[j]$ and $A[j+1] > B[j+1]$
	\Procedure{TwoRunners}{$A[0 \dots n], B[0 \dots n]$}
		\State $lo = 0$
		\State $hi = n$
		\While{$lo + 1 < hi$}
			\State $m = \lfloor \frac{lo + hi}{2} \rfloor$
			\If{$A[m] \leq B[m]$ and $A[m+1] > B[m+1]$}
				\Return $m$
            \EndIf
            \If{$A[m] > B[m]$}
	            \State $hi = m$
            \EndIf
            \If{$A[m + 1] \leq B[m+1]$}
	            \State $lo = m + 1$
            \EndIf
        \EndWhile
        \Return $lo$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`lo`|Index (integer)|Left boundary of search range; always satisfies $A[lo] \le B[lo]$|
|`hi`|Index (integer)|Right boundary of search range; always satisfies $A[hi] > B[hi]$|
|`m`|Index (integer)|Midpoint probe, $m = \lfloor (lo+hi)/2 \rfloor$|

## Helper Functions / Operations Used

- **Array indexing `A[i]`, `B[i]`** — $O(1)$ random access into the given arrays.
- **Comparison `A[i] \le B[i]`** — $O(1)$ per check.
- No auxiliary data structures are needed; the algorithm operates entirely on the two input arrays with two integer pointers.

---

# Proof of Correctness

**Claim:** Upon termination, the algorithm returns an index $j$ such that $A[j] \le B[j]$ and $A[j+1] > B[j+1]$.

**Loop Invariant:** After every iteration (and before the loop begins), $A[lo] \le B[lo]$ and $A[hi] > B[hi]$.

- **Initialization:** Before the loop, $lo = 0$ and $hi = n$. By the given parameters, $A[0] \le B[0]$ and $A[n] > B[n]$, so the invariant holds trivially at the start.
- **Maintenance:** Suppose the invariant holds before an iteration, i.e. $A[lo] \le B[lo]$ and $A[hi] > B[hi]$. Consider the midpoint $m$:
    - If $A[m] \le B[m]$ and $A[m+1] > B[m+1]$, the algorithm has found the turning point directly and terminates, returning $m$ — correctness holds immediately.
    - If $A[m] > B[m]$, the algorithm sets $hi = m$. By the inductive hypothesis $A[lo] \le B[lo]$ is unaffected, and the new $hi$ satisfies $A[hi] > B[hi]$ since $A[m] > B[m]$ by assumption. Invariant preserved.
    - If $A[m+1] \le B[m+1]$, the algorithm sets $lo = m+1$. By the inductive hypothesis $A[hi] > B[hi]$ is unaffected, and the new $lo$ satisfies $A[lo] \le B[lo]$ since $A[m+1] \le B[m+1]$ by assumption. Invariant preserved.
    - These two update branches aren't mutually exclusive, but at least one always fires whenever the direct-return condition fails: failing to return means NOT($A[m] \le B[m]$ **and** $A[m+1] > B[m+1]$), which by De Morgan's means either $A[m] > B[m]$ or $A[m+1] \le B[m+1]$ holds — guaranteeing at least one branch executes and the range strictly shrinks.
- **Termination:** The loop condition is $lo + 1 < hi$, and each iteration either returns directly or strictly shrinks the range $[lo, hi]$ (since $lo$ moves up to $m+1 > lo$ or $hi$ moves down to $m < hi$). So the loop must eventually reach $lo + 1 = hi$ and exit. At that point, by the invariant, $A[lo] \le B[lo]$ and $A[lo+1] = A[hi] > B[hi] = B[lo+1]$ — exactly the turning-point condition — so returning $lo$ is correct.

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(\log n)$|Each iteration does $O(1)$ work and halves the range $[lo, hi]$: $T(n) = T(n/2) + O(1) \Rightarrow T(n) \in O(\log n)$|
|Space|$O(1)$|Only a constant number of index variables (`lo`, `hi`, `m`) are used beyond the input arrays|

## Implementation-Dependent Variations

|Data Structure Choice|Impact on Time|Impact on Space|Notes|
|---|---|---|---|
|Arrays (given)|$O(1)$ random access per comparison|$O(1)$ auxiliary|Assumed representation; enables the $O(\log n)$ binary search|
|Linked lists instead of arrays|$O(n)$ to reach `m` per iteration|$O(1)$ auxiliary|Would degrade total time to $O(n \log n)$ or worse — binary search needs random access|
|Linear scan (brute force) alternative|$O(n)$|$O(1)$|Simpler but asymptotically much slower for large $n$|

## Best / Worst / Average Case

- **Best case:** $O(1)$ — the very first midpoint checked happens to be the turning point.
- **Worst case:** $O(\log n)$ — the search range must be halved all the way down to a single adjacent pair `(lo, hi)`.
- **Average case:** $O(\log n)$ — binary search's halving behavior means the average case matches the worst case asymptotically.

---

# Drawbacks / Constraints

- **Preconditions:** Requires $A[0] \le B[0]$ and $A[n] > B[n]$ to guarantee a turning point exists; requires random-access (array-like) input.
- **Fails / degrades when:** The "at least one boundary condition holds" property between comparisons doesn't extend to guaranteeing a _unique_ turning point — if A passes and re-passes B multiple times, the algorithm returns _some_ valid turning point, not necessarily the first or last.
- **Not suitable for:** Finding _all_ turning points (would need a full scan, $O(n)$) or finding a specific one (e.g., "first" or "last") without additional constraints.
- **Alternatives to consider:** A linear scan trivially finds the first turning point in $O(n)$ if that specific guarantee is required.

---

# References / Links

- Discrete analogue of the Intermediate Value Theorem
- [[Computer Science Introduction/Algorithms/Divide and Conquer/Binary Search]]
- [[Computer Science Introduction/Algorithms/Divide and Conquer/index|Divide and Conquer]]