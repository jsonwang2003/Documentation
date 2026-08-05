---
description: Each event has a value; maximize total value of a non-overlapping subset. Backtracking alone is exponential — the motivating example for Dynamic Programming via memoization.
tags:
  - algorithm
  - dynamic-programming
  - Examples
aliases:
  - Weighted Interval Scheduling
  - Weighted Activity Selection
  - Weighted Event Scheduling
---

> [!abstract] 
>  Like [[Event Scheduling]], but now every event has a **value** — instead of maximizing the _count_ of non-overlapping events, maximize their total _value_.
> 
> ![[Pasted image 20260711122739.png]]
> 
> - **Category:** Dynamic Programming / Interval Scheduling (weighted variant)
> - **Input:** A list of events, each with a start time, finish time, and value
> - **Output:** The maximum total value achievable from a non-overlapping subset
> - **Paradigm:** Backtracking (naive) → Dynamic Programming (via memoization)
> - **Typical use cases:** resource allocation where jobs differ in priority/payoff, not just presence — plain [[Event Scheduling]]'s "maximize count" greedy strategy no longer works once events carry different values

---

# Problem Specification

- **Instance:** $[(s_1, f_1, v_1), \dots, (s_n, f_n, v_n)]$ — start time, finish time, and value for each event.
- **Solution:** A subset of events $S$.
- **Constraints:** No two events in $S$ overlap.
- **Objective:** $\sum_{i \in S} v_i$.
- **Goal:** Maximize the sum.

> [!Question] Why the Old Greedy Strategy Fails
>  [[Event Scheduling]]'s Earliest-End-Time greedy strategy is optimal for maximizing _count_, but says nothing about _value_ — a short, low-value event finishing early can easily block out a much more valuable event that would have overlapped it. Greedy has no way to "look ahead" and weigh that trade-off, which is exactly why this problem needs a different approach.

---

# Candidate Strategies / Approaches

## Backtracking ✘ (exponential)

**Strategy:** sort events by end time. Consider the last event to end, $I_n$ — including it isn't necessarily good, so try both possibilities:

1. **Exclude $I_n$:** recurse on $[I_1, \dots, I_{n-1}]$.
2. **Include $I_n$:** recurse on the set of all intervals that do **not** conflict with $I_n$ — more precisely, $[I_1, \dots, I_k]$ where $I_k$ is the last event to end _before_ $I_n$ starts.

```pseudo
	\begin{algorithm}
	\caption{Weighted Event Scheduling}
	\begin{algorithmic}
	\Input list of events $[I_1, \dots, I_n]$
	\Output list of events that does not overlap each other and maximizes the value
	\Procedure{BTWES}{$I_1, \dots, I_n$}
		\If{$n=0$}
			\Return $0$
        \EndIf
        \If{$n=1$}
	        \Return $value(I_1)$
        \EndIf
        \State Out = $BTWES(I_1, \dots, I_{n-1})$
        \State Let $I_k$ be the last event to end before $I_n$ starts
        \State In = $BTWES(I_1, \dots, I_k) + value(I_n)$
        \Return $\max($Out$, $In$)$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

`Out` costs $T(n-1)$; `In` costs $T(k)$.

**Runtime (worst case):**

$$ T(n) = 2T(n-1) + O(n) = O(2^n) $$

No better than exhaustive search.

### The Key Insight: How Many _Distinct_ Calls Are There?

We make up to $2^n$ recursive calls — but every recursive call has the form $BTWES(I_1, \dots, I_k)$ for some $k$. So there are at most $n+1$ genuinely **different** calls that ever occur: ${\emptyset, (I_1), (I_1,I_2), \dots, (I_1,\dots,I_n)}$.

> [!tip] Key Idea 
> Of the up to $2^n$ recursive calls this algorithm makes, only $n+1$ are actually distinct — the exact same subproblems are being solved over and over along different branches. **Memoization** — storing and reusing each distinct answer, e.g. in a hashmap or array — is the fix. This is the seed of the full [[Computer Science Introduction/Algorithms/Dynamic Programming/index|Dynamic Programming]] solution below.

## Dynamic Programming ✔

Instead of top-down recursion with a cache, solve the same $n+1$ distinct subproblems **bottom-up**, smallest first, filling in an array directly.

---

# Dynamic Programming Solution (The 8 Steps)

## 1. Define Sub-Problems and Corresponding Array

> [!hint] 
> The sub-problems are often restatements of the original problem.

- **Original Problem:** find the max value among all valid schedules of $(I_1, \dots, I_n)$.
- **Sub-Problem:** let $A[k]$ be the max value among all valid schedules of $(I_1, \dots, I_k)$.

## 2. What Are the Base Cases?

$$ A[0] = 0 $$

True for any input — the empty schedule has value 0.

## 3. Give Recursion for Sub-Problems (Case Analysis)

> [!hint] 
> Break up the sub-problem into distinct cases.

$A[k] = \dots$

- **Case 1:** $I_k$ is not part of the max-value schedule $\implies A[k] = A[k-1]$.
- **Case 2:** $I_k$ is part of the max-value schedule $\implies A[k] = v(k) + A[j-1]$, where $j-1$ is the last event before event $k$ starts.

$$ A[k] = \max(A[k-1],\ v(k) + A[j-1]) $$

(with $k-1 < k$ and $j-1 < k$, so both terms reference strictly smaller sub-problems.)

## 4. Order the Sub-Problems

Since each sub-problem depends only on sub-problems of strictly smaller index, order them from $0$ up to $n$.

## 5. What Is the Final Output?

$$ A[n] = \text{max value of non-conflicting subsets} $$

## 6. Put It All Together: Iterative Algorithm

```pseudo
	\begin{algorithm}
	\caption{Max Subset}
	\begin{algorithmic}
	\Procedure{MaxSubset}{$[I_1, \dots, I_n], [v(I_1), \dots , v(I_n)]$}
		\State $A[0] = 0$
		\Comment{Step 2}
		\For{$k = 1 \dots n$}
		\Comment{Step 4}
			\State $j = 1$
			\Comment{Step 3 start}
			\While{$End(I_j) \leq Start(I_k)$}
				\State $j = j+1$
            \EndWhile
            \State In = $v(I_k) + A[j-1]$
            \State Out = $A[k-1]$
            \State $A[k] = \max($In$, $Out$)$
            \Comment{Step 3 end}
        \EndFor
        \Return $A[n]$
        \Comment{Step 5}
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`A`|Array, size $n+1$|`A[k]` holds the max value achievable using only events $I_1, \dots, I_k$|
|`j`|Index|Found via linear scan — the last event that finishes before $I_k$ starts|
|`In`, `Out`|Values|The two candidate values for `A[k]` — including or excluding $I_k$|

### Helper Functions / Operations Used

- **Find $j$ (last non-conflicting event before $I_k$)** — linear scan in the pseudocode above, $O(n)$ worst case per call to the outer loop; see [[#Drawbacks / Constraints]] for a faster alternative.

---
# Proof of Correctness / Optimality

**Claim:** $A[k]$ is the max value out of all valid schedules of $I_1, \dots, I_k$.

- **Base case:** $A[0] = 0$ (Step 2).
- **Inductive Hypothesis:** $A[k]$ is set correctly for all $0 \leq k < n$, for some $n > 0$.
- **Inductive Step (Step 3):** consider $A[n]$.
    - **Case 1 — $I_n$ is not in the max-value schedule:** the best schedule using only $I_1,\dots,I_n$ is then just the best schedule using $I_1,\dots,I_{n-1}$, so $A[n] = A[n-1]$ — correct by the Inductive Hypothesis.
    - **Case 2 — $I_n$ is in the max-value schedule:** every event that conflicts with $I_n$ (i.e. $I_j, \dots, I_{n-1}$) must be excluded, so the best schedule is $I_n$'s value plus the best schedule using only the non-conflicting events $I_1, \dots, I_{j-1}$: $A[n] = v[n] + A[j-1]$ — correct by the Inductive Hypothesis, since $j - 1 < n$.
- Since every valid schedule falls into exactly one of these two cases, and $A[n] = \max(\text{Case 1}, \text{Case 2})$, $A[n]$ is the true maximum. $\blacksquare$

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(n^2)$|Outer loop runs $n$ times; each iteration's `while` loop scanning for $j$ costs $O(n)$ worst case, giving $O(n) \times O(n) = O(n^2)$|
|Space|$O(n)$|The array `A` holds one entry per event|

## Best / Worst / Average Case

- **Best / Worst / Average case:** all $O(n^2)$ with this implementation — the linear scan for $j$ runs regardless of how the events happen to be arranged.

---

# Drawbacks / Constraints

- **The $O(n^2)$ bound isn't tight to the DP idea itself** — it comes from the linear scan used to find $j$ inside the loop. If events are pre-sorted by finish time (already assumed here) and you additionally binary-search for $j$ against the sorted start times, this drops to $O(n\log n)$ total — the same order as sorting the events in the first place.
- **Doesn't handle changing values/weights dynamically** — like most DP solutions, this assumes the full input (including all values) is known up front; recomputation is needed if values change after the array is filled.
- **Not suitable for:** finding the _count_-maximizing schedule when all values happen to be equal — [[Event Scheduling]]'s simpler $O(n\log n)$ greedy strategy is a better fit for that special case, since it doesn't need the full $A[k]$ array at all.

---

# References / Links

- [[Computer Science Introduction/Algorithms/Dynamic Programming/index|Dynamic Programming]]
- [[Event Scheduling]]
- [[Computer Science Introduction/Algorithms/Backtracking/index|Backtracking]]