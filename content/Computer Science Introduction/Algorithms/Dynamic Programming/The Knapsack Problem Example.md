---
aliases:
  - Knapsack Problem
  - Knapsack
  - The Knapsack Problem
description: Maximize total value of items packed into a weight-limited knapsack, where each item can be selected any number of times — solved via a 2D dynamic programming table in O(nC) time.
tags:
  - algorithm
  - Examples
  - dynamic-programming
---
> [!abstract]
> Suppose you are a burglar who breaks into a store and want to leave with the maximum value of items. Your knapsack can only hold 13 lbs, and the items in the store are:
> 
> ||||||||
> |---|---|---|---|---|---|---|
> |Value|4|9|12|15|19|21|
> |Weight|2|4|5|7|8|9|
> 
> What is the maximum value you can carry, given a list of items $a[1]\dots a[n]$ where each item has value $v[i]$ and weight $w[i]$, and total weight can't exceed capacity $C$?
> 
> - **Category:** Dynamic Programming / Combinatorial Optimization
> - **Input:** Items $1 \dots n$ with values $v[i]$ and weights $w[i]$; capacity $C$
> - **Output:** Maximum total value achievable without exceeding capacity $C$
> - **Paradigm:** Backtracking (naive) → Dynamic Programming
> - **Typical use cases:** resource allocation under a budget/capacity constraint; the classic example distinguishing _unbounded_ (items reusable) from _0/1_ (each item used at most once) variants

> [!note] This Is the Unbounded Variant 
> In the backtracking pseudocode below, including item $n$ recurses on `BTKS(w[1...n], v[1...n], C - w[n])` — **the full item list, including $n$ again** — rather than `w[1...n-1]`. That's what makes this **Unbounded Knapsack**: an item can be picked more than once. The classic 0/1 Knapsack (each item used at most once) would instead recurse on `w[1...n-1]` in the "include" branch.

---

# Problem Specification

- **Instance:** Items $1, \dots, n$, each with a value $v[i]$ and weight $w[i]$; a capacity $C$.
- **Solution Format:** A multiset of items (repeats allowed) to carry.
- **Constraints:** Total weight of chosen items $\leq C$.
- **Objective:** $\sum v[i]$ over chosen items (with repetition).
- **Goal:** Maximize.

---

# Candidate Strategies / Approaches

## Backtracking ✘

```pseudo
	\begin{algorithm}
	\caption{Knapsack Problem}
	\begin{algorithmic}
	\Procedure{BTKS}{$w[1 \dots n], v[1 \dots n], C$}
		\If{$C = 0$ or $n=0$}
			\Return $0$
		\EndIf
		\If{$w[n] > C$}
			\Return $BTKS(w[1 \dots n-1], v[1 \dots n-1], C)$
        \EndIf
		\State In = $v(n) + BTKS(w[1\dots n], v[1 \dots n], C - w[n])$
		\State Out = $BTKS(w[1 \dots n-1], v[1 \dots n-1], C)$
		\Return $\max($In$, $Out$)$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!note] Runtime
>  Unlike most of this vault's backtracking examples, $n$ doesn't strictly decrease on every call (the `In` branch keeps the same item list) — only $C$ strictly decreases there (by at least $\min_i w[i]$ each time), while `Out` strictly decreases $n$. So recursion depth is bounded by roughly $\frac{n+C}{\underset{ i }{ \min } w[i]}$, but the branching at every level still makes this exponential in the worst case — no better than exhaustive search, same story as [[Weighted Event Scheduling Example|Weighted Event Scheduling]]'s `BTWES`.

## Dynamic Programming ✔

Replace the recursive calls with an array value: let $KS(j, b)$ be the maximum value you can fit in a $b$-capacity knapsack using only items $1 \dots j$.

---

# Dynamic Programming Solution

## 1. Define Subproblems

Let $KS(j, b)$ be the maximum value you can fit in a $b$-capacity knapsack using only items $1 \dots j$.

## 2. Base Cases

$$ 
KS(j, 0) = 0 \quad \forall\ 0 \leq j \leq n \qquad\qquad KS(0, b) = 0 \quad \forall\ 0\leq b\leq C 
$$

(No capacity, or no items available, both trivially cap value at 0.)

## 3. Recursion Used to Fill the Array

- **Out:** item $j$ is not included $\implies KS(j, b) = KS(j-1, b)$.
- **In:** item $j$ is included $\implies KS(j, b) = v(j) + KS(j, b - w(j))$ — note this stays on row $j$, allowing item $j$ to be reused.

Since we don't know which is bigger, compute both and take the max:

$$ KS(j,b) = \max\big(KS(j-1, b),\ v(j) + KS(j, b-w(j))\big) \quad \text{if } b \geq w(j) $$

(if $b < w(j)$, item $j$ doesn't fit, so $KS(j,b) = KS(j-1,b)$).

## 4. Ordering of the Subproblems

Cell $[j,b]$ depends on $[j, b-w(j)]$ (same row, to its left) and $[j-1, b]$ (row above, same column).

![[Pasted image 20260711140344.png]]

So the problems can be ordered by filling each row left to right, starting from the top row and working down:

```
for j = 1 ... n
    for b = 1 ... C
```

## 5. Final Output

$$ KS(n, C) $$

## 6. Runtime

$$ O(nC) $$

One cell per $(j,b)$ pair, $O(1)$ work each.

---

# Worked Example

**Given:**

|Value|Weight|
|---|---|
|4|2|
|9|4|
|12|5|
|15|7|
|19|8|
|21|9|

**Completed Solution Table:**

|$v[i]$|$w[i]$|0|1|2|3|4|5|6|7|8|9|10|11|12|13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|$\emptyset$|$\emptyset$|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|4|2|0|0|4|4|8|8|12|12|16|16|20|20|24|24|
|9|4|0|0|4|4|9|9|13|13|18|18|22|22|27|27|
|12|5|0|0|4|4|9|12|13|16|18|21|24|25|28|30|
|15|7|0|0|4|4|9|12|13|16|18|21|24|25|28|30|
|19|8|0|0|4|4|9|12|13|16|19|21|24|25|28|31|
|21|9|0|0|4|4|9|12|13|16|19|21|24|25|28|31|

**Reading a cell — row 3 ($v=12, w=5$), $c=13$:**

$$ K[3][13] = \max\big(\underbrace{K[2][13]}_{27},\ \ 12 + \underbrace{K[3][8]}_{18}\big) = \max(27, 30) = 30 $$

> [!tip] Patterns Worth Noticing
> 
> - **Row 4 (item $15/7$) is identical to Row 3.** Its value-per-weight ratio ($15/7 \approx 2.14$) is worse than item 3's ($12/5 = 2.4$), so it never wins a comparison — adding a row to the table doesn't guarantee the answers change.
> - **Item 3 has the best value/weight ratio of all six** (2.4, vs. the next-best 2.375 for item 5/8). For unbounded knapsack, as capacity grows large, the optimal strategy converges toward "just take the best-ratio item repeatedly" — part of why item 3's influence dominates the later columns.

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`KS`|2D array, $(n+1) \times (C+1)$|`KS[j][b]` = max value using items $1\dots j$ with capacity $b$|
|`j`|Row index|Which prefix of items is currently allowed|
|`b`|Column index|Remaining capacity being considered|

## Helper Functions / Operations Used

- **Table lookup** — $O(1)$ per cell, reading `KS[j-1][b]` and `KS[j][b-w(j)]`.

---

# Proof of Correctness / Optimality

**Claim:** $KS(j,b)$ equals the true maximum value achievable using only items $1,\dots,j$ within capacity $b$.

- **Base cases:** $KS(j,0)=0$ (no capacity, nothing fits) and $KS(0,b)=0$ (no items available) are both correct by inspection — the empty selection is the only option, with value 0.
- **Inductive Hypothesis:** every cell computed before $(j,b)$ in the row-by-row, left-to-right order — i.e. $KS(j-1, \cdot)$ and $KS(j, b')$ for $b' < b$ — is correct.
- **Inductive Step:** any valid selection using items $1,\dots,j$ within capacity $b$ either uses item $j$ at least once or doesn't:
    - **Doesn't use item $j$:** the best such selection is exactly the best selection using only items $1,\dots,j-1$, i.e. $KS(j-1,b)$ — correct by the Inductive Hypothesis.
    - **Uses item $j$ (at least once):** taking one copy of item $j$ leaves capacity $b - w(j)$, still allowing item $j$ to be reused, so the best such selection is $v(j) + KS(j, b-w(j))$ — correct by the Inductive Hypothesis, since $b - w(j) < b$.
- Since $KS(j,b) = \max$ of these two cases, and every valid selection falls into exactly one of them, $KS(j,b)$ is the true maximum. 

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(nC)$|One $O(1)$ computation per cell, $(n+1)\times(C+1)$ cells total|
|Space|$O(nC)$|The full table, though this can be reduced to $O(C)$ by only keeping the current and previous row if the item selection itself doesn't need to be reconstructed|

## Best / Worst / Average Case

- **Best / Worst / Average case:** all $O(nC)$ — every cell is filled regardless of the specific values/weights involved.

> [!Important] Pseudo-Polynomial Runtime 
> $O(nC)$ _looks_ polynomial, but it's polynomial in the **value** of $C$, not in the size of its binary representation. Since $C$ only takes $O(\log C)$ bits to write down, this runtime is exponential in the actual input size when $C$ is large — this is the standard example of a **pseudo-polynomial** algorithm, and it's exactly why Knapsack is still NP-hard in general despite this DP solution existing.

---

# Drawbacks / Constraints

- **Pseudo-polynomial time** — see the callout above; this DP approach becomes impractical when $C$ is astronomically large relative to $n$, even though the table-filling logic itself is simple.
- **Unbounded vs. 0/1 matters.** This solution assumes items are reusable. For the 0/1 variant (each item usable at most once), the "In" recursion must reference $KS(j-1, b-w(j))$ instead of $KS(j, b-w(j))$ — forgetting this distinction silently solves the wrong problem.
- **Space can be reduced** if only the optimal _value_ is needed (not which items were chosen) — see the Space row above.

---

# References / Links

- [[Computer Science Introduction/Algorithms/Dynamic Programming/index|Dynamic Programming]]
- [[Weighted Event Scheduling Example|Weighted Event Scheduling]]
- [[Computer Science Introduction/Algorithms/Backtracking/index|Backtracking]]