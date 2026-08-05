---
aliases:
  - Interval Scheduling
  - Activity Selection
tags:
  - algorithm
  - greedy-algorithms
  - Greedy
  - Examples
description: Greedy interval scheduling algorithm that maximizes the number of non-overlapping events by always picking the next event with the earliest finish time.
---
> [!abstract] Abstract 
> Suppose you are running a conference and you have a collection of events (or talks) that each have a start time and an end time. However, there is only one conference room available. **Goal:** schedule the most events possible that day such that no two events overlap.
> 
> - **Category:** Greedy Algorithm / Interval Scheduling
> - **Input:** A collection of events, each with a start and finish time
> - **Output:** A subset of events, none overlapping
> - **Paradigm:** Greedy (Earliest End Time strategy)
> - **Typical use cases:** room/resource booking, single-machine job scheduling, any "pick the max number of non-conflicting intervals" problem

---

# Core Logic (High-Level)

## Specification

- **Instance:** collection of events ${ (s_{1}, f_{1}), (s_{2}, f_{2}), \dots, (s_{n}, f_{n}) }$
- **Solution Format:** subset of events
- **Constraints:** no 2 events in the subset overlap
- **Objective:** cardinality of the subset ($\in \mathbb{R}$)
- **Goal:** maximize

## Strategies to Solve

Before solving and proving a greedy algorithm, we should first find a greedy strategy to solve. Several candidate greedy strategies:

- **Shortest duration ✘** — Counter Example: Solution ${A}$, Better Solution ${B, C}$. ![[Pasted image 20260705210200.png]]
- **Earliest start time ✘** — Counter Example: Solution ${A}$, Better Solution ${B,C,D}$. ![[Pasted image 20260705213655.png]]
- **Fewest conflicts ✘** — Solution: 3, Better Solution: 4. ![[Pasted image 20260705213748.png]]
- **Earliest end time ✔** — Solution: 4 (optimal). ![[Pasted image 20260705213915.png]]

> [!tip] Key Idea 
> Picking the event that **finishes soonest** leaves the maximum possible remaining time for everything else — it's the choice that constrains the future the least. That's exactly why the other three strategies fail: shortest duration, earliest start, and fewest conflicts can all still "use up" time that a later, more useful event needed.

---

# Pseudocode (Mid-Level Implementation)

```pseudo
	\begin{algorithm}
	\caption{Event Scheduling Implementation}
	\begin{algorithmic}
	\Procedure{EventScheduling}{}
		\State Initialize a Queue $S$
		\State Sort the intervals by finish time
		\State Put the first event $E_1$ in $S$
		\State Set $F = f_1$
		\For{$i = 2 \dots n$}
			\If{$s_i \geq F$}
				\State enqueue($E_i, S$)
				\State $F = f_i$
            \EndIf
        \EndFor
        \Return $S$
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`S`|Queue|Accumulates the chosen (non-overlapping) events, in order|
|`F`|Number|The finish time of the most recently accepted event — the earliest time the room becomes free next|
|`E_i`|Event $(s_i, f_i)$|The event currently being considered, in sorted order|

## Helper Functions / Operations Used

- **Sort by finish time** — a one-time $O(n \log n)$ pass that makes the rest of the algorithm a single linear scan
- **`enqueue(E_i, S)`** — accepts event $i$ into the schedule; $O(1)$

---
# Proof of Correctness

We need to show: for every instance $I$, letting $GS$ be the greedy algorithm's solution to $I$ and $OS$ be _any other_ solution for $I$,
$$
	\underbrace{ Value(OS) \leq Value(GS) }_{ \text{Maximize} }
$$

(since this is a maximization problem — cardinality of the chosen subset).

> [!Important] The Tricky Part 
> $OS$ is an _arbitrary_ solution, not one that makes sense to reason about directly — we don't know much about it. This is what makes greedy optimality proofs harder than they first look, and why general techniques exist rather than ad hoc arguments each time.

Two of the three general techniques (see [[Techniques to Prove Optimality]] for the complete overview of all three) apply cleanly here, giving two independent proofs of the same result.

## Proof via Exchange Argument

Let $E={E_{1}​, \dots, E_{n}​}$ be the set of all events, with $(s_{i}, f_{i})$ the start and finish times of $E_{i}$​. Let $G$ be the event with the earliest finish time — the first greedy decision (include $G$). Let $OS$ be any non-overlapping schedule that does **not** include $G$.

**Claim:** there is a schedule $OS′$ that does include $G$ such that $|OS'| \geq |OS|$.

**Proof:** let the events in $OS$ be $J_1, J_2, \dots, J_k$, ordered by start and finish time ($J_1 \neq G$).

![[Pasted image 20260706104423.png]]

Define $OS′$ from $OS$:

$$
OS' = (OS - \{J_1\}) \cup \{G\}
$$

**$OS'$ is valid (no overlapping events):** since $OS$ is valid, no pair $J_i, J_\ell$​ overlaps — so it's enough to show $G$ doesn't overlap $J_2$ (the event right after $J_1$​). Since $OS$ is valid, $f_{J_1} < s_{J_2}$​​. And since $G$ is defined as the event with the earliest finish time overall, $f_G \leq f_{J_1} < s_{J_2}$​​. So $G$ doesn't overlap $J_2$​, and $OS'$ is valid.

**$|OS'| \geq |OS|$:** $OS'$ removes exactly one event ($J_1$​) from $OS$ and adds exactly one ($G$) back in, so $|OS'| = |OS| - 1 + 1 = |OS|$ — $OS'$ is always at least as good (equal count), never worse.

**Induction:** this Exchange Argument claim alone doesn't prove optimality — it needs an inductive argument on top. Prove by strong induction on nn n, the number of events:

- **Base Case ($n=1$):** any choice works, including the greedy choice.
- **Inductive Hypothesis:** suppose $n>1$, and the greedy algorithm is optimal for any $k$ events, $1 \leq k \leq n-1$ — i.e. for $|I| < n$, $|OS(I)| \leq |GS(I)|$ for any solution $OS(I)$.
- **Inductive Step:** let $OS$ be any solution on $I = \{ E_{1}, \dots, E_{n} \}$. By the Exchange Argument above, there's a solution $OS'$ with $|OS| \leq |OS'|$ that includes the first greedy choice $G$. Let $I'$ be the events that don't conflict with $G$, so $OS' = \{ G \} \cup S(I')$ for some solution $S(I')$ of $I'$. Since $|I'| < n$, the inductive hypothesis gives $|S(I')| \leq |GS(I')|$. By definition, $GS(I) = \{ G \} \cup GS(I')$. Putting it together:

$$
	|OS(I')| \leq |OS'(I)| = |\{ G \} \cup S(I')| \leq |\{ G \} \cup GS(I')| = |GS(I)|
$$

So the greedy algorithm is optimal for any $n$.

## Proof via Greedy Stays Ahead

Consider input $I$ with $n$ events. Let $OS(I) = [J_{1}, \dots, J_{k}]$ be an arbitrary set of non-conflicting events (in order), and let $GS(I) = [G_{1}, \dots, G_{\ell}]$ be the greedy strategy's output. **Want to show:** $k\leq \ell$, i.e. $|GS(I)| \geq |OS(I)|$.

![[Pasted image 20260706120832.png]]

Compare a **progress measure** — when the $i^{th}$ event finishes.

**Claim:** $GS$ "stays ahead" of $OS$: $Finish(G_{1}) \leq Finish(J_{1})$ for all $i\geq 1$.

**Proof (induction on $i$):**

- **Base Case:** $Finish(G_{1}) \leq Finish(J_{1})$ by the greedy choice (earliest finish time overall).
- **Inductive Hypothesis:** for some $i\geq 1$, assume $Finish(G_{i}) \leq Finish(J_{i})$.
- **Inductive Step:** want $Finish(G_{i+1}) \leq Finish(J_{i+1})$. Among all events that start after $G_{i}$ finishes, $G_{i+1}$​ is chosen to be the one that ends earliest. Since $J_{i+1}$ starts after $J_{i}$ finishes (validity of $OS$), and $Finish(G_{i}) \leq Finish(J_{i+1}) \leq Start(J_{i+1})$ (inductive hypothesis, then validity of $OS$), $J_{i+1}$ is a candidate the greedy strategy could have picked at step $i+1$ — and since greedy picks the earliest-finishing candidate, $Finish(G_{i+1}) \leq Finish(J_{i+1})$.

**Using this to prove $k \leq \ell$ (by contradiction):** suppose $k > \ell$ (so $OS$ has more events than $GS$). Then $G_{\ell}$ is the last greedy choice, so no event starts after $G_{\ell}$​ finishes. By the Claim, $Finish(G_{\ell}) \leq Finish(J_{\ell})$, and by validity of $OS$, $Finish(J_{\ell}) \leq Start(J_{\ell + 1})$. This implies there's an event, $J_{\ell + 1}$​ that starts after the last greedy choice finishes — but greedy would have picked it, contradicting that $G_{\ell}$​ was the last choice. So $k \leq \ell$, meaning $|GS(I)| \geq |OS(I)|$.

Both proofs conclude the same thing — Earliest End Time is optimal — via genuinely different routes: Exchange inducts on shrinking the _problem size_, while Greedy Stays Ahead inducts on the _sequence of choices_ itself. See [[Techniques to Prove Optimality]] for how these two techniques generalize beyond this example.

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(n \log n)$|Dominated by sorting the $n$ events by finish time; the single linear pass afterward is $O(n)$|
|Space|$O(n)$|Storage for the sorted event list plus the output queue $S$|

## Best / Worst / Average Case

- **Best / Worst / Average case:** All $O(n \log n)$ — sorting must happen regardless of how many events end up overlapping, and the linear scan afterward always touches every event once.

---

# Drawbacks / Constraints

- **The correct greedy criterion isn't obvious.** Three plausible-looking strategies (shortest duration, earliest start time, fewest conflicts) all fail — see the counterexamples above — which is exactly the general warning in [[Computer Science Introduction/Algorithms/Greedy Algorithms/index|Greedy Algorithms]]: a greedy algorithm needs a proof, not just intuition, before you can trust it.
- **Not suitable for:** weighted interval scheduling (maximizing total _value_ of chosen events rather than just their _count_) — earliest-end-time is no longer guaranteed optimal once events have different weights; that variant requires Dynamic Programming instead.
- **Preconditions:** assumes a single resource (one conference room); scheduling across multiple identical rooms is a related but different problem (interval partitioning).

---

# References / Links

- [[The Greedy Method]]
- [[Techniques to Prove Optimality]]
- [[Prove Kruskal's with Exchange Argument]]
- [[Computer Science Introduction/Algorithms/Greedy Algorithms/index|Greedy Algorithms]]