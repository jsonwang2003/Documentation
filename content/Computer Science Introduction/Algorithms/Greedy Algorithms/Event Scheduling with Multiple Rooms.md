---
description: Greedy algorithm that assigns n events to the minimum number of rooms such that no two overlapping events share a room.
tags:
  - algorithm
  - Greedy
  - greedy-algorithms
  - Examples
aliases:
  - Minimum Room Scheduling
  - Interval Partitioning
  - Interval Graph Coloring
---
> [!abstract] Abstract 
> Suppose you have a conference to plan with $n$ events and an unlimited supply of rooms. How can you assign events to rooms in such a way as to **minimize** the number of rooms used?
> 
> - **Category:** Greedy / Interval Scheduling variant
> - **Input:** Start and end times of $n$ events
> - **Output:** An assignment of each event to a room
> - **Paradigm:** Greedy
> - **Typical use cases:** room-booking systems, any "schedule onto the fewest identical resources" problem (interval graph coloring)

---

# Problem Specification

- **Instance:** Start and end times of $n$ events.
- **Solution Format:** An assignment of each event to a room.
- **Constraints:** No two events that overlap are assigned to the same room.
- **Objective:** The total number of rooms used.
- **Goal:** Minimize the number of rooms.

---

# Candidate Strategies / Approaches

## Strategy 1 ✘

1. Run [[Event Scheduling]] to find the max set of non-overlapping events.
2. Assign these events to room 1.
3. Repeat until all events are assigned (room 2, room 3, ...).

> [!note] 
> Conceptually, repeatedly maximizing the non-overlapping set _per room_ doesn't guarantee the fewest total rooms across the whole schedule — locally optimizing one room at a time isn't the same as globally minimizing room count.

## Strategy 2 ✔

1. Number each room from $1$ to $n$.
2. Sort the events by earliest start time: $(E_1, \dots, E_n)$.
3. Assign the first event to room 1.
4. For events $(E_2, \dots, E_n)$, assign each event to the **lowest-numbered room available**.

![[Pasted image 20260706131822.png]]

> [!tip] Key Idea 
> The number of rooms needed is exactly the **depth** of the schedule — the maximum number of events overlapping at any single point in time. Processing events in order of start time and always taking the lowest-numbered free room guarantees the algorithm never opens more rooms than that depth requires.

---

# Pseudocode (Chosen Approach)

> [!note] 
> Following the same greedy structure described in Strategy 2. (The Proof of Correctness further below, however, is sourced directly from the lecture's Achieves-the-Bound derivation.)

```pseudo
	\begin{algorithm}
	\caption{Interval Partitioning}
	\begin{algorithmic}
		\Procedure{IntervalPartitioning}{$[(s_1,f_1), \dots, (s_n,f_n)]$}
			\State Sort events by start time $s_i$
			\State $rooms = []$ \Comment{each entry stores the end time of the last event assigned to that room}
			\For{each event $(s_i, f_i)$ in sorted order}
				\If{there exists a room $r$ with end time $\leq s_i$}
					\State Assign $(s_i, f_i)$ to the lowest-numbered such room $r$
					\State Update room $r$'s end time to $f_i$
				\Else
					\State Open a new room, assign $(s_i, f_i)$ to it, set its end time to $f_i$
				\EndIf
			\EndFor
			\Return room assignment, $|rooms|$
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`rooms`|Array (or min-heap) of end times|Tracks the end time of the most recently assigned event in each open room|
|`(s_i, f_i)`|Event|The event currently being placed, in sorted start-time order|

## Helper Functions / Operations Used

- **Sort by start time** — one-time $O(n \log n)$ pass.
- **Find lowest-numbered available room** — check whether any room's stored end time is $\leq s_i$; naively $O(k)$ per event (scanning all $k$ currently open rooms), or $O(\log k)$ using a min-heap keyed by room end time.

---

# Proof of Correctness / Optimality

We need to show: for every instance $I$, letting $GS$ be the greedy algorithm's solution and $OS$ be _any other_ solution, $Cost(GS) \leq Cost(OS)$ (minimizing room count).

> [!Important] The Tricky Part 
> $OS$ is an arbitrary valid room assignment — we don't know its structure. Directly comparing room-by-room against it isn't feasible, so this is proven instead using the **Greedy Achieves the Bound** technique (see [[Techniques to Prove Optimality]]): find a bound that any solution must respect, then show the greedy algorithm reaches it exactly.

Let $t$ be a certain time during the conference, and let $B(t)$ be the set of all events happening at time $t$ (how _busy_ the conference is at that moment). Let $R$ be the number of rooms used in an arbitrary valid schedule.

**Claim:** $R \geq |B(t)|$ for all $t$ — the total number of rooms must be able to accommodate the conference at every point in time.

**Proof idea:** for any time $t$, you need at least $|B(t)|$ rooms, since all events in $B(t)$ overlap and must each be in a different room.

**Setting up the bound:** let $L = \max_t |B(t)|$. Then $L$ is a lower bound on the number of rooms needed by _any_ solution — i.e. $R \geq L$.

**Greedy achieves this bound:** let $k$ be the number of rooms the greedy strategy uses.

**Claim:** at some point $t$, $|B(t)| = k$.

**Proof:** let $t$ be the start time of the first event scheduled into room $k$. Room $k$ was the minimum-numbered room _available_ at that time, which means at time $t$ there were already events going on in rooms $1, 2, \dots, k-1$, plus the new event now in room $k$. So $|B(t)| = k$ at this point $t$.

$$
k = |B(t_g)| \leq \max_{t} |B(\bar{t})| = L \leq R \implies k \leq L 
$$

Therefore, at some point in time $t$, $k = |B(t)| \leq \max_t |B(t)| = L$ — greedy achieves the bound $L$ exactly.

**Conclusion:** let $GS$ be the greedy solution with $k = Cost(GS)$ rooms, and $OS$ be any schedule with $R = Cost(OS)$ rooms. By the bounding lemma, $R \geq L$. By the achieves-the-bound lemma, $k \leq L$. Putting the two together:

$$
Cost(GS) = k \leq L \leq R = Cost(OS) 
$$

Thus the greedy solution (Strategy 2: sort by start time, assign to lowest-numbered available room) is optimal.

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(n \log n)$|Dominated by sorting; each event's room lookup adds $O(\log n)$ with a heap-based `rooms` structure|
|Space|$O(n)$|One entry per open room (at most $n$), plus the sorted event list|

## Implementation-Dependent Variations

|Data Structure Choice|Impact on Time|Notes|
|---|---|---|
|`rooms` as a min-heap keyed by end time|$O(n \log n)$ total|Peek the minimum end time in $O(1)$, update in $O(\log n)$ — efficiently finds _a_ free room, though not necessarily the lowest-_numbered_ one without extra bookkeeping|
|`rooms` as a plain array, linear scan per event|$O(n^2)$ worst case|Simpler to implement "lowest-numbered room" literally, but scanning all open rooms per event is $O(n)$ each|

## Best / Worst / Average Case

- **Best / Worst / Average case:** All $O(n \log n)$ with a heap-based implementation — sorting must happen regardless of overlap structure, and every event requires one room lookup/update.

---

# Drawbacks / Constraints

- **Preconditions:** assumes an _unlimited_ supply of rooms — this problem is about minimizing count, not fitting into a fixed number. If rooms are capped, this becomes a feasibility/rejection problem instead.
- **Not suitable for:** variants where rooms have different costs or capacities — this algorithm only minimizes the _count_ of identical rooms, not any weighted notion of cost.
- **Only tells you room count and assignment** — it doesn't optimize for anything else (e.g. balancing how full each room is, minimizing room-switching for attendees).

---

# References / Links

- [[Event Scheduling]]
- [[Techniques to Prove Optimality]]
- [[Computer Science Introduction/Algorithms/Greedy Algorithms/index|Greedy Algorithms]]