---
description: Three general techniques for proving a greedy algorithm optimal — Exchange Argument, Greedy Stays Ahead, and Achieves the Bound.
tags:
  - summary
  - greedy-algorithms
  - proofs
aliases:
  - Proving Greedy Optimality
---

> [!Note] Section Overview
> 
> - Every technique below tackles the same core obstacle: an arbitrary "other solution" $OS$ is hard to reason about directly, since we don't know its structure.
> - All three ultimately establish $Value(OS) \leq Value(GS)$ (maximize) or $Cost(GS) \leq Cost(OS)$ (minimize) for _every_ valid $OS$ — they just get there by different routes.
> - Not every technique applies to every problem — picking the right one is itself part of the skill.

---

# Exchange Argument (Modify-the-Solution)

Take an arbitrary solution $OS$ that skips the greedy algorithm's first choice $g$, and show how to _exchange_ one element of $OS$ for $g$ to build a new solution $OS'$ that is still valid and at least as good. Then induct on **instance size**. See [[Event Scheduling]] for the full worked example.

- **General steps:**
    1. State what's known: the definition of $g$, and that $OS$ meets the problem's constraints.
    2. Define $OS'$ from $OS$ and $g$ — usually by swapping $g$ in for one element of $OS$.
    3. Prove $OS'$ is still valid, using the definition of $g$.
    4. Compare $OS'$'s value/cost to $OS$'s.
    5. Induct: assume greedy is optimal for all instances smaller than $n$; show $|OS| \leq |OS'| \leq |GS|$ for the size-$n$ case using the exchange result on the first choice, then the inductive hypothesis on the rest.
- **Key detail:** inducts on **the size of the input**, not on the greedy algorithm's own sequence of choices — this is what distinguishes it from Greedy Stays Ahead below.
- **Core inequality:** $|OS(I)| \leq |OS'(I)| = |{g} \cup S(I')| \leq |{g} \cup GS(I')| = |GS(I)|$

---

# Greedy Stays Ahead

Instead of comparing just the first move, compare the **entire** greedy solution $GS$ against an entire arbitrary solution $OS$, using a running **progress measure** — showing $GS$ is always at least as far along as $OS$ at every step. Induct on the **greedy algorithm's own choices**, not the input size. See [[Event Scheduling]] for the full worked example.

- **General steps:**
    1. Define a progress measure (e.g. "finish time of the $i^{th}$ chosen event").
    2. Line up $OS$'s decisions with $GS$'s decisions in the same order.
    3. Prove by induction that $GS$'s progress after step $i$ is at least as good as $OS$'s.
    4. Assume by contradiction that $OS$ is strictly better than $GS$.
    5. Use the progress argument to derive a contradiction.
- **Key detail:** inducts on the **index of the choice being made** ($i$-th greedy pick vs. $i$-th choice in $OS$), not on shrinking the problem size.
- **Core inequality:** $Finish(G_i) \leq Finish(J_i)$ for all $i$ — if this held all the way through, $OS$ having _more_ events than $GS$ would force an event to start after $GS$'s last pick finishes, which is a contradiction.

---

# Achieves the Bound

Find a **bound** that (1) any valid solution must respect as a lower/upper limit, and (2) the greedy solution exactly reaches. This splits the proof into two separate, often-easier inequalities: $Cost(GS) \leq Bound \leq Cost(OS)$, which together force $Cost(GS) = Bound = $ optimal. See [[Event Scheduling with Multiple Rooms]] (Interval Partitioning) for the full worked example.

- **General steps:**
    1. Identify a quantity every valid solution is forced to respect (e.g. "at least this many rooms, since this many events overlap at once").
    2. Show that quantity is a genuine lower/upper bound for _any_ solution.
    3. Show the greedy solution's cost exactly equals that bound.
    4. Conclude $Cost(GS) = Bound \leq Cost(OS)$ (or the maximize analogue), so greedy is optimal.
- **Key detail:** does **not** work for all problems — it requires a bound to exist that greedy provably hits exactly. Also comes up outside greedy algorithms entirely, in approximation algorithms, LP, and network flow.
- **Core inequality:** $Cost(GS) = k \leq L \leq R = Cost(OS)$, where $L$ is the bound (e.g. max simultaneous overlap) and $k, R$ are greedy's and any-solution's actual costs.

---

# Quick Reference Table

|Technique|Compares|Inducts On|Best For|
|---|---|---|---|
|**Exchange Argument**|$GS$ vs. arbitrary $OS$, one element swapped|Instance size (strong induction)|Most general — try this first if unsure|
|**Greedy Stays Ahead**|$GS$ vs. arbitrary $OS$, step by step|The greedy algorithm's own sequence of choices|More intuitive when there's a natural running "progress" measure|
|**Achieves the Bound**|$GS$ vs. a bound vs. $OS$|N/A — no induction, uses a bounding argument instead|Only when a clean bound exists; also used in approximation algorithms, LP, network flow|

---

# References / Links

- [[Computer Science/Algorithms/Greedy Algorithms/index|Greedy Algorithms]]
- [[Event Scheduling]] — worked example for both Exchange Argument and Greedy Stays Ahead
- [[Event Scheduling with Multiple Rooms]] — worked example for Achieves the Bound (same problem as [[Event Scheduling with Multiple Rooms]] in this vault)
- [[Prove Kruskal's with Exchange Argument]]