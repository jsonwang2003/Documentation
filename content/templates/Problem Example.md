---
description: "{{One-sentence summary of the problem and its solution strategy}}"
aliases:
tags:
---


> [!abstract] 
> {{The motivating scenario, in plain English — e.g. "Suppose you are running a conference and have a collection of events, each with a start and end time, but only one room. Goal: schedule the most events possible without overlap."}}
> 
> - **Category:** {{e.g. Greedy / Interval Scheduling / Graph Optimization}}
> - **Input:** {{e.g. a collection of events, each with a start and finish time}}
> - **Output:** {{e.g. a subset of events}}
> - **Paradigm:** {{e.g. Greedy, Dynamic Programming, Divide and Conquer}}
> - **Typical use cases:** {{e.g. resource booking, single-machine job scheduling}}

---

# Problem Specification

<!-- The 4-5 part formal breakdown of what counts as a valid, optimal solution. This is what makes a "problem" note different from an "algorithm" note — the problem is defined independently of any particular way of solving it. -->

- **Instance:** {{the input, formally — e.g. a set ${(s_1,f_1), \dots, (s_n,f_n)}$}}
- **Solution Format:** {{what shape an answer takes — e.g. a subset of events, a sequence of edges}}
- **Constraints:** {{what makes a candidate solution valid at all — e.g. no two chosen events overlap}}
- **Objective:** {{the quantity being optimized, and its type — e.g. cardinality of the subset, $\in \mathbb{R}$}}
- **Goal:** {{Maximize / Minimize}}

---

# Candidate Strategies / Approaches

<!-- Optional but often the most valuable section for a "problem" note — explore more than one plausible strategy, and show which ones fail (with a concrete counterexample) before landing on the one that works. Delete this section if there's only one reasonable approach worth presenting. -->

- **{{Strategy 1 name}} ✘** — Counterexample: Solution {{X}}, Better Solution {{Y}}. {{Optional image/diagram embed}}
- **{{Strategy 2 name}} ✘** — Counterexample: {{...}}
- **{{Strategy 3 name}} ✔** — {{Why this one works, briefly. Full proof below.}}

> [!tip] Key Idea 
> {{The single "aha" that explains why the winning strategy works and the others don't — e.g. "picking the choice that constrains the future the least."}}

---

# Pseudocode (Chosen Approach)

```pseudo
	\begin{algorithm}
	\caption{ {{Problem Name}} }
	\begin{algorithmic}
		\Procedure{ {{ProcedureName}} }{$...$}
			\State ...
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`{{var}}`|{{type}}|{{purpose}}|

## Helper Functions / Operations Used

- **`{{helper}}`** — {{what it does, complexity}}

---

# Proof of Correctness / Optimality

<!-- For optimization problems, this is usually of the shape: "for every instance I, GS = greedy/chosen solution, OS = any other solution, show Value(OS) ≤ Value(GS) [maximize] or Cost(GS) ≤ Cost(OS) [minimize]." If the full proof lives in a separate technique-focused note, link out instead of duplicating it — state the obligation and which technique applies. -->

We need to show: for every instance $I$, letting $GS$ be the chosen algorithm's solution and $OS$ be _any other_ solution,

$$ \underbrace{ \boxed{Value(OS) \leq Value(GS)} }_{ \text{Maximize} } \text{ or } \underbrace{ \boxed{Cost(GS) \leq Cost(OS)} }_{ \text{Minimize} } $$

> [!Important] The Tricky Part {{Why OS is hard to reason about directly — usually because it's arbitrary and we know nothing about its structure.}}

{{Either the full proof, or a pointer to which technique applies and a link to the dedicated proof note — e.g. see [[Techniques to Prove Optimality]].}}

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|{{O(...)}}|{{what dominates — e.g. sorting}}|
|Space|{{O(...)}}|{{what's being stored}}|

## Best / Worst / Average Case

- **Best / Worst / Average case:** {{usually all the same order for problems requiring a full pass/sort regardless of input shape — state why if so.}}

---

# Drawbacks / Constraints

- **{{What breaks the chosen strategy}}** — {{e.g. "no longer optimal once events have different weights — that variant needs Dynamic Programming instead."}}
- **Preconditions:** {{assumptions baked into the problem setup — e.g. single resource, one room.}}
- **Not suitable for:** {{a related-but-different variant that needs a different technique.}}

---

# References / Links

- {{Related technique or algorithm notes}}
- {{Sibling index}}