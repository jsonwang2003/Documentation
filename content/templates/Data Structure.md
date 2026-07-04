---
description: "{{One-sentence summary for search/graph previews}}"
aliases:
tags:
  - data-structures
---
> [!abstract] Abstract 
> {{One-sentence definition — e.g. "A Disjoint Set (Union-Find) tracks a partition of elements into disjoint subsets, supporting fast merge and membership queries."}}
> 
> - **Category:** {{e.g. Linear / Tree / Graph / Hash-based / Priority Structure}}
> - **Stores:** {{e.g. a partition of elements, ordered key-value pairs, a set of elements with priorities}}
> - **Built on top of:** {{e.g. arrays, linked lists, trees — if this structure is composed from a simpler one}}
> - **Typical use cases:** {{e.g. cycle detection in Kruskal's, priority scheduling, autocomplete}}

---

# Core Structure

<!-- How is it represented in memory/on paper? What's the key idea that makes it work? Aim for "how would I explain the shape of this to someone in 60 seconds". -->

{{Describe the representation — e.g. "Each element points to a parent; roots point to themselves and act as the name of their set."}}

> [!tip] Key Idea 
> {{The single "aha" idea that makes the structure work — e.g. "Attaching the shorter tree under the taller one keeps height logarithmic."}}

## Properties

- **Invariant(s):** {{e.g. "Every non-root node's key is ≤ its children's keys" (heap property)}}
- **Shape guarantee:** {{e.g. "Height is O(log n)" / "Always a complete binary tree"}}
- **Space complexity:** {{e.g. "O(n) for n elements, plus O(log n) recursion stack during operations"}}
- **What it does NOT guarantee:** {{e.g. "Not sorted overall" / "No guaranteed O(1) worst-case lookup"}}

## Why the Invariant Holds

<!-- The structure equivalent of "Proof of Correctness" — why does each operation preserve the invariant/shape guarantee above? Keep this proportional to how non-obvious it is; a one-line justification is fine if it's simple, a full induction if it's not (see the Disjoint Sets note for an example of the latter). -->

{{e.g. "Rotations preserve BST order because they only rearrange pointers among three nodes whose relative key ordering is fixed — the rotation just changes which one is the local root."}}

---

# Data Structure Operations

<!-- Main operations only — the ones that define the structure's interface. Pseudocode is optional; include it when the operation's logic isn't obvious from its name/complexity alone. -->

## `{{Operation1(args)}}`

{{What it does, in one or two sentences.}}

- **Time complexity:** {{e.g. O(log n) amortized}}
- **Notes:** {{e.g. any preconditions, side effects, or gotchas}}

```pseudo
	\begin{algorithm}
	\caption{ {{Operation1}} }
	\begin{algorithmic}
		\Procedure{ {{Operation1}} }{$x$}
			\State ...
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

## `{{Operation2(args)}}`

{{What it does, in one or two sentences.}}

- **Time complexity:** {{...}}
- **Notes:** {{...}}

## `{{Operation3(args)}}`

{{What it does, in one or two sentences.}}

- **Time complexity:** {{...}}
- **Notes:** {{...}}

---

# Common Pitfalls

<!-- Implementation-focused gotchas — edge cases and off-by-ones that trip people up when actually using/coding this structure. Different from Tradeoffs below, which is about choosing this structure over another. -->

- {{e.g. "Forgetting to handle the empty structure case — calling deletemin on an empty heap"}}
- {{e.g. "Off-by-one on child indices: children of a[i] are a[2i] and a[2i+1], not a[2i-1]/a[2i]"}}
- {{e.g. "Duplicate keys — does the structure allow them, and if so, how are ties broken?"}}

---

# Tradeoffs Compared to Other Data Structures

|Structure|{{Operation1}}|{{Operation2}}|{{Operation3}}|Notes|
|---|---|---|---|---|
|**{{This structure}}**|{{O(...)}}|{{O(...)}}|{{O(...)}}|{{what it's best at}}|
|{{Alternative A}}|{{O(...)}}|{{O(...)}}|{{O(...)}}|{{when to prefer this instead}}|
|{{Alternative B}}|{{O(...)}}|{{O(...)}}|{{O(...)}}|{{when to prefer this instead}}|

> [!note] When to reach for this structure {{e.g. "Use this over a plain array when you need frequent merges but rarely need to split a group back apart."}}

---

# Related Notes

**Algorithms that use this structure:**

- [[{{Algorithm A}}]] — {{how it uses this structure, e.g. "cycle detection via Find"}}
- [[{{Algorithm B}}]]

**Other structures built on top of this one:**

- [[{{Structure A}}]]

**Structures this one is built from:**

- [[{{Structure B}}]]