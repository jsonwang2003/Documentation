---
title: "{{Filename - no extension}}"
description: "{{One-sentence summary of what this section covers}}"
tags:
aliases:
---
> [!abstract] Overview 
> {{One or two sentences on what this category covers and why it's grouped together — e.g. "Algorithms and data structures for traversing and analyzing graphs."}}

---

# Foundational Concepts

<!-- Definitions/notation shared by every note in this section, so individual notes don't have to re-define them. Keep this to things that are genuinely prerequisite to everything below, not specific to any one note. -->

## {{Concept 1, e.g. Graphs}}

{{Definition, notation, key properties.}}

## {{Concept 2, e.g. Graph Representations}}

{{Definition, comparison table if there are multiple representations/variants.}}

---

# Core Algorithm / Shared Building Block

<!-- Optional — only include if there's a generic algorithm or structure that several notes in this section are specific instances of (e.g. "Graph Search" is the generalization that DFS, BFS, and Dijkstra's each specialize). If nothing in this section shares a common ancestor, delete this whole block. -->

## {{Name}}

- **Instance:** {{input}}
- **Output:** {{output}}

### Pseudocode

```pseudo
	\begin{algorithm}
	\caption{ {{Name}} }
	\begin{algorithmic}
		\Procedure{ {{Name}} }{$G, s$}
			\State ...
		\EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

### Runtime Analysis

{{Derivation, e.g. summed over vertices/edges, ending in a boxed final bound.}}

### Correctness

{{Loop invariant / induction, following the same Initialization → Maintenance → Termination shape as the algorithm template.}}

---

# Notes in This Section

|Note|One-line description|
|---|---|
|[[{{Note A}}]]|{{what it specializes/adds relative to the core algorithm above}}|
|[[{{Note B}}]]|{{...}}|
|[[{{Note C}}]]|{{...}}|

---

# Related Categories

- [[{{Sibling Index A}}]]
- [[{{Sibling Index B}}]]