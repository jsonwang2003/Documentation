---
title: "Theory of Computability Index"
description: "What can and can't be computed — automata, formal languages, Turing machines, decidability, and computational complexity." 
tags:
- CS/index
- CS/computability 
aliases: ["Theory of Computability", "Computability Theory"] 
---

> [!abstract] Overview 
> The theoretical limits of computation — what problems can be solved by an algorithm at all (decidability), and, for those that can, how efficiently (complexity). Builds directly on the logic and proof techniques from [[Discrete Structures Index]].

---

# Foundational Concepts

## Alphabets, Strings, and Languages

{{An alphabet is a finite set of symbols; a string is a finite sequence of symbols from an alphabet; a language is a set of strings — the common vocabulary every model below is defined in terms of.}}

## Models of Computation

{{Finite automata (DFA/NFA), pushdown automata, and Turing machines — increasingly powerful models, each capturing a different class of languages.}}

---

# Core Concept / Shared Building Block

## Decidability

A language (or problem) is **decidable** if some Turing machine halts on every input and correctly accepts/rejects it. Central open question format for this whole section: "is problem X decidable, and if so, how efficiently?"

- **Diagonalization** — the standard technique for proving a problem is _undecidable_ (e.g. the Halting Problem), by constructing an input that contradicts the existence of an assumed decider.
- **Reduction** — showing problem A is at least as hard as known-undecidable problem B, by reducing B to A (see [[Adapting Algorithms: Modification vs. Reduction]] for the same general technique applied to algorithm design rather than decidability proofs).

---

# Notes in This Section

|Note|One-line description|
|---|---|
|{{Finite Automata (DFA/NFA)}}|{{not yet written}}|
|{{Regular Languages & Regular Expressions}}|{{not yet written}}|
|{{Context-Free Grammars & Pushdown Automata}}|{{not yet written}}|
|{{Turing Machines}}|{{not yet written}}|
|{{The Halting Problem & Undecidability}}|{{not yet written}}|
|{{P, NP, and NP-Completeness}}|{{not yet written}}|

---

# Related Categories

- [[Discrete Structures Index]] — logic and proof techniques (especially diagonalization-style contradiction proofs) used throughout this section.
- [[Algorithms]] — complexity classes here formalize the runtime analysis done throughout that section.