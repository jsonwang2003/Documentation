---
description: Determine whether a string of letters (no spaces/punctuation) can be split into a sequence of valid English words, via a dynamic programming array of boolean subproblems.
tags:
  - algorithm
  - dynamic-programming
  - Examples
aliases:
  - String Reconstruction
  - Word Break
  - Word Segmentation
---
> [!abstract] 
>  Given a string of letters with no spaces or punctuation, how would you figure out how to separate it into words? (Determine if there is a way to separate it into a sequence of valid English words.)
> 
> - **Category:** Dynamic Programming / String Processing
> - **Input:** A string $x[1 \dots n]$ of letters, and (implicitly) a dictionary or oracle that can check whether a given substring is a valid word
> - **Output:** Whether a valid word-split exists, and if so, one such split
> - **Paradigm:** Dynamic Programming
> - **Typical use cases:** tokenization/NLP preprocessing, spell-checking and autocomplete, any "can this sequence be decomposed into valid pieces" problem

---

# Problem Specification

- **Instance:** A string $x[1\dots n]$ of letters, with no spaces or punctuation.
- **Solution Format:** A boolean — does a valid word-split exist — plus, if true, the sequence of break points recovering the actual split.
- **Constraints:** The substrings between consecutive break points must each individually be a valid dictionary word, and together must cover the entire string with no leftover characters.
- **Objective / Goal:** Like [[Selection]], this is a decision/existence problem rather than an optimization over many valid solutions — there's no "better" split to maximize, just "does at least one valid split exist."

---

# Candidate Strategies / Approaches

## Brute Force ✘

Try every possible way of placing word-breaks between the $n-1$ gaps in the string, then check whether every resulting piece is a valid word. There are $2^{n-1}$ ways to choose which gaps become breaks — exponential, and (just like [[Weighted Event Scheduling Example|Weighted Event Scheduling]]'s naive backtracking) this recomputes validity checks for the same substrings over and over across different candidate splits.

## Dynamic Programming ✔

Same insight as [[Weighted Event Scheduling Example|Weighted Event Scheduling]]: define a small number of genuinely distinct sub-problems — "can the prefix ending at position $k$ be validly split?" — and solve them smallest-first, reusing each answer instead of re-deriving it. See [[Computer Science/Algorithms/Dynamic Programming/index#The 8 Steps to Design a Dynamic Programming Algorithm|The 8 Steps]] for the general recipe this follows.

---

# Dynamic Programming Solution

## 1. Define the Array Values (Sub-Problems)

Let $S(k)$ be true if $x_1, \dots, x_k$ can be separated into a sequence of English words, and false otherwise.

## 2. Base Case

$$ S[0] = \text{True} $$

(The empty prefix is vacuously a valid — empty — sequence of words.)

## 3. Express Recursively

$$ 
S[k] = 
\begin{cases}  \\
\text{True} &\text{if } \exists j \text{ such that } S[j] = \text{True and } x_{j+1}, \dots, x_{k} \text{ is a word} \\ \\
 \text{False} &\text{otherwise} \\
 \end{cases} 
 $$

## 4. Order the Problems

$0 \dots n$ — each $S[k]$ only ever depends on some $S[j]$ with $j < k$, so solving in increasing order of $k$ guarantees every dependency is already computed.

## 5. Output

$$ S[n] $$

## 6. Iterative Algorithm

```pseudo
	\begin{algorithm}
	\caption{String Reconstruction}
	\begin{algorithmic}
	\Procedure{StringReconstruction}{$x[1\dots n]$}
		\State Initialize all $S[.]$ to be False and all $prev(.)$ to be $\emptyset$
		\State $S(0) = true$
		\For{$k$ from $1$ to $n$}
			\State $j = k-1$
			\While{not $S(k)$ and $j \geq 0$}
				\If{$S(j)$ is true and $x[j+1 \dots k]$ is a valid word}
					\State $S(k)$ = true
					\State $prev(k) = j$
				\Else
					\State $j = j-1$
                \EndIf
            \EndWhile
        \EndFor
        \If{$S(n)$}
	        \State $p = n$
	        \While{$p > 0$}
		        \State print($p$)
				\State $p = prev(p)$
            \EndWhile
        \EndIf
    \EndProcedure
	\end{algorithmic}
	\end{algorithm}
```

> [!note] Fixed an Off-by-One 
> The source pseudocode's inner loop condition was `while not S(k) and j > 0`, which means `j` is decremented down to (but never actually tests) `j = 0`. That's a real bug: `j = 0` is exactly the case "the entire prefix $x[1\dots k]$ is itself a single valid word," using the base case $S(0) = \text{True}$ — a case that must be checked (e.g. for $k$ being the length of the very first word in the string). Changed the condition to `j \geq 0` above so `j=0` is actually tested before the loop exits.

## Variables & Data Structures

|Name|Type|Purpose|
|---|---|---|
|`S[]`|Boolean array, size $n+1$|`S[k]` = whether the prefix $x[1\dots k]$ has a valid word-split|
|`prev[]`|Array, size $n+1$|`prev[k]` = the break point $j$ that produced `S[k] = True`, used to reconstruct the actual split|
|`j`|Index|Candidate previous break point, checked from $k-1$ down to $0$|

## Helper Functions / Operations Used

- **`x[j+1...k]` is a valid word** — a dictionary lookup; $O(1)$ if backed by a hash set, though extracting/hashing the substring itself costs $O(k-j)$.
- **Reconstruction via `prev`** — once $S(n)$ is known true, walk the `prev` pointers from $n$ back to $0$, printing each break point; at most $n$ hops, so $O(n)$.

---

# Proof of Correctness / Optimality

**Claim:** $S(k)$ is set to `True` if and only if $x_1, \dots, x_k$ can be validly split into a sequence of words.

- **Base case:** $S(0) = \text{True}$ — the empty prefix trivially has a valid (empty) split.
- **Inductive Hypothesis:** for all $j < k$, $S(j)$ is set correctly.
- **Inductive Step:** the algorithm sets $S(k) = \text{True}$ exactly when it finds some $j$ with $S(j) = \text{True}$ (correct, by the Inductive Hypothesis) and $x[j+1\dots k]$ a valid word. This matches the recursive definition in Step 3 directly: $S(k)$ should be true iff _some_ such $j$ exists. Since the (corrected) loop checks every $j$ from $k-1$ down to $0$ inclusive, it examines every possible split point — so it sets $S(k) = \text{True}$ if and only if a valid $j$ actually exists. $\blacksquare$

**Why the off-by-one mattered for correctness:** without checking $j=0$, the algorithm would incorrectly conclude $S(k) = \text{False}$ for any $k$ where the _only_ valid split has the entire prefix $x[1\dots k]$ as a single word — a real, not just cosmetic, correctness gap.

---

# Time & Space Complexity Analysis

## General Case

| |Complexity|Notes|
|---|---|---|
|Time|$O(n^2)$ to $O(n^3)$|Outer loop runs $n$ times; inner `while` loop checks up to $n$ candidate values of $j$; each check's word-validity lookup costs $O(1)$ with a precomputed hash set (giving $O(n^2)$ total) or $O(k-j)$ if the substring must be extracted/hashed fresh each time (giving $O(n^3)$ worst case)|
|Space|$O(n)$|`S[]` and `prev[]` are both size $n+1$|

## Best / Worst / Average Case

- **Best case:** $O(n)$ — if every prefix is only ever split at the immediately preceding position ($j=k-1$ always works first), each `while` loop exits after one check.
- **Worst case:** as above ($O(n^2)$–$O(n^3)$ depending on the word-lookup cost) — occurs when many candidate $j$ values must be tried before (or without) finding a valid split.
- **Average case:** depends heavily on the actual dictionary and input string; not meaningfully different from the worst case in general without further assumptions.

---

# Drawbacks / Constraints

- **Depends on an unspecified dictionary/oracle.** The algorithm assumes "is $x[j+1..k]$ a valid word" can be checked, but doesn't specify how the dictionary itself is represented or looked up — this is where the real-world implementation cost hides (see complexity table above).
- **Finds _one_ split, not all of them.** Since `prev[k]` stores only the first (highest) $j$ found, the algorithm can't enumerate every possible decomposition when a string is ambiguous (e.g. a string splittable multiple different ways) — it just proves _existence_ and recovers one witness.
- **No notion of "best" split.** If disambiguating between multiple valid splits matters (e.g. preferring fewer, longer words, or the most common words), this needs to become an optimization variant — assign each word a value/cost and maximize/minimize over valid splits, the same way [[Weighted Event Scheduling Example|Weighted Event Scheduling]] extends plain [[Event Scheduling]].

---

# References / Links

- [[Computer Science/Algorithms/Dynamic Programming/index|Dynamic Programming]]
- [[Weighted Event Scheduling Example|Weighted Event Scheduling]]
- [[Computer Science/Algorithms/Backtracking/index|Backtracking]]