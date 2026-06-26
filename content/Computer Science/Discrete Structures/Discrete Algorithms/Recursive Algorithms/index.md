---
title: Recursive Algorithms
---
> [!ABSTRACT]
> 
> While iterative algorithms focus on a step-by-step "state" change, Recursive Algorithms focus on the Big Picture: solving a complex problem by reducing it to a simpler version of itself. This folder explores the deep mathematical symmetry between Recursion (problem-solving) and Induction (formal verification).

---
## The Logic of Recursion
- **Iteration vs. Recursion**: Any iterative algorithm can be transformed into a recursive one. Recursion allows us to view the solution as a composition of sub-problems rather than a sequence of instructions.
- **The Anatomy of a Recursive Solution**:
    - **Base Case**: The simplest possible input (e.g., $n=1$ or an empty string) where the answer is known immediately.
    - **Recursive Step**: How to solve a problem of size $n$, assuming we already have the answer for a problem of size $n-1$ (or smaller).

---
## Knowledge Map

## [[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Recursive Proofs|Recursive Proof Strategies]]
- Establishes the formal link: **The base case of recursion is the base case of induction.**
- Defines the 3-step proof: (1) Express algorithm behavior, (2) Apply **Strong Inductive Hypothesis**, (3) Verify the result for size $k+1$.
- Highlights the difference between **Regular Induction** (using $n-1$) and **Strong Induction** (using any size strictly less than $n$).
- Essential for verifying that complex logic like "Tower of Hanoi" or "Pattern Counting" actually produces the correct result.
## [[Time Analysis For Recursion|Time Analysis for Recursion]]
- Focuses on constructing a recurrence relation $T(n)$ based on the algorithm's structure.
- Identifies three variables: Number of recursive calls, input size change ($n'$), and work done outside the call ($c$).
- Uses the **Unraveling Method** to substitute recursive terms until the pattern reaches the base case.
- Provides the framework to prove that recursive algorithms like `FindMax` or `PatternCount` scale linearly ($O(n)$).
## [[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Divide and Conquer/index|Divide and Conquer]]
- A high-level paradigm where problems are split into **independent** sub-parts.
- Covers algorithms where the input size is reduced geometrically (e.g., $n/2$) rather than linearly ($n-1$).
- Critical for achieving logarithmic or linearithmic time complexities ($O(\log n)$ or $O(n \log n)$).
- Bridges basic recursion with advanced efficiency gains used in Merge Sort and Binary Search.
## [[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Solving Recurrence Closed Form/index|Solving Recurrence Closed Form]]
- The mathematical "toolbox" used to simplify $T(n)$ expressions into standard Big-O notation.
- Includes the **Master Theorem** for rapid analysis and **HRRCC** for linear sequences like Fibonacci.
- Explores **Guess and Check** for non-standard patterns discovered during the "finding pattern" phase.
- Provides the final "Closed Form" answer to the complexity questions raised in Time Analysis.
## [[Computer Science/Discrete Structures/Discrete Algorithms/Recursive Algorithms/Stirling's Approximation|Stirling's Approximation]]
- Provides a high-level approximation for **factorials** ($n!$) as $n$ grows large.
- **The Formula**: $n! \approx \sqrt{2\pi n} \left(\frac{n}{e}\right)^n$.
- Essential for time analysis when a recurrence involves $n!$, often simplifying $\log(n!)$ to **$O(n \log n)$**.
- Used in the analysis of comparison-based sorting lower bounds and recursive permutations.

---
## Applied Examples (Summary)

### 1. Find Max (Linear Search)
- **The Big Picture**: The maximum of a list is simply the `MAXIMUM` of (the first element, the maximum of the rest of the list).
- **Proof**: Uses induction to show that if $M_1$ is correct for $k$, then $MAX(M_1, a_{k+1})$ must be correct for $k+1$.
- **Analysis**: $T(n) = T(n-1) + c \implies \mathbf{O(n)}$.
### 2. Counting a Pattern ("00" Occurrences)
- **The Big Picture**: To count "00" in a string, we check the first two bits and then recursively count the rest starting from the second bit.
- **Recursive Step**: If $b_1, b_2 = 0$, the result is $1 + \text{Recurse}(b_2 \dots b_n)$.
- **Analysis**: Unraveling shows that stepping through the string one bit at a time results in $\mathbf{O(n)}$ time complexity.