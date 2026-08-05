---
title: Discrete Algorithms
---
> [!ABSTRACT]
> 
> This repository contains a structured study of discrete algorithms, focusing on their design, formal verification via induction/loop invariants, and asymptotic efficiency.

---
## Foundational Toolkits
Before diving into specific algorithms, these core tools provide the mathematical and procedural framework for analysis:
- **[[Sum of an Arithmetic Series]]**: The essential mathematical tool for deriving closed-form expressions for nested loops (like Bubble and Selection Sort). It explains why $1 + 2 + \dots + n$ results in $O(n^2)$.
- **[[Algorithm Checklist]]**: Your step-by-step "source of truth" for analyzing any new algorithm. It ensures you never miss a step—from defining the input/output to proving correctness and extracting the recurrence.

---
## Knowledge Map
## [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Algorithm Analysis/index|Algorithm Analysis]]
- Focuses on the **foundational tools and notation** used to measure the "goodness" and efficiency of code.
- Covers **Asymptotic Notation** ($O, \Omega, \Theta$) for growth rates, **Loop Invariants** for iterative correctness, and general **Time Analysis**.
- Essential for establishing a rigorous mathematical baseline to compare different algorithmic approaches.
- Provides the framework for understanding how resource consumption scales with input size.
## [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Recursive Algorithms/index|Recursive Algorithms]]
- Explores the transition from simple iterative loops to **self-referential logic** and complex problem decomposition.
- Includes specialized sub-directories for **Divide and Conquer** paradigms and **Solving Recurrence Closed Form** (via Master Theorem, Unraveling, or HRRCC).
- Covers **Recursive Proofs** and mathematical approximations like **Stirling's Approximation** for factorial growth.
- Critical for solving problems that exhibit optimal substructure and overlapping subproblems.
## [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Searching/index|Searching]]
- A systematic study of **finding specific elements** within data sets under varying constraints.
- Contrasts **Linear Search** ($O(n)$) with the optimized **Binary Search** ($O(\log n)$) to demonstrate the power of data organization.
- Explores the **Lower Bounds for Searching**, providing the theoretical proof that comparison-based searching cannot beat logarithmic time.
- Serves as a practical application of how data properties (like being sorted) drastically change algorithmic complexity.
## [[Computer Science Theory/Discrete Structures/Discrete Algorithms/Sorting/index|Sorting]]
- Focuses on the **systematic organization of data** to enable faster downstream operations.
- Categorizes methods into **Iterative** $O(n^2)$ approaches (Bubble, Selection, and Insertion Sort) and links to recursive strategies like Merge Sort.
- Important for understanding the trade-offs between algorithm simplicity, memory overhead, and computational speed.
- Provides the operational logic for optimizing data retrieval and processing pipelines.

---
## Quick Complexity Reference

|**Class**|**Algorithm Example**|**Growth Rate**|
|---|---|---|
|**Logarithmic**|Binary Search|$O(\log n)$|
|**Linear**|Linear Search|$O(n)$|
|**Log-Linear**|Merge Sort|$O(n \log n)$|
|**Quadratic**|Selection Sort|$O(n^2)$|
|**Sub-Exponential**|Karatsuba Multiplication|$O(n^{1.58})$|

---
## Related Toolkits
- **[[Computer Science Theory/Discrete Structures/Counting/index|Counting]]**: Combinatorial foundations used to determine the size of search spaces and the number of possible permutations.
- **[[Computer Science Theory/Discrete Structures/Probability/index|Probability]]**: Essential for analyzing randomized algorithms and determining average-case complexity.