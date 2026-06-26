---
title: Algorithm Analysis
---
# Algorithm Analysis Toolkit

> [!ABSTRACT]
> 
> This toolkit provides the formal framework for measuring the "cost" of an algorithm. It focuses on asymptotic analysis—determining how execution time and memory requirements scale as the input size ($n$) grows.

---
## Foundational Metrics

_The core methods for assessing performance before and during execution._

- **[[Time Analysis]]**
    - **Definition:** Estimating the number of elementary operations (additions, assignments, etc.) an algorithm performs.
    - **Purpose:** Allows for hardware-independent performance comparisons.
- **[[Runtime of Algorithms]]**
    - **Focus:** Measuring the actual wall-clock time required for execution.
    - **Variables:** Covers how environment factors (CPU, RAM, Background tasks) can influence raw timing data.

---
## Formal Notation & Logic

_The mathematical language used to categorize growth rates and prove correctness._

- **[[Asymptotic Notation]]**
    - **Big-O ($O$):** The upper bound; the "worst-case" scenario for growth.
    - **Big-Omega ($\Omega$):** The lower bound; the "best-case" scenario.
    - **Big-Theta ($\Theta$):** The tight bound; where growth is exactly defined.
- **[[Loop Invariants]]**
    - **Definition:** A property of a program loop that is true before (and after) each iteration.
    - **Why it matters:** A critical tool for **formal verification**; used to mathematically prove that an algorithm will produce the correct result.

---
## Complexity Cheat Sheet

| **Notation**  | **Growth Rate** | **Typical Algorithm**      |
| ------------- | --------------- | -------------------------- |
| $O(1)$        | Constant        | Accessing an array element |
| $O(\log n)$   | Logarithmic     | Binary Search              |
| $O(n)$        | Linear          | Linear Search              |
| $O(n \log n)$ | Linearithmic    | Merge Sort                 |
| $O(n^2)$      | Quadratic       | Bubble Sort                |
