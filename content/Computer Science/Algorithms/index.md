---
title: Algorithms
---
## What is an Algorithm?

> [!INFO] Definition
> A procedure for performing a computation broken into a well-specified steps
> - Input: $X$ instance
> - Output: $Y$ solution
> 
> Both $X$ and $Y$ should be finitely describable

> [!Note] Good Algorithm must produce **correct answer** in *reasonable* amount of **time** and **space** using **least energy**

---
## Hierarchy of Obviousness

![[Pasted image 20260109102601.png]]

1. **Obvious algorithms**: Implicit in the problem statement
	- Brute force
	- Exhaustive Search 
2. **Methodical algorithms**: Applying *general principles* and *paradigms* that improve algorithms for a wide variety of problems
3. **Clever algorithms**: Stretching the *general paradigms* in a way to best fit a particular problem
---
## Time for Arithmetic
The CPU is designed to process instructions on word sized inputs
- Inputs less than word size → Performed on the CPU in a single access
- Inputs greater than word size → Need to be broken down into word size chunks

|                | floating point   | $n < \text{wordsize}$ | $\text{arbitrary} \ n$ |
| -------------- | ---------------- | --------------------- | ---------------------- |
| Addition       | $\mathcal{O}(1)$ | $\mathcal{O}(1)$      | $\mathcal{O}(n)$       |
| Subtraction    | $\mathcal{O}(1)$ | $\mathcal{O}(1)$      | $\mathcal{O}(n)$       |
| Comparison     | $\mathcal{O}(1)$ | $\mathcal{O}(1)$      | $\mathcal{O}(n)$       |
| Multiplication | $\mathcal{O}(1)$ | $\mathcal{O}(1)$      | $\mathcal{O}(n^2)$     |
 
---
## Types of Algorithms
### [[Computer Science/Algorithms/Graph Algorithms/index|Graph Search]]
### [[Computer Science/Algorithms/Greedy Algorithms/index|Greedy Algorithm]]
### [[Computer Science/Algorithms/Divide and Conquer/index|Divide and Conquer]]
### [[Computer Science/Algorithms/Dynamic Programming/index|Dynamic Programming]]
### [[Computer Science/Algorithms/Iterative Improvement/index|Iterative Improvement]]

