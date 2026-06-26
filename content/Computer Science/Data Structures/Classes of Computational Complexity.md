## 1. Defining the Complexity Classes

The note focuses on **Decision Problems** (problems with a Yes/No answer) and categorizes them into four primary classes based on how hard they are to solve or verify.

| **Class**       | **Definition**                       | **Key Characteristics**                                                                                                 |
| --------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| **P**           | **Polynomial Time**                  | Problems that can be **solved** in $O(n^c)$ time. These are considered "efficiently solvable."                          |
| **NP**          | **Nondeterministic Polynomial Time** | Problems where a proposed solution can be **verified** in $O(n^c)$ time. Note: $P \subseteq NP$.                        |
| **NP-Hard**     | **NP-Hard**                          | Problems that are at least as hard as the hardest problems in $NP$. Every problem in $NP$ can be reduced to these.      |
| **NP-Complete** | **The Intersection**                 | Problems that are both in **NP** and **NP-Hard**. They are the "hardest" problems in $NP$ to solve, but easy to verify. |

---
## 2. Practical Examples

### Class P: The Oldest Person Problem

Finding the oldest person in a list of $n$ people takes $O(n)$ time. Since $O(n)$ is a polynomial, this problem is in class **P**.

### Class NP: The Subset Sum Problem

Given a set of integers, find a non-empty subset that adds up to 0.

- **Solving it:** There is no known polynomial-time algorithm (finding the subset is hard).
    
- **Verifying it:** If someone _gives_ you a subset, you can simply add the numbers in $O(n)$ time to check if they equal 0. Because it's easy to check, it is in **NP**.
    

### NP-Complete: Boolean Satisfiability (SAT)

This asks if a set of variables can make a Boolean formula TRUE. It is the foundation of modern encryption.

The "Boolean Satisfiability Problem" (SAT) is the historical "patient zero" of **NP-Complete** problems. This means it is among the hardest problems in the class **NP** (Nondeterministic Polynomial time).

While it is simple to **verify** if a specific assignment of `TRUE/FALSE` satisfies a formula (like your $x = \text{TRUE}, y = \text{FALSE}$ example), there is currently no known algorithm to **find** that assignment in polynomial time for any arbitrary, complex formula.

> [!Question]- If someone were to find a polynomial-time solution to _any_ **NP-Hard** problem (not necessarily the "Boolean Satisfiability Problem"), what would be the repercussions to data encryption, if any?
> If someone found a polynomial-time solution to an $NP$-Hard problem, **encryption would break.** Malicious actors could decrypt sensitive data as easily as we currently verify a password.

---
## 3. The "P vs. NP" Problem

This is one of the greatest unsolved mysteries in Computer Science.
- **If $P = NP$:** Anything that can be verified quickly can also be solved quickly. This would imply that for many "hard" problems, an efficient solution exists and we just haven't found it yet.
- **If $P \neq NP$:** There are problems that are fundamentally harder to solve than they are to verify. This is what most computer scientists believe to be true.

---
## 4. Strategies for "Hard" Problems

If you encounter a problem that is $NP$-Complete (like the **Traveling Salesman Problem**), and you cannot simplify it into class **P**, you generally have two paths:

1. **Small Input Sizes:** If $n$ is very small, even an $O(2^n)$ or $O(n!)$ algorithm might finish in a reasonable amount of time.    
2. **Heuristics:** Develop a polynomial-time "good enough" algorithm. It won't guarantee the absolute best (optimal) solution, but it will get close enough for practical use.

---

> [!Note] Summary: 
> Complexity analysis saves you from wasting "hours, days, or even years" trying to find a perfect solution for a problem that is mathematically proven to be incredibly difficult.