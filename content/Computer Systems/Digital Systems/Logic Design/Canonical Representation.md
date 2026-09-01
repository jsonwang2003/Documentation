---
title: "Canonical Representation"
description: "Canonical representations of Boolean functions: Sum of Products (SOP / DNF), Product of Sums (POS / CNF), minterm and maxterm algebraic expansions, shorthand index notations, conversion rules, and algebraic simplification."
aliases:
  - Canonical Representation
  - Canonical Form
  - Sum of Products
  - Product of Sums
  - SOP and POS
  - Minterms and Maxterms
tags:
  - computer-systems
  - digital-systems
  - boolean-algebra
  - combinational-logic
---
> [!abstract] Abstract
> **Canonical Representation** provides a standardized, unambiguous algebraic expression for any Boolean switching function directly derived from its truth table. The two primary canonical forms are **Sum of Products (SOP / Disjunctive Normal Form)**—constructed from function 1s using **minterms** ($\Sigma m$)—and **Product of Sums (POS / Conjunctive Normal Form)**—constructed from function 0s using **maxterms** ($\Pi M$). While canonical expressions contain full literal expansions, they are rarely minimal and serve as the baseline input for algebraic and K-map logic optimization.

---

## The Need for Canonical Forms

Truth tables uniquely specify digital logic functions, but they become unwieldy as input count grows ($2^n$ rows). Canonical forms provide a standardized algebraic notation where every variable appears exactly once (in true or complemented form) in each term.

---

## Baseline Reference Truth Table

Consider the 3-variable Boolean function $F(A, B, C)$ and its complement $F'(A, B, C)$:

| Index ($i$) | $A$ | $B$ | $C$ | $F$ | $F'$ | Minterm ($m_i$) | Maxterm ($M_i$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | $0$ | $0$ | $0$ | **0** | **1** | $A'B'C'$ | $A + B + C$ |
| **1** | $0$ | $0$ | $1$ | **1** | **0** | $A'B'C$ | $A + B + C'$ |
| **2** | $0$ | $1$ | $0$ | **0** | **1** | $A'BC'$ | $A + B' + C$ |
| **3** | $0$ | $1$ | $1$ | **1** | **0** | $A'BC$ | $A + B' + C'$ |
| **4** | $1$ | $0$ | $0$ | **0** | **1** | $AB'C'$ | $A' + B + C$ |
| **5** | $1$ | $0$ | $1$ | **1** | **0** | $AB'C$ | $A' + B + C'$ |
| **6** | $1$ | $1$ | $0$ | **1** | **0** | $ABC'$ | $A' + B' + C$ |
| **7** | $1$ | $1$ | $1$ | **1** | **0** | $ABC$ | $A' + B' + C'$ |

---

## Sum of Products (SOP) / Disjunctive Normal Form

The **Sum of Products (SOP)** form expresses a function as a logical OR (sum) of product terms (**minterms**) for all input combinations where the output $F = 1$.

### Minterm Properties
* An ANDed product of literals where each input variable appears exactly once (true if variable is $1$, complemented if variable is $0$).
* Evaluates to $1$ for exactly **one** specific input combination.

### Canonical Expansion ($\Sigma m$)

Using the reference truth table where $F = 1$ at indices $\{1, 3, 5, 6, 7\}$:

$$\begin{aligned}
F(A, B, C) &= \Sigma m(1, 3, 5, 6, 7) \\
&= m_1 + m_3 + m_5 + m_6 + m_7 \\
&= A'B'C + A'BC + AB'C + ABC' + ABC
\end{aligned}$$

### Algebraic Simplification to Minimal SOP

Canonical expressions are usually non-minimal. Applying Boolean algebra theorems:

$$\begin{aligned}
F(A, B, C) &= A'B'C + A'BC + AB'C + ABC + ABC' \\
&= (A'B' + A'B + AB' + AB)C + ABC' && \quad \text{(Distributivity on $C$)} \\
&= \left((A' + A)(B' + B)\right)C + ABC' && \quad \text{(Factoring terms)} \\
&= (1 \cdot 1)C + ABC' && \quad \text{(Complementarity)} \\
&= C + ABC' && \quad \text{(Identity)} \\
&= C + AB && \quad \text{(Redundancy: } X + X'Y = X + Y\text{)} \\
F(A, B, C) &= \mathbf{AB + C}
\end{aligned}$$

---

## Product of Sums (POS) / Conjunctive Normal Form

The **Product of Sums (POS)** form expresses a function as a logical AND (product) of sum terms (**maxterms**) for all input combinations where the output $F = 0$.

### Maxterm Properties
* An ORed sum of literals where each variable appears exactly once (true if variable is $0$, complemented if variable is $1$).
* Evaluates to $0$ for exactly **one** specific input combination.

### Canonical Expansion ($\Pi M$)

Using the reference truth table where $F = 0$ at indices $\{0, 2, 4\}$:

$$\begin{aligned}
F(A, B, C) &= \Pi M(0, 2, 4) \\
&= M_0 \cdot M_2 \cdot M_4 \\
&= (A + B + C)(A + B' + C)(A' + B + C)
\end{aligned}$$

### Algebraic Simplification to Minimal POS

Applying Idempotency ($X = X \cdot X$) to duplicate $(A + B + C)$:

$$\begin{aligned}
F(A, B, C) &= \left[(A + B + C)(A + B' + C)\right] \cdot \left[(A + B + C)(A' + B + C)\right] \\
&= \left[(A + C) + BB'\right] \cdot \left[(B + C) + AA'\right] && \quad \text{(Distributivity)} \\
&= \left[(A + C) + 0\right] \cdot \left[(B + C) + 0\right] && \quad \text{(Complementarity)} \\
F(A, B, C) &= \mathbf{(A + C)(B + C)}
\end{aligned}$$

> [!note] Equivalence Check
> Expanding the minimal POS form yields: $(A + C)(B + C) = AB + AC + BC + C = AB + C(A + B + 1) = AB + C$, proving equivalence to the minimal SOP form.

---

## SOP vs. POS Selection Strategy

| Form | Focus | Primary Operator | Use Case Strategy |
|---|---|---|---|
| **SOP ($\Sigma m$)** | Tracks **1s** of the function | Outer **OR**, Inner **AND** | Preferred when $F = 1$ for **fewer** rows than $F = 0$. |
| **POS ($\Pi M$)** | Tracks **0s** of the function | Outer **AND**, Inner **OR** | Preferred when $F = 0$ for **fewer** rows than $F = 1$. |

---

## Mapping Between Canonical Forms

Converting between canonical notations relies on set complementation over the complete minterm/maxterm index universe $U = \{0, 1, \dots, 2^n - 1\}$:

| Conversion Type | Rule | Example ($n=3$ variables) |
|---|---|---|
| **SOP to POS ($F \to F$)** | Use Maxterm indices **absent** in Minterm list | $\Sigma m(1, 3, 5, 6, 7) \iff \Pi M(0, 2, 4)$ |
| **POS to SOP ($F \to F$)** | Use Minterm indices **absent** in Maxterm list | $\Pi M(0, 2, 4) \iff \Sigma m(1, 3, 5, 6, 7)$ |
| **SOP of $F$ to SOP of $F'$** | Select missing minterm indices | $F = \Sigma m(1, 3, 5, 6, 7) \implies F' = \Sigma m(0, 2, 4)$ |
| **POS of $F$ to POS of $F'$** | Select missing maxterm indices | $F = \Pi M(0, 2, 4) \implies F' = \Pi M(1, 3, 5, 6, 7)$ |

---

## Next Steps for Minimization

Canonical expansions are rarely optimal for hardware implementation because they require excessive gate inputs. To minimize gate count and propagation delay, canonical expressions are simplified using:

1. **Boolean Algebraic Theorems** (as demonstrated above).
2. **Karnaugh Maps (K-Maps)** (visual optimization using Gray code adjacency).
3. **Quine-McCluskey Algorithm** (algorithmic minimization for large variable sets).

---

## Related Notes

- [[Computer Systems/Digital Systems/Number Representation & Basic Logic Gates/Number Systems and Boolean Algebra|Number Systems and Boolean Algebra]]
- [[Computer Systems/Digital Systems/Combinational Logic Design|Combinational Logic Design]]
- [[Computer Systems/Digital Systems/SOP, POS, K-Maps & Logic Simplification|SOP, POS, K-Maps & Logic Simplification]]
- [[Computer Systems/Digital Systems/index|Digital Systems Index]]