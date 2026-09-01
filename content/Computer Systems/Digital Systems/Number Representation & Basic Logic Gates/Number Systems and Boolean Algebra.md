---
title: "Number Systems and Boolean Algebra"
description: "Foundational digital logic concepts: positional number encoding (binary, octal, hex), basic and derived logic gates, Boolean axioms, algebraic theorems, duality, and proof techniques."
aliases:
  - Number Systems and Boolean Algebra
  - Boolean Algebra
  - Logic Gates
  - Positional Number Systems
tags:
  - computer-systems
  - digital-systems
  - boolean-algebra
  - logic-gates
---
> [!abstract] Abstract
> **Number Systems and Boolean Algebra** form the mathematical foundation of digital system design. Positional number representations (Binary, Octal, Hexadecimal) allow compact abstractions for binary physical signals. **Boolean Algebra** defines operations over two-valued logic ($B = \{0, 1\}$) using primary operators ($\text{AND}, \text{OR}, \text{NOT}$) and derived logic gates ($\text{NAND}, \text{NOR}, \text{XOR}, \text{XNOR}$). Circuit analysis and simplification rely on core Boolean axioms, the Principle of Duality, algebraic theorems, and formal proof techniques.

---

## Positional Number Encoding

In positional number systems, each digit's position represents a specific power of the base (radix). A symbol in a given position indicates the quantity of that power:

$$\text{Value} = \sum_{i} d_i \times b^i$$

* **Base 10 (Decimal):** Uses symbols $\{0, 1, 2, 3, 4, 5, 6, 7, 8, 9\}$. Positions represent powers of 10.
* **Base 2 (Binary):** Uses symbols $\{0, 1\}$. Positions represent powers of 2. This matches physical two-state digital hardware (high/low voltage levels).
* **Base 16 (Hexadecimal):** Uses symbols $\{0 \dots 9, A \dots F\}$ (where $A=10 \dots F=15$). Functions as a compact human-readable shorthand for binary (each hex digit corresponds to exactly 4 binary bits / 1 nibble).
* **Base 8 (Octal):** Uses symbols $\{0 \dots 7\}$. Positions represent powers of 8 (each octal digit corresponds to 3 binary bits).

---

## Primary Logic Gates & Operations

Boolean algebra operates on a two-element set $B = \{0, 1\}$. Variables evaluate strictly to $0$ or $1$, and operators return results in $B$.

### Primary Operators

1. **Logical AND (Intersection):** Denoted by $\cdot$ or concatenation ($a \cdot b$ or $ab$). Returns $1$ if and only if all inputs are $1$.
2. **Logical OR (Union):** Denoted by $+$ ($a + b$). Returns $1$ if at least one input is $1$.
3. **Logical NOT (Complement):** Denoted by $a'$ or $\overline{a}$. Inverts the binary input value.
4. **Buffer:** Returns the input value unchanged ($f(a) = a$). Used in physical circuits for signal amplification and propagation delay.

| Inputs ($a, b$) | AND ($a \cdot b$) | OR ($a + b$) | NOT ($\overline{a}$) | Buffer ($a$) |
|:---:|:---:|:---:|:---:|:---:|
| $0, 0$ | $0$ | $0$ | $1$ | $0$ |
| $0, 1$ | $0$ | $1$ | $1$ | $0$ |
| $1, 0$ | $0$ | $1$ | $0$ | $1$ |
| $1, 1$ | $1$ | $1$ | $0$ | $1$ |

![[Pasted image 20260806152300.png]]
*AND Gate Symbol*

![[Pasted image 20260806152310.png]]
*OR Gate Symbol*

![[Pasted image 20260806151414.png]]
*NOT Gate (Inverter) Symbol*

---

## Derived Logic Gates

Derived gates combine primary operators to perform specialized logic functions:

* **NAND:** Inverted AND, expressed as $(a \cdot b)'$. Universal gate.
* **NOR:** Inverted OR, expressed as $(a + b)'$. Universal gate.
* **XOR (Exclusive-OR):** Returns $1$ when inputs differ ($a \oplus b = a'b + ab'$).
* **XNOR (Equivalence):** Inverted XOR, returns $1$ when inputs are identical ($(a \oplus b)' = ab + a'b'$).

![[Pasted image 20260806152330.png]]
*NAND Gate Symbol*

![[Pasted image 20260806152343.png]]
*NOR Gate Symbol*

![[Pasted image 20260806150753.png]]
*XOR Gate Symbol*

![[Pasted image 20260806152759.png]]
*XNOR Gate Symbol*

### Truth Table for Derived Gates

| $a$ | $b$ | NAND ($(ab)'$) | NOR ($(a+b)'$) | XOR ($a \oplus b$) | XNOR ($(a \oplus b)'$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $0$ | $0$ | $1$ | $1$ | $0$ | $1$ |
| $0$ | $1$ | $1$ | $0$ | $1$ | $0$ |
| $1$ | $0$ | $1$ | $0$ | $1$ | $0$ |
| $1$ | $1$ | $0$ | $0$ | $0$ | $1$ |

---

## Boolean Axioms and Theorems

### Fundamental Axioms

| Axiom Name | Primary Form | Dual Form |
|---|---|---|
| **Binary Field** | $B = 0 \text{ if } B \neq 1$ | $B = 1 \text{ if } B \neq 0$ |
| **NOT Operation** | $\overline{0} = 1$ | $\overline{1} = 0$ |
| **Identity Axiom** | $0 \cdot 0 = 0$ | $1 + 1 = 1$ |
| **Unit Axiom** | $1 \cdot 1 = 1$ | $0 + 0 = 0$ |
| **Null Axiom** | $0 \cdot 1 = 1 \cdot 0 = 0$ | $1 + 0 = 0 + 1 = 1$ |

### Single-Variable Theorems

| Theorem Name | Theorem | Dual Form |
|---|---|---|
| **Identity** | $B \cdot 1 = B$ | $B + 0 = B$ |
| **Null Element** | $B \cdot 0 = 0$ | $B + 1 = 1$ |
| **Idempotency** | $B \cdot B = B$ | $B + B = B$ |
| **Involution** | $\overline{\overline{B}} = B$ | — |
| **Complements** | $B \cdot \overline{B} = 0$ | $B + \overline{B} = 1$ |

### Multi-Variable Theorems

| Theorem Name | Primary Form | Dual Form |
|---|---|---|
| **Commutativity** | $B \cdot C = C \cdot B$ | $B + C = C + B$ |
| **Associativity** | $(B \cdot C) \cdot D = B \cdot (C \cdot D)$ | $(B + C) + D = B + (C + D)$ |
| **Distributivity** | $B + (C \cdot D) = (B + C) \cdot (B + D)$ | $B \cdot (C + D) = (B \cdot C) + (B \cdot D)$ |
| **Covering** | $B \cdot (B + C) = B$ | $B + (B \cdot C) = B$ |
| **Combining** | $(B \cdot C) + (B \cdot \overline{C}) = B$ | $(B + C) \cdot (B + \overline{C}) = B$ |
| **Consensus** | $(B \cdot C) + (\overline{B} \cdot D) + (C \cdot D) = B \cdot C + \overline{B} \cdot D$ | $(B + C) \cdot (\overline{B} + D) \cdot (C + D) = (B + C) \cdot (\overline{B} + D)$ |
| **De Morgan's Law** | $\overline{B_0 \cdot B_1 \cdot B_2 \dots} = \overline{B_0} + \overline{B_1} + \overline{B_2} \dots$ | $\overline{B_0 + B_1 + B_2 \dots} = \overline{B_0} \cdot \overline{B_1} \cdot \overline{B_2} \dots$ |

---

## Principle of Boolean Duality

The dual of a Boolean expression is obtained by swapping $\cdot$ with $+$, $0$ with $1$, and vice versa, while leaving all variables unchanged.

$$\text{General Duality: } f(X_1, X_2, \dots, X_n, 0, 1, +, \cdot) \iff f(X_1, X_2, \dots, X_n, 1, 0, \cdot, +)$$

> [!tip] Property of Duality
> If an algebraic theorem or equality is proven true, its **dual form is guaranteed to be true** without requiring a separate proof. Note that Duality is distinct from De Morgan's Theorem (Duality does not negate variables).

---

## Proof Techniques for Boolean Expressions

### 1. Proof by Perfect Induction (Truth Tables)

Exhaustively verifies that both sides of a Boolean expression evaluate to identical truth values across all possible input variable combinations.

**Example:** Prove De Morgan's Law $(X + Y)' = X' \cdot Y'$

| $X$ | $Y$ | $X'$ | $Y'$ | $(X + Y)$ | **$(X + Y)'$** | **$X' \cdot Y'$** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $0$ | $0$ | $1$ | $1$ | $0$ | **$1$** | **$1$** |
| $0$ | $1$ | $1$ | $0$ | $1$ | **$0$** | **$0$** |
| $1$ | $0$ | $0$ | $1$ | $1$ | **$0$** | **$0$** |
| $1$ | $1$ | $0$ | $0$ | $1$ | ** $0$** | **$0$** |

*Result:* Columns $(X + Y)'$ and $X' \cdot Y'$ are identical for all inputs, proving the identity.

---

### 2. Algebraic Proof

Applies axiomatic identities step-by-step to transform one side of an equation into the target expression.

**Example:** Prove the Combining Theorem $X \cdot Y + X \cdot Y' = X$

$$\begin{aligned}
X \cdot Y + X \cdot Y' &= X \cdot (Y + Y') && \quad \text{(Distributivity)} \\
&= X \cdot (1) && \quad \text{(Complementarity: } Y + Y' = 1\text{)} \\
&= X && \quad \text{(Identity: } X \cdot 1 = X\text{)}
\end{aligned}$$

---

## Related Notes

- [[Computer Systems/Digital Systems/Transistors & Gates|Transistors & Gates]]
- [[Computer Systems/Digital Systems/Combinational Logic Design|Combinational Logic Design]]
- [[Computer Systems/Digital Systems/SOP, POS, K-Maps & Logic Simplification|SOP, POS, K-Maps & Logic Simplification]]
- [[Computer Systems/Digital Systems/index|Digital Systems Index]]