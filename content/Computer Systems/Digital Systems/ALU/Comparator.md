---
description: "Design of digital magnitude comparators: Bitwise XNOR N-bit Equality Comparators, Subtraction-based Less-Than Comparators, and relational operator derivation rules."
aliases:
  - Comparators
  - Equality Comparator
  - Less Than Comparator
  - Magnitude Comparator
tags:
  - computer-systems
  - digital-systems
  - alu
  - comparators
  - arithmetic-circuits
---
> [!abstract] Abstract
> **Magnitude Comparators** are combinational circuits that compare two $N$-bit binary numbers ($A$ and $B$) to determine their relative values. An **Equality Comparator** evaluates bitwise equivalence using **XNOR** gates combined through an **AND** tree (or by checking if $A - B = 0$). A **Less-Than Comparator ($A < B$)** leverages two's complement subtraction ($A - B$) and checks if the resulting sign bit (Most Significant Bit) indicates a negative value. By manipulating input order ($A \leftrightarrow B$) and negating outputs, all six standard relational operations ($=, \neq, <, \le, >, \ge$) can be synthesized from basic equality and less-than logic.

---

## 1. Equality Comparators ($A = B$)

An **Equality Comparator** evaluates to Logic $1$ if and only if every corresponding pair of bits in two $N$-bit inputs $A$ and $B$ are identical ($A_i = B_i$ for all $i \in \{0 \dots N-1\}$).

### Circuit Architecture

1. **Bitwise Comparison (XNOR Gate):** For each bit position $i$, an XNOR gate outputs $1$ if $A_i$ and $B_i$ match.
   $$x_i = A_i \text{ XNOR } B_i = A_i B_i + A_i' B_i' = \overline{A_i \oplus B_i}$$
2. **Global Equivalence (AND Tree):** The overall equality output $A_{eq}B$ is active only when all bitwise XNOR outputs are $1$.
   $$A_{eq}B = x_{N-1} \cdot x_{N-2} \cdots x_1 \cdot x_0 = \prod_{i=0}^{N-1} \overline{A_i \oplus B_i}$$

![[Pasted image 20260821145930.png]]
*N-Bit Equality Comparator using Bitwise XNOR Gates and an N-Input AND Gate*

### Alternative Design: Subtraction-Based Equality

Equality can also be evaluated using an **Adder-Subtractor** block:

1. Compute the difference $D = A - B$.
2. If $A = B$, then $D = 0$ (all difference bits $D_i = 0$).
3. Feed all difference bits $D_i$ into an **$N$-input NOR gate**:

$$A_{eq}B = \overline{D_{N-1} + D_{N-2} + \dots + D_0}$$

> [!note] Mux/Gate Swap Comparison
> Using an dedicated XNOR-AND tree requires fewer gate levels and has lower propagation delay compared to waiting for carry propagation through a full subtractor array.

---

## 2. Less-Than Comparators ($A < B$)

A **Less-Than Comparator** determines whether $A$ is strictly less than $B$.

### Subtraction-Based Implementation

To evaluate $A < B$, the circuit performs two's complement subtraction:

$$D = A - B = A + \overline{B} + 1$$

In two's complement representation:
* If $A \ge B$, the result $D$ is non-negative, and the Most Significant Bit (MSB / sign bit) is $0$ ($D_{N-1} = 0$).
* If $A < B$, the result $D$ is negative, and the Most Significant Bit (MSB / sign bit) is $1$ ($D_{N-1} = 1$).

![[Pasted image 20260821150101.png]]
*N-Bit Less-Than Comparator derived from Subtractor Sign Bit ($D_{N-1}$)*

> [!important] Overflow-Corrected Signed Comparisons
> For signed numbers where arithmetic overflow ($V$) can occur, the true less-than condition incorporates overflow correction:
>
> $$\text{Less Than } (A < B) = D_{N-1} \oplus V$$

---

## 3. Deriving All Relational Operators

All six standard relational operators can be constructed using an **Equality block ($A_{eq}B$)** and a **Less-Than block ($A_{lt}B$)** by swapping input ports ($A \leftrightarrow B$) or inverting output signals.

| Desired Relation | Hardware Expression | Derivation Rule |
|:---:|:---:|---|
| **Equal ($A = B$)** | $A_{eq}B$ | Direct Equality Output |
| **Not Equal ($A \neq B$)** | $\overline{A_{eq}B}$ | Invert Equality Output |
| **Less Than ($A < B$)** | $A_{lt}B$ | Direct Less-Than Output ($A - B < 0$) |
| **Greater Than or Equal ($A \ge B$)** | $\overline{A_{lt}B}$ | Invert Less-Than Output ($\text{NOT } [A < B]$) |
| **Greater Than ($A > B$)** | $B_{lt}A$ | Swap Input Ports ($B < A$) |
| **Less Than or Equal ($A \le B$)** | $\overline{B_{lt}A}$ | Invert Swapped Less-Than Output ($\text{NOT } [B < A]$) |

---

## Related Notes

- [[Mux & Demux]]
- [[Encoder & Decoder]]
- [[Adders & Subtractors|Adders]]
- [[Multiplier & Divider]]
- [[Computer Systems/Digital Systems/ALU/Arithmetic Logic Unit|Arithmetic Logic Unit Integration]]
- [[Computer Systems/Digital Systems/ALU/index|Arithmetic Logic Units Hub]]