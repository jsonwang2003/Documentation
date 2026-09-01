---
title: "Combinational Logic Design"
description: "Converting Boolean equations to gate-level logic circuits, specifying logic functions via truth tables, designing Half and Full Adders, and core canonical definitions."
aliases:
  - Combinational Logic Design
  - Logic Functions
  - Half Adder and Full Adder
  - Minterms and Maxterms
tags:
  - computer-systems
  - digital-systems
  - combinational-logic
  - adders
  - boolean-logic
---
> [!abstract] Abstract
> **Combinational Logic Design** translates functional behavioral descriptions into physical gate-level circuits. Logic problems are specified using **Truth Tables** that define output behavior for every combination of input signals. Standard arithmetic building blocks such as **Half Adders** and **Full Adders** illustrate how multi-output truth tables synthesize into optimized Boolean expressions. Formal circuit analysis relies on fundamental canonical terms: **literals, implicants, implicates, minterms, and maxterms**.

---

## Converting Boolean Equations to Gate Circuits

To construct a circuit diagram from a Boolean expression:
1. Identify the primary output function.
2. Work backwards from the output gate toward the input variables.
3. Sub-expressions inside parentheses or under complement bars represent intermediate logic gate outputs.

### Example Construction

Given the Boolean expression:

$$F = a \cdot \overline{b + \overline{c}}$$

* **Output Gate:** An **AND** gate combining $a$ with the intermediate signal $\overline{b + \overline{c}}$.
* **Intermediate Logic:** A **NOR** gate operating on input $b$ and the inverted input $\overline{c}$ (**NOT** gate on $c$).

![[Pasted image 20260806204928.png]]
*Gate-level Circuit Diagram for $F = a \cdot \overline{b + \overline{c}}$*

---

## Specifying Logic Functions with Truth Tables

Truth tables provide a complete, non-ambiguous specification of a combinational circuit by listing all $2^n$ possible input state combinations and their corresponding outputs.

### 1. Half Adder

A **Half Adder** adds two 1-bit binary inputs ($a, b$) and produces a 1-bit Sum ($S$) and a Carry-out ($C_{out}$).

![[Pasted image 20260807232234.png]]
*Half Adder Logic Gate Implementation*

#### Half Adder Truth Table

| $a$ | $b$ | Carry ($C_{out}$) | Sum ($S$) |
|:---:|:---:|:---:|:---:|
| $0$ | $0$ | $0$ | $0$ |
| $0$ | $1$ | $0$ | $1$ |
| $1$ | $0$ | $0$ | $1$ |
| $1$ | $1$ | $1$ | $0$ |

#### Half Adder Boolean Equations

$$\begin{aligned}
\text{Sum}(a, b) &= a'b + ab' = a \oplus b \\
\text{Carry}(a, b) &= ab
\end{aligned}$$

---

### 2. Full Adder

A **Full Adder** adds three 1-bit binary inputs: two primary operands ($A, B$) and an incoming Carry-in ($C_{in}$) from a less significant bit position. It produces a Sum ($S$) and a Carry-out ($C_{out}$).

![[Pasted image 20260807232559.png]]
*Full Adder Circuit Schematic*

#### Full Adder Truth Table

| $C_{in}$ | $A$ | $B$ | Carry-out ($C_{out}$) | Sum ($S$) |
|:---:|:---:|:---:|:---:|:---:|
| $0$ | $0$ | $0$ | $0$ | $0$ |
| $0$ | $0$ | $1$ | $0$ | $1$ |
| $0$ | $1$ | $0$ | $0$ | $1$ |
| $0$ | $1$ | $1$ | $1$ | $0$ |
| $1$ | $0$ | $0$ | $0$ | $1$ |
| $1$ | $0$ | $1$ | $1$ | $0$ |
| $1$ | $1$ | $0$ | $1$ | $0$ |
| $1$ | $1$ | $1$ | $1$ | $1$ |

#### Full Adder Boolean Equations

$$\begin{aligned}
S &= A \oplus B \oplus C_{in} \\
C_{out} &= AB + AC_{in} + BC_{in} = AB + C_{in}(A + B)
\end{aligned}$$

---

## Canonical Definitions & Terminology

Understanding algebraic structure requires precise terminology for variables and expressions:

| Term | Definition | Examples |
|---|---|---|
| **Complement** | A variable negated with a bar or prime. | $A'$, $B'$, $\overline{C}$ |
| **Literal** | A single variable or its complement. | $A, A', B, B', C, C'$ |
| **Implicant** | A product (AND term) of one or more literals. | $ABC, A'C, BC, AC$ |
| **Implicate** | A sum (OR term) of one or more literals. | $(A+B+C), (A+C), (A+B)$ |
| **Minterm ($m_i$)** | A product (AND term) containing **every** input variable in true or complemented form. Evaluates to $1$ for exactly one input combination. | $ABC, A'BC, AB'C'$ |
| **Maxterm ($M_i$)** | A sum (OR term) containing **every** input variable in true or complemented form. Evaluates to $0$ for exactly one input combination. | $(A+B+C), (A'+B+C), (A'+B'+C)$ |

---

## Related Notes

- [[Computer Systems/Digital Systems/Number Representation & Basic Logic Gates/Number Systems and Boolean Algebra|Number Systems and Boolean Algebra]]
- [[Computer Systems/Digital Systems/Number Representation & Basic Logic Gates/Transistors & Gates|Transistors & Gates]]
- [[Computer Systems/Digital Systems/SOP, POS, K-Maps & Logic Simplification|SOP, POS, K-Maps & Logic Simplification]]
- [[Computer Systems/Digital Systems/Mux, Demux, Decoders, Adders|Mux, Demux, Decoders, Adders]]
- [[Computer Systems/Digital Systems/index|Digital Systems Index]]