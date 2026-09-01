---
title: "Adders & Subtractors"
description: "1-bit Half/Full Adders, multi-bit Ripple-Carry and Carry-Lookahead Adders, two's complement arithmetic, overflow detection methods, and integrated Adder-Subtractor circuits."
aliases:
  - Adders & Subtractors
  - Adders and Subtractors
  - Half Adder and Full Adder
  - Ripple-Carry Adder
  - Carry-Lookahead Adder
  - Binary Subtractor
  - Overflow Detection
tags:
  - computer-systems
  - digital-systems
  - alu
  - adders
  - subtractors
  - arithmetic-circuits
---
> [!abstract] Abstract
> **Adders and Subtractors** form the fundamental arithmetic engine of an Arithmetic Logic Unit (ALU). Basic 1-bit primitives—the **Half Adder** and **Full Adder**—can be cascaded into multi-bit architectures such as the linear **Ripple-Carry Adder (RCA)** or the high-speed **Carry-Lookahead Adder (CLA)**, which eliminates carry-propagation bottlenecking using **Generate ($G$)** and **Propagate ($P$)** logic. By leveraging **Two's Complement** arithmetic, subtraction is executed as addition ($A - B = A + \overline{B} + 1$), enabling unified, hardware-efficient **Adder-Subtractor** circuits with built-in **overflow detection**.

---

## 1. 1-Bit Adder Building Blocks

### Half Adder
A **Half Adder** adds two 1-bit inputs ($A, B$) and produces a Sum ($S$) and Carry-out ($C_{out}$). It cannot accept an incoming carry from a less significant bit position.

![[Pasted image 20260821141255.png]]
*Half Adder Block Schematic*

#### Half Adder Truth Table & Logic Equations

| $A$ | $B$ | $C_{out}$ | $S$ |
|:---:|:---:|:---:|:---:|
| $0$ | $0$ | $0$ | $0$ |
| $0$ | $1$ | $0$ | $1$ |
| $1$ | $0$ | $0$ | $1$ |
| $1$ | $1$ | $1$ | $0$ |

$$\begin{aligned}
S &= A \oplus B \\
C_{out} &= AB
\end{aligned}$$

### Full Adder
A **Full Adder** adds three 1-bit inputs: two operands ($A, B$) and an incoming Carry-in ($C_{in}$).

![[Pasted image 20260821141306.png]]
*Full Adder Circuit Diagram*

#### Full Adder Truth Table & Logic Equations

| $C_{in}$ | $A$ | $B$ | $C_{out}$ | $S$ |
|:---:|:---:|:---:|:---:|:---:|
| $0$ | $0$ | $0$ | $0$ | $0$ |
| $0$ | $0$ | $1$ | $0$ | $1$ |
| $0$ | $1$ | $0$ | $0$ | $1$ |
| $0$ | $1$ | $1$ | $1$ | $0$ |
| $1$ | $0$ | $0$ | $0$ | $1$ |
| $1$ | $0$ | $1$ | $1$ | $0$ |
| $1$ | $1$ | $0$ | $1$ | $0$ |
| $1$ | $1$ | $1$ | $1$ | $1$ |

$$\begin{aligned}
S &= A \oplus B \oplus C_{in} \\
C_{out} &= AB + AC_{in} + BC_{in} = AB + C_{in}(A \oplus B)
\end{aligned}$$

---

## 2. Multi-Bit Adders & Propagation Delay

![[Pasted image 20260821141422.png]]
*Multi-Bit Adder Symbol*

### Ripple-Carry Adder (RCA)

A **Ripple-Carry Adder** chains $N$ 1-bit Full Adders in series, passing $C_{out}$ from bit position $i$ directly into $C_{in}$ of bit position $i+1$.

![[Pasted image 20260821142904.png]]
*4-Bit Ripple-Carry Adder Chain*

![[Pasted image 20260821141709.png]]
*Carry Signal Ripple Delay Path*

> [!warning] Propagation Delay Bottleneck
> The worst-case propagation delay occurs when a carry generated at the least significant bit ($bit_0$) ripples through every stage to the most significant bit ($bit_{N-1}$). For an $N$-bit adder with Full Adder delay $t_{FA}$:
>
> $$t_{ripple} = N \cdot t_{FA}$$

### Carry-Lookahead Adder (CLA)

A **Carry-Lookahead Adder** speeds up addition by calculating carry signals in parallel using **Generate ($G$)** and **Propagate ($P$)** signals, avoiding sequential bit-by-bit carry ripple.

![[Pasted image 20260821142609.png]]
*Carry-Lookahead Adder Architecture*

#### Generate & Propagate Definitions

For any bit stage $i$:
* **Generate ($G_i$):** A carry is generated internally regardless of $C_i$.
  $$G_i = A_i B_i$$
* **Propagate ($P_i$):** An incoming carry $C_i$ will propagate to $C_{i+1}$.
  $$P_i = A_i \oplus B_i$$

The next carry $C_{i+1}$ is computed as:

$$C_{i+1} = G_i + P_i C_i$$

![[Pasted image 20260821142524.png]]
*Unrolled Carry Lookahead Logic*

#### Unrolled Carry Equations

Expanding the recursive carry relation eliminates intermediate carry dependencies:

$$\begin{aligned}
C_1 &= G_0 + C_0 P_0 \\
C_2 &= G_1 + G_0 P_1 + C_0 P_0 P_1 \\
C_3 &= G_2 + G_1 P_2 + G_0 P_1 P_2 + C_0 P_0 P_1 P_2 \\
C_4 &= G_3 + G_2 P_3 + G_1 P_2 P_3 + G_0 P_1 P_2 P_3 + C_0 P_0 P_1 P_2 P_3
\end{aligned}$$

#### Block Propagate and Generate ($k$-bit Blocks)

To construct wider adders hierarchically without massive fan-in gates, individual stages are grouped into $k$-bit blocks (e.g., 4-bit blocks):

$$\begin{aligned}
G_{3:0} &= G_3 + P_3(G_2 + P_2(G_1 + P_1 G_0)) \\
P_{3:0} &= P_3 P_2 P_1 P_0 \\
C_i &= G_{i:j} + P_{i:j} C_{in}
\end{aligned}$$

---

## 3. Two's Complement Arithmetic & Subtraction

In fixed-width binary systems, two's complement notation converts subtraction into an addition operation:

$$A - B = A + (-B) = A + \overline{B} + 1$$

### Two's Complement Negation Procedure ($N^*$)
To negate a binary number $N$:
1. Take the bitwise complement ($\overline{N}$).
2. Add $1$.

$$\text{Negation Formula: } N^* = -N = \overline{N} + 1$$

* **Example (+7 to -7 in 4 bits):** $7_{10} = 0111_2 \implies \overline{0111} + 1 = 1000_2 + 1 = 1001_2 \, (-7_{10})$
* **Example (-7 to +7 in 4 bits):** $-7_{10} = 1001_2 \implies \overline{1001} + 1 = 0110_2 + 1 = 0111_2 \, (+7_{10})$

#### Worked Subtraction Example ($4 - 7$)

$$\begin{aligned}
y &= 4 - 7 = 4 + (-7) \\
&= 0100_2 + 1001_2 \\
&= 1101_2 \quad (-3_{10})
\end{aligned}$$

---

## 4. Arithmetic Overflow Detection

Overflow occurs when an arithmetic operation produces a result that exceeds the representable range of the fixed $N$-bit container.

### Method 1: Sign Bit Anomaly Detection

Overflow occurs **if and only if** two numbers with the **same sign** are added, but produce a result with a **different sign**. Adding a positive and a negative number can never result in overflow.

$$\text{Overflow Logic Equation: } V = a_{N-1}' b_{N-1}' s_{N-1} + a_{N-1} b_{N-1} s_{N-1}'$$

![[Pasted image 20260821144433.png]]
*Overflow Circuit based on Sign-Bit Comparison*

### Method 2: Carry-In vs. Carry-Out XOR

A simpler hardware implementation detects overflow by taking the **XOR** of the carry into the sign bit ($C_{N-1}$) and the carry out of the sign bit ($C_N$):

$$V = C_{N-1} \oplus C_N = C_{N-1} C_N' + C_{N-1}' C_N$$

![[Pasted image 20260821144826.png]]
*Simplified Overflow Detection Circuit using $C_{in} \oplus C_{out}$ at MSB*

| $C_{N-1}$ (Carry Into MSB) | $C_N$ (Carry Out of MSB) | Overflow ($V$) | Condition Description |
|:---:|:---:|:---:|---|
| $0$ | $0$ | $0$ | No Overflow |
| $0$ | $1$ | $1$ | **Negative Overflow** (Sum wrapped into positive range) |
| $1$ | $0$ | $1$ | **Positive Overflow** (Sum wrapped into negative range) |
| $1$ | $1$ | $0$ | No Overflow |

---

## 5. Configurable Adder-Subtractor Circuit

By combining bitwise **XOR gates** with a multi-bit adder, a single circuit can perform both addition and subtraction governed by a select control signal ($Sel$ / $Sub$).

![[Pasted image 20260821144956.png]]
*Bitwise Conditional Inversion using XOR Gates*

![[Pasted image 20260821145319.png]]
*Integrated $N$-Bit Adder-Subtractor Circuit with Overflow Output*

### Operational Modes

1. **Addition Mode ($Sel = 0$):**
   * XOR gates pass $B$ unchanged ($B_i \oplus 0 = B_i$).
   * Carry-in $C_0 = 0$.
   * Output: $Y = A + B$.

2. **Subtraction Mode ($Sel = 1$):**
   * XOR gates invert $B$ ($B_i \oplus 1 = \overline{B}_i$).
   * Carry-in $C_0 = 1$.
   * Output: $Y = A + \overline{B} + 1 = A - B$.

---

## Related Notes

- [[Mux & Demux]] 
- [[Comparator]]
- [[Multiplier & Divider]]
- [[Computer Systems/Digital Systems/ALU/Arithmetic Logic Unit|Arithmetic Logic Unit Integration]]
- [[Computer Systems/Digital Systems/ALU/index|Arithmetic Logic Units Hub]]