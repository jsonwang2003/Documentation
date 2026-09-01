---
title: Mux, Demux
description: "Data steering building blocks: Transmission gate switches, Tristate buffers and buses, N-to-1 Multiplexers, general-purpose logic implementation via Muxes, and 1-to-N Demultiplexers."
aliases:
  - Mux and Demux
  - Multiplexers and Demultiplexers
  - Tristate Buffers
  - Mux Logic Implementation
tags:
  - computer-systems
  - digital-systems
  - alu
  - multiplexer
  - demultiplexer
  - combinational-logic
---
> [!abstract] Abstract
> **Multiplexers (Muxes)** and **Demultiplexers (Demuxes)** are the foundational data-steering components of digital architecture. Built from **Transmission Gates** and **Tristate Buffers** (which enable high-impedance $Z$ floating states for shared bus communication), a multiplexer selects one of $N = 2^n$ data inputs to route to a single output based on $n$ select control lines. Beyond data steering, $2^n:1$ multiplexers function as universal logic generators. Conversely, a demultiplexer directs a single input signal to one of $2^n$ outputs governed by control select inputs and enable signals.

---

## 1. Physical Building Blocks: Transmission Gates & Tristate Buffers

### Transmission Gate Switches
Because standalone nMOS switches pass $1$s poorly and standalone pMOS switches pass $0$s poorly, **Transmission Gates (TGs)** combine nMOS and pMOS transistors in parallel to pass full-rail $0$s and $1$s efficiently:

* **When $EN = 1$ ($EN' = 0$):** Switch is **ON**. Input $A$ is connected directly to output $B$.
* **When $EN = 0$ ($EN' = 1$):** Switch is **OFF**. Input $A$ is disconnected from $B$.

### Tristate Buffers & Shared Busses

When a switch is OFF, the output node enters a **High-Impedance ($Z$) / Floating / Open** state. 

Floating nodes allow multiple distinct drivers to connect to a single **Tristate Bus**, provided that **exactly one driver is active ($E = 1$)** at any given time.

![[Pasted image 20260810220408.png]]
*Tristate Buffer Symbol and Bus Architecture*

#### Tristate Buffer Truth Table

| Enable ($E$) | Input ($A$) | Output ($Y$) | State Description |
|:---:|:---:|:---:|---|
| $0$ | $0$ | **$Z$** | High Impedance (Disconnected) |
| $0$ | $1$ | **$Z$** | High Impedance (Disconnected) |
| $1$ | $0$ | **$0$** | Driven Low |
| $1$ | $1$ | **$1$** | Driven High |

---

## 2. Multiplexers (Mux)

A **Multiplexer** selects one of $N$ data inputs and routes it to a single output using $n = \log_2 N$ select control lines.

![[Pasted image 20260810220814.png]]
*2:1 Multiplexer Block Diagram and Internal Logic*

### 2:1 Multiplexer

* **Inputs:** Data $D_0, D_1$, Select $S$
* **Boolean Equation:** $Z = S' D_0 + S D_1$

#### 2:1 Mux Truth Table & State Mapping

| Select ($S$) | Input $D_0$ | Input $D_1$ | Output ($Y$) |
|:---:|:---:|:---:|:---:|
| $0$ | $0$ | $X$ | $0$ |
| $0$ | $1$ | $X$ | $1$ |
| $1$ | $X$ | $0$ | $0$ |
| $1$ | $X$ | $1$ | $1$ |

$$
\text{Compact Function: } Y = \begin{cases} 
D_0 & \text{if } S = 0 \\ 
D_1 & \text{if } S = 1 
\end{cases}
$$

### General $2^n:1$ Multiplexer Equations

For a $2^n:1$ Mux with $n$ select lines, the output is the sum of products of each minterm $m_k$ (formed by the select lines) and its corresponding data input $I_k$:

$$Z = \sum_{k=0}^{2^n - 1} m_k I_k$$

![[Pasted image 20260810235055.png]]
*General Minterm-based Multiplexer Structure*

#### 4:1 Multiplexer ($n = 2$, Selects $A, B$)
$$Z = A'B' I_0 + A'B I_1 + AB' I_2 + AB I_3$$

#### 8:1 Multiplexer ($n = 3$, Selects $A, B, C$)
$$Z = A'B'C' I_0 + A'B'C I_1 + A'BC' I_2 + A'BC I_3 + AB'C' I_4 + AB'C I_5 + ABC' I_6 + ABC I_7$$

---

## 3. Multiplexers as General-Purpose Logic Generators

A $2^n:1$ Mux can implement **any $(n+1)$-variable Boolean function** by connecting $n$ variables to the select lines and driving the data inputs $I_k$ with $0$, $1$, or the remaining variable (and its complement).

### Example Design Problem
Implement the 3-variable function $Z(A, B, C) = AC + BC' + A'B'C$ using a **4:1 Multiplexer** (Select lines connected to $A, B$).

### Method 1: Plug-and-Chug (Algebraic Evaluation)

Evaluate $Z(A, B, C)$ for each fixed combination of select variables $(A, B)$:

$$
\begin{aligned}
\text{For } (A, B) = (0, 0): \quad Z(0, 0, C) &= (0)C + (0)C' + (1)(1)C = C \implies \mathbf{I_0 = C} \\
\text{For } (A, B) = (0, 1): \quad Z(0, 1, C) &= (0)C + (1)C' + (1)(0)C = C' \implies \mathbf{I_1 = C'} \\
\text{For } (A, B) = (1, 0): \quad Z(1, 0, C) &= (1)C + (0)C' + (0)(1)C = C \implies \mathbf{I_2 = C} \\
\text{For } (A, B) = (1, 1): \quad Z(1, 1, C) &= (1)C + (1)C' + (0)(0)C = C + C' = 1 \implies \mathbf{I_3 = 1}
\end{aligned}
$$

### Method 2: K-Map Partitioning

Partition the 3-variable K-Map into sub-blocks corresponding to select states $AB \in \{00, 01, 10, 11\}$ and identify the output relationship relative to variable $C$:

![[Pasted image 20260819172331.png]]
*K-Map Partitioning for Mux Data Input Derivation*

---

## 4. Demultiplexers (Demux)

A **Demultiplexer** performs the inverse operation of a multiplexer: it routes a single data input line $X$ to one of $2^n$ output lines $Y_i$, selected by $n$ control lines $(S_{n-1}, \dots, S_0)$.

![[Pasted image 20260819172726.png]]
*1-to-N Demultiplexer Logic Diagram*

### Mathematical Function Definition

$$Y_i = \begin{cases} X & \text{if } i = (S_{n-1}, \dots, S_0)_2 \text{ and } EN = 1 \\ 0 & \text{otherwise} \end{cases}$$

> [!note] Enable Line Behavior
> If the enable line $EN = 0$, **all outputs $Y_i$ remain forced to $0$**, regardless of input $X$ or select line values.

---
## Related Notes

- [[Computer Systems/Digital Systems/Logic Design/Logic Functions|Logic Functions]]
- [[Adders & Subtractors]]
- [[Comparator]]
- [[Computer Systems/Digital Systems/ALU/Arithmetic Logic Unit|Arithmetic Logic Unit Integration]]
- [[Computer Systems/Digital Systems/ALU/index|Arithmetic Logic Units Hub]]