---
title: "Registers & Counters"
description: "Multi-bit storage registers, Serial-In/Serial-Out shift registers, combinational sequence pattern recognizers, 4:1 multiplexer-based universal shift registers, and state sequence counters."
aliases:
  - Registers & Counters
  - Registers and Counters
  - Shift Registers
  - Universal Shift Register
  - Pattern Recognizer
  - Synchronous Counters
tags:
  - computer-systems
  - digital-systems
  - sequential-logic
  - registers
  - counters
---
> [!abstract] Abstract
> **Registers and Counters** are multi-bit sequential logic structures constructed by grouping D Flip-Flops under a shared clock signal. A **Basic Register** stores an $N$-bit data word in parallel. A **Shift Register** connects flip-flops serially, propagating data bits sequentially per clock cycle. Connecting combinational decoding logic to shift register outputs forms a **Pattern Recognizer (Sequence Detector)**. Adding 4:1 multiplexers to each bit slice creates a **Universal Shift Register** capable of holding state, shifting left/right, or loading parallel inputs. **Counters** step through a predetermined sequence of binary patterns, repeating after a fixed modulus cycle count.

---

## 1. Basic Multi-Bit Registers

A **Register** is an array of $N$ flip-flops sharing a common clock signal that stores an $N$-bit binary word simultaneously.

![[Pasted image 20260825123624.png]]
*4-Bit Register Symbol*

### Parallel Architecture

Each bit of an incoming multi-bit bus ($D_3 \dots D_0$) is connected directly to the data input of an individual D Flip-Flop. On the active clock transition, all input bits are captured in parallel to update outputs ($Q_3 \dots Q_0$).

![[Pasted image 20260825124509.png]]
*Internal Schematic of a 4-Bit Register using Parallel D Flip-Flops*

---

## 2. Shift Registers

A **Shift Register** connects $N$ flip-flops in series such that the output of stage $i$ feeds the input of stage $i+1$ ($Q_i \to D_{i+1}$).

![[Pasted image 20260825124728.png]]
*4-Bit Serial Shift Register Schematic*

### Serial Bit Propagation

Because edge-triggered D Flip-Flops sample inputs strictly on clock transitions, a single input bit propagates rightward by **exactly one stage per clock cycle**.

#### Worked Example: Serial Propagation

Feeding the bit stream **`0110111`** (inserted right-to-left) into an initially zeroed 4-bit shift register ($Q_3 Q_2 Q_1 Q_0 = 0000_2$):

$$\begin{aligned}
\text{Initial State } (t_0): & \quad 0000_2 \\
\text{Cycle 1 } (t_1): & \quad 1000_2 \quad (\text{1 shifted in}) \\
\text{Cycle 2 } (t_2): & \quad 1100_2 \quad (\text{1 shifted in}) \\
\text{Cycle 3 } (t_3): & \quad 1110_2 \quad (\text{1 shifted in}) \\
\text{Cycle 4 } (t_4): & \quad 0111_2 \quad (\text{0 shifted in}) \\
\text{Cycle 5 } (t_5): & \quad 1011_2 \quad (\text{1 shifted in}) \\
\text{Cycle 6 } (t_6): & \quad 1101_2 \quad (\text{1 shifted in}) \\
\text{Cycle 7 } (t_7): & \quad 0110_2 \quad (\text{0 shifted in})
\end{aligned}$$

---

## 3. Pattern Recognizers (Sequence Detectors)

A **Pattern Recognizer** combines a shift register with combinational output decoding logic to detect when a specific sequence of binary bits has been received across consecutive clock cycles.

![[Pasted image 20260825125519.png]]
*Pattern Recognizer Circuit Detecting Target Sequence $1001_2$*

### Logic Decoding Function

The example circuit monitors four parallel tap points ($Q_3 Q_2 Q_1 Q_0$). When the captured bit pattern matches **`1001`** ($Q_3 = 1, Q_2 = 0, Q_1 = 0, Q_0 = 1$), the NAND decoding gate asserts an active-low flag:

$$\text{OUT} = \overline{Q_3 \cdot \overline{Q_2} \cdot \overline{Q_1} \cdot Q_0} = \begin{cases} 0 & \text{if } Q_3 Q_2 Q_1 Q_0 = 1001_2 \text{ (Match)} \\ 1 & \text{otherwise} \end{cases}$$

---

## 4. Universal Shift Registers

A **Universal Shift Register** combines multiple register modes—holding data, shifting right, shifting left, and parallel loading—into a single integrated module governed by select lines ($s_1, s_0$) and a clear control.

![[Pasted image 20260825125953.png]]
*4-Bit Universal Shift Register Block Symbol*

### Mode Control Selection Table

| Clear | $s_1$ | $s_0$ | New State Output ($Q_i^+$) | Operational Mode Description |
|:---:|:---:|:---:|:---:|---|
| $1$ | $X$ | $X$ | $0000_2$ | **Clear:** Asynchronously or synchronously resets register to 0. |
| $0$ | $0$ | $0$ | $Q_i$ | **Hold:** Retains current value (No change). |
| $0$ | $0$ | $1$ | $Q_{i+1}$ | **Shift Right:** Passes value from flip-flop to the left (serial in right). |
| $0$ | $1$ | $0$ | $Q_{i-1}$ | **Shift Left:** Passes value from flip-flop to the right (serial in left). |
| $0$ | $1$ | $1$ | $I_i$ | **Parallel Load:** Loads external input bus ($I_3 \dots I_0$). |

![[Pasted image 20260825130038.png]]
*Internal Bit-Slice Logic showing 4:1 Multiplexer driving D Flip-Flop Input*

---

## 5. Counters

A **Counter** is a specialized sequential state machine that cycles through a predefined sequence of binary states upon receiving clock pulses.

![[Pasted image 20260825130501.png]]
*Synchronous Sequence Counter Schematic*

### State Sequence Iteration

Counters repeat their fixed state pattern after completing $M$ clock cycles (where $M$ is the **modulus** of the counter).

#### Worked 8-State Cycle Example

Starting from initial pattern $1001_2$, the counter steps through eight unique states before recycling back to the initial state:

$$\begin{aligned}
\text{State 0 } (t_0): & \quad 1001_2 \\
\text{State 1 } (t_1): & \quad 0100_2 \\
\text{State 2 } (t_2): & \quad 1010_2 \\
\text{State 3 } (t_3): & \quad 1101_2 \\
\text{State 4 } (t_4): & \quad 0110_2 \\
\text{State 5 } (t_5): & \quad 1011_2 \\
\text{State 6 } (t_6): & \quad 0101_2 \\
\text{State 7 } (t_7): & \quad 0010_2 \\
\text{State 8 } (t_8): & \quad \mathbf{1001_2} \quad \text{(Sequence repeats)}
\end{aligned}$$

---

## Related Notes

- [[Computer Systems/Digital Systems/ALU/Mux & Demux|Mux & Demux]]
- [[Computer Systems/Digital Systems/Logic Design/Logic Functions|Logic Functions]]
- [[Computer Systems/Digital Systems/index|Digital Systems Index]]