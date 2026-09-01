---
title: "Encoder & Decoder"
description: "Combinational logic blocks for code conversion and address decoding: Binary Encoders, Priority Encoders, N-to-2^N One-hot Decoders, enable gating, logic equations, and memory address decoding applications."
aliases:
  - Encoder and Decoder
  - Encoders
  - Decoders
  - Address Decoders
  - Binary Encoders
tags:
  - computer-systems
  - digital-systems
  - alu
  - encoder
  - decoder
  - combinational-logic
---
> [!abstract] Abstract
> **Encoders** and **Decoders** are fundamental code-converting combinational circuits. A **Binary Encoder** compresses $2^N$ input lines into an $N$-bit binary code under the operational assumption that at most one input is active. Conversely, an **$N$-to-$2^N$ Decoder** expands an $N$-bit binary select input into $2^N$ **one-hot** active output lines governed by an enable signal ($G$ or $EN$). Decoders serve as the backbone of memory address decoding, device assertion, and minterm generation.

---

## 1. Binary Encoders

An **Encoder** performs the reverse operation of a decoder. It accepts $2^N$ input lines ($I_0, I_1, \dots, I_{2^N-1}$) and converts the active input index into an $N$-bit binary output code $(y_{n-1}, \dots, y_0)$.

![[Pasted image 20260819173328.png]]
*Binary Encoder Symbol and Internal Logic Structure*

### Functional Assumption & Active Output Flag
A basic binary encoder assumes that **at most one input $I_i$ is HIGH ($1$)** at any given time. An auxiliary **Active Flag ($A$)** output signals whether at least one input line is currently asserted.

### Mathematical Definition

For an enabled encoder ($EN = 1$):

$$(y_{n-1}, \dots, y_0) = \begin{cases} i & \text{if } I_i = 1 \text{ and } EN = 1 \\ 0 & \text{otherwise} \end{cases}$$

$$A = \begin{cases} 1 & \text{if } EN = 1 \text{ and } \exists \, i \text{ such that } I_i = 1 \\ 0 & \text{otherwise} \end{cases}$$

![[Pasted image 20260819173726.png]]
*Detailed Logic Schematics for a 4-to-2 Binary Encoder*

---

## 2. Decoders ($N \to 2^N$)

A **Decoder** decodes an $N$-bit binary input code (referred to as **Select inputs $S$**) to activate exactly one of $2^N$ output lines.

![[Pasted image 20260819173900.png]]
*N-to-2^N Decoder Block Diagram with Enable Line*

### One-Hot Output Behavior
When the enable signal is active ($EN = 1$ or $G = 1$), the decoder outputs operate in a **one-hot** configuration: exactly **one** output line is asserted HIGH ($1$), while all other $2^N - 1$ output lines remain LOW ($0$).

$$\text{Output } y_i = \begin{cases} 1 & \text{if } EN = 1 \text{ and } (S_{n-1}, \dots, S_0)_2 = i \\ 0 & \text{otherwise} \end{cases}$$

> [!note] Enable Line ($G$ / $EN$) Gating
> If $EN = 0$ (or $G = 0$), **all outputs are forced to $0$**, disabling the driven devices regardless of the select inputs.

![[Pasted image 20260819173913.png]]
*Logic Truth Table for Standard Decoders*

---

## 3. Decoder Logic Equations & Architectures

The enable signal $G$ acts as a product factor across all output minterm equations:

### 1:2 Decoder ($1$ Select, $2$ Outputs)
$$\begin{aligned}
Y_0 &= G \cdot S' \\
Y_1 &= G \cdot S
\end{aligned}$$

### 2:4 Decoder ($2$ Selects, $4$ Outputs)
$$\begin{aligned}
Y_0 &= G \cdot S_1' \cdot S_0' \\
Y_1 &= G \cdot S_1' \cdot S_0 \\
Y_2 &= G \cdot S_1 \cdot S_0' \\
Y_3 &= G \cdot S_1 \cdot S_0
\end{aligned}$$

### 3:8 Decoder ($3$ Selects, $8$ Outputs)
$$\begin{aligned}
Y_0 &= G \cdot S_2' \cdot S_1' \cdot S_0' \\
Y_1 &= G \cdot S_2' \cdot S_1' \cdot S_0 \\
Y_2 &= G \cdot S_2' \cdot S_1 \cdot S_0' \\
Y_3 &= G \cdot S_2' \cdot S_1 \cdot S_0 \\
Y_4 &= G \cdot S_2 \cdot S_1' \cdot S_0' \\
Y_5 &= G \cdot S_2 \cdot S_1' \cdot S_0 \\
Y_6 &= G \cdot S_2 \cdot S_1 \cdot S_0' \\
Y_7 &= G \cdot S_2 \cdot S_1 \cdot S_0
\end{aligned}$$

---

## 4. Decoder Applications

Decoders are fundamental building blocks in computer memory architectures and bus controllers:

![[Pasted image 20260821141015.png]]

1. **Memory Address Decoding:** Converts an $N$-bit memory address from the CPU bus into a single active word line select signal in RAM/ROM chips.
2. **Device Assertion / Chip Select (CS):** Routes control signals to assert specific peripheral devices connected to a shared system bus.
3. **Universal Minterm Generator:** Because each decoder output line generates a unique minterm $m_i = S_{n-1} \dots S_0$, any arbitrary $N$-variable Boolean function can be implemented by feeding the relevant decoder output lines into an **OR** gate.

---

## Related Notes

- [[Mux & Demux]]
- [[Adders & Subtractors|Adders]]
- [[Comparator]]
- [[Computer Systems/Digital Systems/ALU/Arithmetic Logic Unit|Arithmetic Logic Unit Integration]]
- [[Computer Systems/Digital Systems/ALU/index|Arithmetic Logic Units Hub]]