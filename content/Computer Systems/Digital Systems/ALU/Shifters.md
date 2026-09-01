---
title: "Shifters"
description: "Bitwise shift and rotate operations: Logical Shifters, Arithmetic Shifters, Rotators, and Mux-based logarithmic Barrel Shifter hardware implementations."
aliases:
  - Shifters
  - Bit Shifters
  - Logical Shifter
  - Arithmetic Shifter
  - Rotator
  - Barrel Shifter
tags:
  - computer-systems
  - digital-systems
  - alu
  - shifters
  - barrel-shifter
  - combinational-logic
---
> [!abstract] Abstract
> **Shifters** are combinational or sequential hardware blocks used in ALUs for bit manipulation, multiplication/division by powers of 2, and field alignment. The three primary shift operations are **Logical Shifts** (filling vacated positions with zeros), **Arithmetic Shifts** (preserving signed integer values via MSB sign-extension on right shifts), and **Rotators** (circulating bits without signal loss). High-performance hardware implements variable-distance shifting in constant $O(1)$ clock time using multi-stage multiplexer networks known as **Barrel Shifters**.

---

## 1. Shift and Rotate Operations

Shifting moves the binary bits of an $N$-bit vector left or right by a specified distance $k$.

### 1. Logical Shifter

* **Behavior:** Shifts bits left or right, padding all vacated bit positions with **zeros ($0$)**.
* **Arithmetic Function:** Unsigned multiplication by $2^k$ (left shift) or division by $2^k$ (right shift).

#### Worked 5-Bit Examples (Input: $11001_2$)
* **Logical Shift Right by 2 ($11001_2 \gg 2$):** $\mathbf{00110_2}$
* **Logical Shift Left by 2 ($11001_2 \ll 2$):** $\mathbf{00100_2}$

### 2. Arithmetic Shifter

* **Behavior:** Performs bitwise shifts while preserving the two's complement sign of the number.
  * **Arithmetic Shift Right ($\ggg$):** Fills vacated high-order bits with the **old Most Significant Bit (MSB / sign bit)**.
  * **Arithmetic Shift Left ($\lll$):** Identical to logical shift left (fills low-order bits with $0$).
* **Arithmetic Function:** Signed division by $2^k$ (right shift) preserving negative sign bits.

#### Worked 5-Bit Examples (Input: $11001_2$, MSB = $1$)
* **Arithmetic Shift Right by 2 ($11001_2 \ggg 2$):** $\mathbf{11110_2}$ *(MSB $1$ replicated twice)*
* **Arithmetic Shift Left by 2 ($11001_2 \lll 2$):** $\mathbf{00100_2}$

### 3. Rotator (Circular Shift)

* **Behavior:** Shifts bits circularly so that bits shifted off one end wrap around and enter at the opposite end. No bits are discarded.

#### Worked 5-Bit Examples (Input: $11001_2$)
* **Rotate Right by 2 ($11001_2 \text{ ROR } 2$):** $\mathbf{01110_2}$ *(Last two bits $01_2$ wrapped to front)*
* **Rotate Left by 2 ($11001_2 \text{ ROL } 2$):** $\mathbf{00111_2}$ *(First two bits $11_2$ wrapped to end)*

## Operation Summary Table

| Operation | Symbol | Direction | Vacated Bit Fill Strategy | 5-Bit Example ($11001_2$, $k=2$) |
|---|:---:|:---:|---|:---:|
| **Logical Shift Right** | $\gg$ | Right | Zeros ($0$) | $00110_2$ |
| **Logical Shift Left** | $\ll$ | Left | Zeros ($0$) | $00100_2$ |
| **Arithmetic Shift Right** | $\ggg$ | Right | Sign Bit (Original MSB $b_{N-1}$) | $11110_2$ |
| **Arithmetic Shift Left** | $\lll$ | Left | Zeros ($0$) | $00100_2$ |
| **Rotate Right** | $\text{ROR}$ | Right | Wrapped Low-Order Bits | $01110_2$ |
| **Rotate Left** | $\text{ROL}$ | Left | Wrapped High-Order Bits | $00111_2$ |

---

## 2. General Shifter Hardware Architecture (Barrel Shifter)

Naive single-bit iterative shifters require $k$ clock cycles to shift by $k$ bits. A **Barrel Shifter** executes arbitrary $k$-bit shifts in a **single combinational delay pass** using a crossbar matrix or a logarithmic cascade of multiplexers.

![[Pasted image 20260821151436.png]]
*High-Level General Shifter Block Schematic*

### Logarithmic Multiplexer Array Design

For an $N$-bit shifter supporting shifts from $0$ to $N-1$ positions, the design uses **$\log_2 N$ sequential stages** of $2:1$ multiplexers:

1. **Stage 0 ($Shift_0$):** Shifts by **$2^0 = 1$ bit** or $0$ bits.
2. **Stage 1 ($Shift_1$):** Shifts by **$2^1 = 2$ bits** or $0$ bits.
3. **Stage 2 ($Shift_2$):** Shifts by **$2^2 = 4$ bits** or $0$ bits.
4. **Stage $m$ ($Shift_m$):** Shifts by **$2^m$ bits** or $0$ bits.

![[Pasted image 20260821151920.png]]
*4-Bit Barrel Shifter Array Implementation using 2:1 Multiplexers*

### Advantages of Logarithmic Barrel Shifters
* **Propagation Delay:** Grows logarithmically as $O(\log_2 N)$ gate levels rather than linearly $O(N)$.
* **Control Simplicity:** The shift amount $k$ in binary ($k_{m-1} \dots k_0$) directly drives the select lines of each multiplexer stage.

---
## Related Notes

- [[Mux & Demux]]
- [[Adders & Subtractors]]
- [[Comparator]]
- [[Computer Systems/Digital Systems/ALU/Arithmetic Logic Unit|Arithmetic Logic Unit Integration]]
- [[Computer Systems/Digital Systems/ALU/index|Arithmetic Logic Units Hub]]