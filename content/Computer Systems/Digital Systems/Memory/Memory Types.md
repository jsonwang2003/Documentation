---
title: "Memory Types"
description: "Functional breakdown of RAM vs. ROM vs. NVM, m x n array organization, word widening and address depth composition, and SRAM (6T) vs. DRAM (1T1C) cell mechanics."
aliases:
  - Memory Types
  - ROM vs RAM
  - Memory Composition
  - SRAM and DRAM Mechanics
tags:
  - computer-systems
  - digital-systems
  - memory-types
  - sram-dram
  - memory-composition
---
> [!abstract] Abstract
> Memory modules are classified across **ROM (Read-Only Memory)**, **RAM (Random-Access Memory)**, and **NVM (Non-Volatile Memory)**. This note details $m \times n$ array organization, memory expansion composition methods (widen words and deepen address spaces), square matrix topology, and physical cell mechanics across Register Files ($\approx 46\text{T}$), Static RAM ($6\text{T}$), and Dynamic RAM ($1\text{T}1\text{C}$).

---

## 1. $m \times n$ Array Organization & Signal Interface

![[Pasted image 20260829144430.png]]

A memory module of dimensions **$m \times n$** stores $m$ addressable words of $n$ bits each.

* **$m$ Words:** Total addressable memory locations ($m = 2^k$).
* **$k$ Address Lines:** Calculated via $k = \log_2 m$.
* **$n$ Data Lines:** Multi-bit word width ($Q_{n-1} \dots Q_0$).
* **Control Lines:** Read/Write ($\text{r/w}$) and Enable ($\text{CS}$ / Chip Select).

![[Pasted image 20260829144406.png]]

> [!example] $4096 \times 8$ Memory Module
> * **Capacity:** $4096 \times 8 = 32,768 \text{ bits}$ ($4 \text{ KB}$).
> * **Address Lines ($k$):** $\log_2(4096) = 12 \text{ lines } (A_0 \dots A_{11})$.
> * **Data Bus ($n$):** $8 \text{ bidirectional lines } (Q_0 \dots Q_7)$.

 ---

## 2. Memory Composition Techniques

### 1. Wider Words (Bit-Width Expansion)
To widen word size ($n$) while maintaining word count ($m$):
* Connect memory modules **side-by-side**.
* Connect address ($A$) and control ($\text{r/w}$, $\text{Enable}$) lines in parallel.
* **Concatenate data lines** to form a wider multi-bit bus ($1024 \times 8 \to 1024 \times 32$).

![[Pasted image 20260829144847.png]]
*Word Widening ($1024 \times 8 \to 1024 \times 32$)*

---

### 2. More Words (Address-Space Expansion)
To increase total addressable words ($m$) while maintaining word width ($n$):
* Stack memory modules **vertically**.
* Drive lower address bits ($A_0 \dots A_{k-1}$) to all chips in parallel.
* Use upper address bit(s) to drive a **decoder** asserting the target chip's **Enable** line ($1024 \times 8 \to 2048 \times 8$).

![[Pasted image 20260829145153.png]]
*Address Expansion ($1024 \times 8 \to 2048 \times 8$)*

---

## 3. Physical Storage Cell Comparison (RAM Spectrum)

| Property | Register File (FF) | Static RAM (SRAM) | Dynamic RAM (DRAM) |
|---|:---:|:---:|:---:|
| **Transistor Count / Cell** | $\approx 46\text{T}$ (Multi-port D-FF) | $\mathbf{6\text{T}}$ | $\mathbf{1\text{T} + 1\text{C}}$ (1 Transistor, 1 Capacitor) |
| **Access Latency** | $< 1\text{ ns}$ | $\approx 10\text{ ns}$ | $\approx 20\text{--}50\text{ ns}$ (+ Refresh overhead) |
| **Density & Area** | Lowest density / Largest area | Compact | **Extremely Compact / Highest Density** |
| **Volatility** | Volatile | Volatile | Volatile (Requires periodic refresh) |
| **Diagram** | ![[Pasted image 20260829154901.png]] | ![[Pasted image 20260829154913.png]] | ![[Pasted image 20260829154935.png]] |

---

## 4. RAM Internal Architecture & Cell Operations

RAM arrays are laid out in a **square matrix grid** to balance row (word line) and column (bit line) capacitance.

Logically the same as register file, but RAM has only 1 port; register file has two or more

### RAM vs. Register File
RAM is larger, stores more bits using a bit storage vs. FFs, and implemented on a chip in a square ― keeps longest wires (hence delay) short
![[Pasted image 20260829160642.png]]

### RAM Internal Structure
![[Pasted image 20260829160800.png]]
*Square Array Grid and Internal Decoder Layout*

Similar internal structure as register file
- Decoder enables appropriate word based on address inputs
- $rw$ controls whether cell is written or read
- Let's see what's inside each RAM cell

#### **SRAM 6T Mechanics:** 
Uses two cross-coupled inverters and two access transistors.

**Write**
- *word enable* input comes from decoder
- When $0$, value $d$ loops around inverters ― storing the bit
- When $1$, the *data* bit value enters the loop
	- *data* is the bit to be stored in this cell
	- *data'* enters on the other side
![[Pasted image 20260829161129.png]]

**Read**
- When $rw$ set to read, the RAM logic sets both *data* and *data'* to $1$
- The stored but $d$ will pull either the left bit or right bit down slightly below $1$
- Evaluated by **Sense Amplifiers** which detects which side is slightly pulled down by sensing voltage difference
![[Pasted image 20260829161315.png]]
#### **DRAM 1T1C Mechanics:** 
Stores charge on a large capacitor via a single pass transistor. Because charge leaks off over time, DRAM requires **periodic refresh cycles** (typically every $64\text{ ms}$).
- **write**: transistor conducts, data voltage level gets stored on top plate of capacitor
- **read**: just look at value of $d$

 ![[Pasted image 20260829161547.png]]

---

## Related Notes

- [[Computer Systems/Digital Systems/Memory/Memory Hierarchy|Memory Hierarchy]]
- [[Computer Systems/Digital Systems/Memory/Cache Design|Cache Design]]
- [[Computer Systems/Digital Systems/Memory/Non-Volatile Memory (NVM)|Non-Volatile Memory (NVM)]]
- [[Computer Systems/Digital Systems/Memory/index|Memory Index]]