---
title: "Arithmetic Logic Units (ALUs)"
description: "Master index for the ALU module covering Mux/Demux, Encoder/Decoder, Adders/Subtractors, Comparators, Shifters, Multipliers/Dividers, and full 32-bit ALU integration."
aliases:
  - ALU Index
  - Arithmetic Logic Units Hub
  - Digital Design ALUs
tags:
  - index
  - digital-systems
  - alu
  - arithmetic-circuits
  - combinational-logic
---
> [!abstract] Overview
> An **Arithmetic Logic Unit (ALU)** is the computational core of a computer processor, executing arithmetic operations (addition, subtraction, multiplication, division) and logical operations (AND, OR, shifts, comparisons). Operating as a combinational network governed by control signals, the ALU routes inputs through specialized datapath blocks to compute results and generate status flags ($Zero$, $Overflow$, $CarryOut$).

---

## Submodule Notes

- [[Computer Systems/Digital Systems/ALU/Mux & Demux|Mux & Demux]]  
  Transmission Gate switches, Tristate buffers, shared busses ($Z$ state), $2^n:1$ Multiplexers, general-purpose logic generation via Muxes, and $1:2^n$ Demultiplexers.

- [[Computer Systems/Digital Systems/ALU/Encoder & Decoder|Encoder & Decoder]]  
  Binary Encoders, Active flags ($A$), $N$-to-$2^N$ One-Hot Decoders, enable gating ($G/EN$), logic minterm equations, and memory address decoding applications.

- [[Computer Systems/Digital Systems/ALU/Adders & Subtractors|Adders & Subtractors]]  
  1-bit Half/Full Adders, Ripple-Carry Adders ($t_{ripple} = N \cdot t_{FA}$), Carry-Lookahead Adders ($P/G$ logic), Two's Complement subtraction, overflow detection, and unified Adder-Subtractor circuits.

- [[Computer Systems/Digital Systems/ALU/Comparator|Comparator]]  
  Bitwise XNOR Equality Comparators, subtraction-based magnitude comparison, Less-Than Comparators ($A < B$), and deriving all six relational operators ($=, \neq, <, \le, >, \ge$).

- [[Computer Systems/Digital Systems/ALU/Shifters|Shifters]]  
  Logical Shifters ($\ll, \gg$), Arithmetic Shifters ($\lll, \ggg$), Rotators ($\text{ROL}, \text{ROR}$), and Mux-based logarithmic Barrel Shifters for $O(1)$ multi-bit shifting.

- [[Computer Systems/Digital Systems/ALU/Multiplier & Divider|Multiplier & Divider]]  
  Partial product generation, combinational Array Multipliers (AND arrays + Full Adder grids), and iterative repeated subtraction division algorithms.

- [[Computer Systems/Digital Systems/ALU/Arithmetic Logic Unit|Arithmetic Logic Unit Integration]]  
  Building 1-bit ALU primitives (AND, OR, Full Adder, Mux) and cascading them into modular 32-bit CPU ALUs governed by central `ALUop` control signals.

---

## Related Submodules & Directories

- [[Computer Systems/Digital Systems/Logic Design/index|Logic Design & K-Maps]]
- [[Computer Systems/Digital Systems/Number Representation & Basic Logic Gates/index|Number Representation & Basic Logic Gates]]
- [[Computer Systems/Digital Systems/index|Digital Systems Main Index]]
- [[Computer Systems/index|Computer Systems]]